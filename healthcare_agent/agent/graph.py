"""LangGraph orchestration of the Agentic Healthcare Assistant.

    START -> safety -> context -> planner -> executor (loops once per step) -> synthesizer -> memory -> END

* safety      deterministic red-flag screen (emergencies are surfaced first)
* context     identifies the acting user, their dependents, the active patient
              (short-term memory) and recalls long-term memories (FAISS)
* planner     LLM goal decomposition -> validated Plan (tools + dependencies)
* executor    runs one tool per visit, injects patient id / patient context,
              skips steps whose dependencies failed
* synthesizer grounded response from tool results
* memory      writes interaction + extracted facts to long-term memory

Short-term memory: the checkpointer persists state (messages, active patient)
per ``thread_id`` so follow-ups like "also book a dietitian for him" work.
"""
from __future__ import annotations

import functools
import re
import time
from datetime import date
from typing import Annotated, Any, Callable, Iterator, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .. import db, memory, records
from ..context import Actor, current_actor
from ..llm import get_llm
from ..observability import current_run_id, log_memory, new_run_id
from ..prompts import MEMORY_EXTRACTION_SYSTEM
from ..tools import TOOLS, call_tool
from . import safety as safety_mod
from .planner import heuristic_plan, llm_plan, validate_plan
from .responder import summarize_outcome, synthesize


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    actor: dict
    run_id: str
    safety: dict
    context: dict
    plan: dict
    plan_mode: str
    repairs: list[str]
    step_index: int
    step_results: list[dict]
    active_patient: dict | None
    answer: str
    answer_mode: str
    trace: list[dict]


def _node(name: str) -> Callable:
    """Bind actor/run-id context vars for the node and append a trace event."""
    def deco(fn: Callable[[AgentState, Actor], dict]) -> Callable[[AgentState], dict]:
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict:
            actor = Actor(**state["actor"])
            tok_a, tok_r = current_actor.set(actor), current_run_id.set(state["run_id"])
            t0 = time.perf_counter()
            try:
                update = fn(state, actor)
            finally:
                current_actor.reset(tok_a)
                current_run_id.reset(tok_r)
            event = {"node": name, "ms": int((time.perf_counter() - t0) * 1000), "detail": update.pop("_detail", None)}
            update["trace"] = list(state.get("trace") or []) + [event]
            return update
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@_node("safety")
def safety_node(state: AgentState, actor: Actor) -> dict:
    res = safety_mod.screen(state["query"])
    return {"safety": res, "_detail": res}


@_node("context")
def context_node(state: AgentState, actor: Actor) -> dict:
    deps = records.get_dependents(actor.user_id)
    active = state.get("active_patient")
    subject_ids = [actor.user_id] + [d["patient_id"] for d in deps] + ([active["patient_id"]] if active else [])
    hits = memory.recall(state["query"], list(dict.fromkeys(subject_ids)), k=5)
    msgs = state.get("messages") or []
    conversation = [("user" if isinstance(m, HumanMessage) else "assistant", str(m.content)) for m in msgs[:-1]][-6:]
    log_memory("short_term", active["patient_id"] if active else None,
               {"thread_turns": len(msgs), "active_patient": active["name"] if active else None})
    ctx = {"dependents": [{k: d[k] for k in ("relation", "patient_id", "name", "age")} for d in deps],
           "active_patient": {k: active[k] for k in ("patient_id", "name")} if active else None,
           "memories": [h["text"][:250] for h in hits], "conversation": conversation}
    return {"context": ctx, "_detail": {"dependents": len(deps), "memories_recalled": len(hits),
                                        "active_patient": ctx["active_patient"], "prior_turns": len(conversation)}}


@_node("planner")
def planner_node(state: AgentState, actor: Actor) -> dict:
    ctx = state["context"]
    if get_llm().live:
        plan, mode = llm_plan(state["query"], actor, ctx)
    else:
        plan, mode = heuristic_plan(state["query"], actor, ctx), "offline"
    plan, repairs = validate_plan(plan, actor, ctx)
    if (state.get("safety") or {}).get("level") == "emergency":
        # Keep the reply focused on getting help; background reading can wait.
        dropped = [s for s in plan.steps if s.tool == "medical_info_search"]
        if dropped:
            plan.steps = [s for s in plan.steps if s.tool != "medical_info_search"]
            renumber = {s.step: i for i, s in enumerate(plan.steps, 1)}
            for s in plan.steps:
                s.step = renumber[s.step]
                s.depends_on = [renumber[d] for d in s.depends_on if d in renumber]
            repairs.append("emergency: deferred medical information search")
    p = plan.model_dump()
    return {"plan": p, "plan_mode": mode, "repairs": repairs, "step_index": 0, "step_results": [],
            "_detail": {"mode": mode, "steps": [f"{s['step']}. {s['tool']}" for s in p["steps"]], "repairs": repairs}}


def _patient_context(hist: dict) -> str:
    p = hist["patient"]
    return (f"{p['name']}, {p.get('age') or '?'}-year-old {(p.get('gender') or '').lower()}; "
            f"active diagnoses: {', '.join(hist['active_diagnoses']) or 'none'}; "
            f"current medications: {', '.join(hist['active_medications']) or 'none'}; "
            f"alerts: {'; '.join(a['text'] for a in hist['alerts'][:4]) or 'none'}")


@_node("executor")
def executor_node(state: AgentState, actor: Actor) -> dict:
    steps = state["plan"]["steps"]
    i = state["step_index"]
    step = steps[i]
    tool = step["tool"]
    spec = TOOLS[tool]
    results = list(state.get("step_results") or [])
    active = dict(state["active_patient"]) if state.get("active_patient") else None
    status_of = {r["step"]: r["status"] for r in results}
    args = {k: v for k, v in (step.get("args") or {}).items() if v not in (None, "", [])}
    if args.get("updates"):
        args["updates"] = [dict(u) for u in args["updates"]]

    record = {"step": step["step"], "goal": step["goal"], "tool": tool, "args": dict(args), "status": "skipped",
              "result": None, "reason": None, "ms": 0}
    failed = [d for d in step.get("depends_on", []) if status_of.get(d) != "success"]
    if failed:
        record["reason"] = f"depends on step(s) {failed} which did not succeed"
    elif spec.needs_patient and not active:
        record["reason"] = "no patient identified"
    else:
        if spec.needs_patient:
            args["patient_id"] = active["patient_id"]
        if tool == "medical_info_search" and active:
            args["patient_id"] = active["patient_id"]
            args["patient_context"] = active.get("context", "")
        t0 = time.perf_counter()
        res = call_tool(tool, args)
        record.update(args=args, result=res, status="success" if res.get("ok") else "failed",
                      ms=int((time.perf_counter() - t0) * 1000))
        if res.get("ok"):
            if tool in ("identify_patient", "register_patient"):
                active = dict(res["patient"])
            elif tool == "get_patient_history" and active:
                active["context"] = _patient_context(res)
    results.append(record)
    return {"step_results": results, "step_index": i + 1, "active_patient": active,
            "_detail": {"step": step["step"], "tool": tool, "status": record["status"],
                        "reason": record["reason"] or (record["result"] or {}).get("error")}}


@_node("synthesizer")
def synthesizer_node(state: AgentState, actor: Actor) -> dict:
    answer, mode = synthesize(state["query"], state.get("safety") or {}, state.get("plan") or {},
                              state.get("step_results") or [])
    return {"answer": answer, "answer_mode": mode, "_detail": {"mode": mode, "chars": len(answer)}}


class MemoryFacts(BaseModel):
    facts: list[str] = Field(default_factory=list)


_FACT_RE = re.compile(r"\b(?:has|have|had|is|suffers?|diagnosed|allergic|prefers?|takes|on medication|lives)\b", re.I)


def _heuristic_facts(query: str, actor: Actor) -> MemoryFacts:
    facts = []
    for sent in re.split(r"(?<=[.!?])\s+", query):
        s = sent.strip()
        if (_FACT_RE.search(s) and not s.endswith("?")
                and not re.match(r"(?i)(?:can|could|please|book|schedule|i want|i'd like|also)\b", s)):
            facts.append(f"Reported by {actor.name}: {s}")
    return MemoryFacts(facts=facts[:3])


@_node("memory")
def memory_node(state: AgentState, actor: Actor) -> dict:
    active = state.get("active_patient")
    results = state.get("step_results") or []
    written = []
    if active and any(r["status"] == "success" for r in results):
        outcome = summarize_outcome(results)
        note = f"{date.today().isoformat()}: {actor.name} ({actor.role}) asked: \"{state['query'][:180]}\". Outcome: {outcome}."
        memory.remember(active["patient_id"], note, kind="interaction")
        written.append(note)
        facts, _ = get_llm().structured("memory_extraction", MEMORY_EXTRACTION_SYSTEM,
                                        f"Speaker: {actor.name} ({actor.role}); patient: {active['name']}\nMessage: {state['query']}",
                                        MemoryFacts, fallback=lambda: _heuristic_facts(state["query"], actor))
        for f in facts.facts[:4]:
            memory.remember(active["patient_id"], f, kind="fact")
            written.append(f)
    return {"messages": [AIMessage(content=state.get("answer", ""))], "_detail": {"written": written}}


def _route(state: AgentState) -> str:
    return "executor" if state.get("step_index", 0) < len((state.get("plan") or {}).get("steps", [])) else "synthesizer"


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("safety", safety_node), ("context", context_node), ("planner", planner_node),
                     ("executor", executor_node), ("synthesizer", synthesizer_node), ("memory", memory_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "safety")
    g.add_edge("safety", "context")
    g.add_edge("context", "planner")
    g.add_conditional_edges("planner", _route, ["executor", "synthesizer"])
    g.add_conditional_edges("executor", _route, ["executor", "synthesizer"])
    g.add_edge("synthesizer", "memory")
    g.add_edge("memory", END)
    return g


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class HealthcareAgent:
    def __init__(self, checkpointer=None):
        self.graph = build_graph().compile(checkpointer=checkpointer or InMemorySaver())

    def stream(self, query: str, actor: Actor, thread_id: str = "default") -> Iterator[dict[str, Any]]:
        """Yields {"type": "node", "node", "update"} events, then {"type": "result", "result"}."""
        run_id = new_run_id()
        t0 = time.perf_counter()
        inputs: AgentState = {"messages": [HumanMessage(content=query)], "query": query, "actor": actor.to_dict(),
                              "run_id": run_id, "trace": [], "step_results": [], "step_index": 0, "plan": {},
                              "repairs": [], "answer": "", "safety": {}}
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        error = None
        try:
            for chunk in self.graph.stream(inputs, config, stream_mode="updates"):
                for node, update in chunk.items():
                    yield {"type": "node", "node": node, "update": update or {}}
        except Exception as exc:          # keep the UI alive; the failure is logged as a failed run
            error = f"{type(exc).__name__}: {exc}"
        state = self.graph.get_state(config).values
        results = state.get("step_results") or []
        ok_steps = sum(r["status"] == "success" for r in results)
        result = {
            "run_id": run_id, "query": query, "answer": state.get("answer") or f"Sorry, something went wrong: {error}",
            "plan": state.get("plan") or {}, "plan_mode": state.get("plan_mode"), "repairs": state.get("repairs", []),
            "safety": state.get("safety") or {}, "steps": results, "trace": state.get("trace") or [],
            "active_patient": state.get("active_patient"), "answer_mode": state.get("answer_mode"),
            "latency_ms": int((time.perf_counter() - t0) * 1000), "error": error,
            "success": error is None and bool(results) and ok_steps == len(results),
            "steps_ok": ok_steps, "steps_total": len(results),
        }
        db.insert("agent_runs", {
            "run_id": run_id, "ts": db.now_iso(), "session_id": thread_id, "user_id": actor.user_id, "query": query,
            "plan": db.to_json(result["plan"]), "trace": db.to_json({"trace": result["trace"], "steps": [
                {k: v for k, v in r.items() if k != "result"} | {"ok": (r.get("result") or {}).get("ok"),
                                                                 "error": (r.get("result") or {}).get("error")}
                for r in results], "repairs": result["repairs"], "error": error}),
            "answer": result["answer"], "success": int(result["success"]), "latency_ms": result["latency_ms"],
            "llm_mode": f"{get_llm().label} | plan={result['plan_mode']} answer={result['answer_mode']}",
        })
        yield {"type": "result", "result": result}

    def run(self, query: str, actor: Actor, thread_id: str = "default") -> dict[str, Any]:
        result = {}
        for ev in self.stream(query, actor, thread_id):
            if ev["type"] == "result":
                result = ev["result"]
        return result
