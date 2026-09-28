"""Agent memory traces, planning breakdowns and tool logs."""
import json

import pandas as pd
import streamlit as st

from healthcare_agent import memory
from ui.common import db, sidebar, status_badge

actor = sidebar()
st.title("🧠 Memory & logs")

tab_runs, tab_mem, tab_tools = st.tabs(["Agent runs & plans", "Memory", "Tool & LLM logs"])

with tab_runs:
    runs = db.query("SELECT run_id, ts, user_id, query, success, latency_ms, llm_mode FROM agent_runs ORDER BY ts DESC LIMIT 200")
    if not runs:
        st.info("No agent runs yet.")
    else:
        df = pd.DataFrame(runs)
        df["result"] = df["success"].map(lambda s: "✅" if s else "⚠️")
        st.dataframe(df[["ts", "result", "user_id", "query", "latency_ms", "llm_mode", "run_id"]], hide_index=True, width="stretch")
        run_id = st.selectbox("Inspect run", df["run_id"], format_func=lambda r: f"{r} - {df.set_index('run_id').loc[r, 'query'][:80]}")
        run = db.query_one("SELECT * FROM agent_runs WHERE run_id=?", (run_id,))
        plan, trace = json.loads(run["plan"] or "{}"), json.loads(run["trace"] or "{}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Planning breakdown")
            st.markdown(f"**Intent:** {plan.get('intent_summary', '-')}")
            for s in plan.get("steps", []):
                args = {k: v for k, v in s["args"].items() if v}
                st.markdown(f"{s['step']}. {s['goal']}  \n  `{s['tool']}({json.dumps(args)})`"
                            + (f" ← depends on {s['depends_on']}" if s["depends_on"] else ""))
            if trace.get("repairs"):
                st.caption("Repairs: " + "; ".join(trace["repairs"]))
            if plan.get("needs_clarification"):
                st.info(plan["needs_clarification"])
        with c2:
            st.markdown("#### Execution")
            for s in trace.get("steps", []):
                st.markdown(f"**{s['step']}. {s['tool']}** - {status_badge(s.get('ok'), s['status'] == 'skipped')} ({s.get('ms', 0)} ms)"
                            + (f"  \n  _{s.get('reason') or s.get('error')}_" if s.get("reason") or s.get("error") else ""))
            st.markdown("#### Graph path")
            st.dataframe(pd.DataFrame([{"node": t["node"], "ms": t["ms"], "detail": json.dumps(t.get("detail"), default=str)[:200]}
                                       for t in trace.get("trace", [])]), hide_index=True, width="stretch")
        st.markdown("#### Memory operations in this run")
        ev = db.query("SELECT ts, op, subject_id, detail FROM memory_events WHERE run_id=? ORDER BY id", (run_id,))
        if ev:
            st.dataframe(pd.DataFrame(ev), hide_index=True, width="stretch")
        else:
            st.caption("none")
        with st.expander("Final answer"):
            st.markdown(run["answer"])

with tab_mem:
    st.markdown("**Long-term memory** (SQLite + FAISS `patient_memory`). Written after each conversation; recalled by semantic search during context loading.")
    mems = memory.list_memories()
    if mems:
        st.dataframe(pd.DataFrame(mems)[["created_at", "name", "kind", "content"]], hide_index=True, width="stretch")
    q = st.text_input("Test semantic recall", placeholder="e.g. kidney function and potassium")
    if q:
        hits = memory.recall(q, k=6, min_score=0.0)
        st.dataframe(pd.DataFrame([{"score": round(h["score"], 3), "patient": h["metadata"].get("patient_name"),
                                    "kind": h["metadata"].get("kind"), "text": h["text"][:300]} for h in hits]),
                     hide_index=True, width="stretch")
    st.markdown("**Memory event trace** (recall / write / short_term)")
    ev = pd.DataFrame(db.query("SELECT ts, run_id, op, subject_id, detail FROM memory_events ORDER BY id DESC LIMIT 300"))
    if not ev.empty:
        ops = st.multiselect("Operation", sorted(ev["op"].unique()), default=sorted(ev["op"].unique()))
        st.dataframe(ev[ev["op"].isin(ops)], hide_index=True, width="stretch")

with tab_tools:
    logs = pd.DataFrame(db.query("SELECT ts, run_id, tool, success, latency_ms, error, args, result FROM tool_logs ORDER BY id DESC LIMIT 500"))
    if logs.empty:
        st.info("No tool calls logged yet.")
    else:
        f1, f2 = st.columns(2)
        tools = f1.multiselect("Tool", sorted(logs["tool"].unique()))
        outcome = f2.segmented_control("Outcome", ["all", "success", "failure"], default="all")
        view = logs[logs["tool"].isin(tools)] if tools else logs
        if outcome == "success":
            view = view[view["success"] == 1]
        elif outcome == "failure":
            view = view[view["success"] == 0]
        st.dataframe(view, hide_index=True, width="stretch")
    st.markdown("**LLM calls**")
    st.dataframe(pd.DataFrame(db.query("SELECT * FROM llm_calls ORDER BY id DESC LIMIT 200")), hide_index=True, width="stretch")
