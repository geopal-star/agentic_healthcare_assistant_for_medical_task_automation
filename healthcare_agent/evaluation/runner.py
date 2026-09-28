"""Evaluation harness (LLMOps).

Modules
  planner     tool-selection precision/recall/F1, exact plan match, specialty and patient-identification accuracy
  summary     history summaries vs reference summaries (semantic sim, ROUGE-L, token F1, QAEvalChain)
  medical_qa  RAG answers vs references (semantic sim, key-fact coverage, groundedness, QAEvalChain)
  retrieval   FAISS hit@k / MRR for patient memory and the medical knowledge base
  booking     booking success rate, constraint satisfaction, idempotency, race safety
  tools       per-tool success rate and latency from the tool logs (live operations)

    python -m healthcare_agent.evaluation.runner [--modules planner summary ...]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from typing import Any

from .. import db, records
from ..agent.planner import heuristic_plan, llm_plan, validate_plan
from ..context import Actor, current_actor
from ..domain import normalize_specialty, parse_date_window, time_window
from ..llm import get_llm
from ..observability import current_run_id
from ..seed import STAFF
from ..tools import TOOLS
from ..tools.medical_search import answer_question
from ..tools.patient import identify_patient, summarize_history
from ..tools.scheduling import _claim_slot, book_appointment, cancel_appointment
from ..vectorstore import get_store
from . import cases
from .metrics import (groundedness, keyword_coverage, percentile, qa_eval_chain_grades, rouge_l_recall,
                      semantic_similarity, set_prf, token_f1)

ALL_MODULES = ["planner", "summary", "medical_qa", "retrieval", "booking", "tools"]


def _actor(user_id: str, role: str) -> Actor:
    staff = next((s for s in STAFF if s["user_id"] == user_id), None)
    if staff:
        return Actor(user_id, staff["name"], staff["role"])
    p = records.get_patient(user_id)
    return Actor(user_id, p["name"] if p else user_id, role)


def _pid(name: str) -> str | None:
    m = records.find_patients(name, cutoff=0.95)
    return m[0]["patient_id"] if m else None


class Recorder:
    def __init__(self, eval_run_id: str):
        self.id = eval_run_id
        self.rows: list[dict] = []

    def add(self, module: str, case_id: str, metric: str, score: float, details: Any = None) -> None:
        row = {"eval_run_id": self.id, "ts": db.now_iso(), "module": module, "case_id": case_id,
               "metric": metric, "score": float(score), "details": db.to_json(details) if details is not None else None}
        self.rows.append(row)
        db.insert("eval_results", row)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, list[float]]] = {}
        for r in self.rows:
            out.setdefault(r["module"], {}).setdefault(r["metric"], []).append(r["score"])
        return {m: {k: round(sum(v) / len(v), 4) for k, v in mets.items()} for m, mets in out.items()}


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

def eval_planner(rec: Recorder, log) -> None:
    llm = get_llm()
    for i, (uid, role, query, gold_tools, gold_spec, gold_patient) in enumerate(cases.PLANNER_CASES, 1):
        actor = _actor(uid, role)
        ctx = {"dependents": [{k: d[k] for k in ("relation", "patient_id", "name", "age")} for d in records.get_dependents(uid)],
               "active_patient": None, "memories": [], "conversation": []}
        tok = current_actor.set(actor)
        try:
            plan, mode = llm_plan(query, actor, ctx) if llm.live else (heuristic_plan(query, actor, ctx), "offline")
            plan, _ = validate_plan(plan, actor, ctx)
            tools = {s.tool for s in plan.steps}
            p, r, f1 = set_prf(tools, gold_tools)
            cid = f"P{i:02d}"
            rec.add("planner", cid, "tool_precision", p)
            rec.add("planner", cid, "tool_recall", r)
            rec.add("planner", cid, "tool_f1", f1, {"query": query, "predicted": sorted(tools), "expected": sorted(gold_tools), "mode": mode})
            rec.add("planner", cid, "exact_match", float(tools == gold_tools))
            if gold_spec:
                book = next((s for s in plan.steps if s.tool == "book_appointment"), None)
                got = normalize_specialty(book.args.specialty or book.args.condition or "") if book else None
                rec.add("planner", cid, "specialty_accuracy", float(got == gold_spec), {"predicted": got, "expected": gold_spec})
            if gold_patient:
                ident = next((s for s in plan.steps if s.tool == "identify_patient"), None)
                res = identify_patient(**{k: v for k, v in ident.args.model_dump().items()
                                          if k in ("name", "relation", "age") and v}) if ident else {"ok": False}
                got = (res.get("patient") or {}).get("name") if res.get("ok") else None
                rec.add("planner", cid, "patient_id_accuracy", float(got == gold_patient), {"predicted": got, "expected": gold_patient})
            log(f"  planner {cid}: F1={f1:.2f} tools={sorted(tools)}")
        finally:
            current_actor.reset(tok)


def eval_summary(rec: Recorder, log) -> None:
    llm = get_llm()
    examples, preds, ids = [], [], []
    tok = current_actor.set(Actor("attendant-01", "Evaluator", "attendant"))
    try:
        for name, ref in cases.SUMMARY_CASES.items():
            pid = _pid(name)
            if not pid:
                continue
            res = summarize_history(pid)
            s = res["summary"]
            cid = f"S-{name.split()[0]}"
            rec.add("summary", cid, "semantic_similarity", semantic_similarity(s, ref), {"patient": name, "mode": res["mode"]})
            rec.add("summary", cid, "rouge_l_recall", rouge_l_recall(s, ref))
            rec.add("summary", cid, "token_f1", token_f1(s, ref))
            examples.append({"query": f"Summarize the medical history of {name}", "answer": ref})
            preds.append({"result": s})
            ids.append(cid)
            log(f"  summary {cid}: sim={semantic_similarity(s, ref):.3f}")
        if llm.live and examples:
            for cid, grade in zip(ids, qa_eval_chain_grades(llm.chat_model(), examples, preds)):
                rec.add("summary", cid, "qa_eval_correct", float(grade == "CORRECT"), {"grade": grade})
    finally:
        current_actor.reset(tok)


def eval_medical_qa(rec: Recorder, log) -> None:
    llm = get_llm()
    examples, preds, ids = [], [], []
    for i, (q, ref, kws) in enumerate(cases.MEDICAL_QA_CASES, 1):
        res = answer_question(q)
        cid = f"Q{i:02d}"
        ans = res.get("answer", "") if res.get("ok") else ""
        rec.add("medical_qa", cid, "answered", float(bool(ans)), {"question": q, "mode": res.get("mode"),
                                                                 "sources": [s["url"] for s in res.get("sources", [])]})
        rec.add("medical_qa", cid, "semantic_similarity", semantic_similarity(ans, ref) if ans else 0.0)
        rec.add("medical_qa", cid, "keyword_coverage", keyword_coverage(ans, kws))
        # Same key facts measured on the retrieved passages: separates retrieval misses from generation misses.
        rec.add("medical_qa", cid, "context_recall", keyword_coverage(" ".join(res.get("contexts", [])), kws))
        rec.add("medical_qa", cid, "groundedness", groundedness(ans, res.get("contexts", [])))
        rec.add("medical_qa", cid, "cited_sources", float(bool(re.search(r"\[\d+\]", ans))))
        examples.append({"query": q, "answer": ref})
        preds.append({"result": ans})
        ids.append(cid)
        log(f"  medical_qa {cid}: kw={keyword_coverage(ans, kws):.2f}")
    if llm.live and examples:
        for cid, grade in zip(ids, qa_eval_chain_grades(llm.chat_model(), examples, preds)):
            rec.add("medical_qa", cid, "qa_eval_correct", float(grade == "CORRECT"), {"grade": grade})


def eval_retrieval(rec: Recorder, log) -> None:
    store = get_store("patient_memory")
    for i, (q, name) in enumerate(cases.PATIENT_RETRIEVAL_CASES, 1):
        hits = store.search(q, k=5)
        names = [h["metadata"].get("patient_name") for h in hits]
        rank = next((j for j, n in enumerate(names, 1) if n == name), None)
        cid = f"R-P{i:02d}"
        rec.add("retrieval", cid, "patient_hit@1", float(rank == 1), {"query": q, "top": names[:3], "expected": name})
        rec.add("retrieval", cid, "patient_hit@3", float(bool(rank and rank <= 3)))
        rec.add("retrieval", cid, "patient_mrr", 1 / rank if rank else 0.0)
    kstore = get_store("medical_knowledge")
    for i, (q, pat) in enumerate(cases.KNOWLEDGE_RETRIEVAL_CASES, 1):
        hits = kstore.search(q, k=3)
        ok = any(re.search(pat, (h["metadata"]["url"] + h["metadata"]["title"]).lower()) for h in hits)
        rec.add("retrieval", f"R-K{i:02d}", "knowledge_hit@3", float(ok), {"query": q, "top": [h["metadata"]["title"] for h in hits]})
    log(f"  retrieval: {len(cases.PATIENT_RETRIEVAL_CASES)} patient + {len(cases.KNOWLEDGE_RETRIEVAL_CASES)} knowledge queries")


def eval_booking(rec: Recorder, log) -> None:
    tok = current_actor.set(Actor("attendant-01", "Evaluator", "attendant"))
    created: list[str] = []
    try:
        for i, (name, spec, date_hint, tod, expect_ok) in enumerate(cases.BOOKING_CASES, 1):
            pid = _pid(name)
            cid = f"B{i:02d}"
            res = book_appointment(patient_id=pid, specialty=spec, preferred_date=date_hint, time_of_day=tod,
                                   reason="evaluation")
            ok = bool(res.get("ok"))
            rec.add("booking", cid, "outcome_as_expected", float(ok == expect_ok),
                    {"patient": name, "specialty": spec, "date": date_hint, "time": tod, "ok": ok,
                     "error": res.get("error"), "alternatives": len(res.get("alternatives") or [])})
            if ok:
                a = res["appointment"]
                if not res.get("already_booked"):
                    created.append(a["appointment_id"])
                start = datetime.fromisoformat(a["start_ts"])
                d0, d1 = parse_date_window(date_hint)
                h0, h1 = time_window(tod)
                satisfied = (a["specialty"] == spec and (d0 is None or d0 <= start.date() <= d1) and h0 <= start.hour < h1)
                rec.add("booking", cid, "constraint_satisfaction", float(satisfied), {"start": a["start_ts"], "doctor": a["doctor_name"]})
                # Idempotency: the same request again must not create a second appointment.
                again = book_appointment(patient_id=pid, specialty=spec, preferred_date=date_hint, time_of_day=tod)
                rec.add("booking", cid, "idempotent", float(bool(again.get("already_booked"))))
            elif not expect_ok:
                rec.add("booking", cid, "offers_alternatives", float(bool(res.get("alternatives"))))
        # Race safety: two claims on one slot -> exactly one wins.
        slot = db.query_one("""SELECT s.slot_id, s.start_ts, d.doctor_id, d.name AS doctor_name, d.specialty, d.location
                               FROM slots s JOIN doctors d ON d.doctor_id=s.doctor_id
                               WHERE s.status='available' AND s.start_ts > datetime('now', '+1 day') LIMIT 1""")
        if slot:
            pid = _pid("Rahul Negi")
            first, second = _claim_slot(slot, pid, "race-test"), _claim_slot(slot, pid, "race-test")
            rec.add("booking", "B-race", "race_safe", float(bool(first) and second is None))
            if first:
                created.append(first["appointment_id"])
    finally:
        for appt_id in created:                    # leave the calendar as we found it
            cancel_appointment(appointment_id=appt_id)
        db.execute("DELETE FROM appointments WHERE reason IN ('evaluation', 'race-test')")
        current_actor.reset(tok)
    log(f"  booking: {len(cases.BOOKING_CASES)} scenarios (+ idempotency and race checks), cleaned up {len(created)}")


def eval_tools(rec: Recorder, log) -> None:
    """Operational metrics from real agent usage (excludes evaluation traffic)."""
    rows = db.query("SELECT tool, success, latency_ms FROM tool_logs WHERE run_id IS NULL OR run_id NOT LIKE 'eval-%'")
    by_tool: dict[str, list[dict]] = {}
    for r in rows:
        by_tool.setdefault(r["tool"], []).append(r)
    for tool, rs in sorted(by_tool.items()):
        lat = [r["latency_ms"] for r in rs]
        rec.add("tools", tool, "success_rate", sum(r["success"] for r in rs) / len(rs),
                {"calls": len(rs), "module": TOOLS[tool].module if tool in TOOLS else "?",
                 "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95)})
    runs = db.query("SELECT success, latency_ms FROM agent_runs")
    if runs:
        rec.add("tools", "agent", "run_success_rate", sum(r["success"] for r in runs) / len(runs),
                {"runs": len(runs), "p50_ms": percentile([r["latency_ms"] for r in runs], 50)})
    llm_rows = db.query("SELECT mode FROM llm_calls")
    if llm_rows:
        rec.add("tools", "llm", "live_call_share", sum(r["mode"] == "live" for r in llm_rows) / len(llm_rows),
                {"calls": len(llm_rows), "fallbacks": sum(r["mode"] == "fallback" for r in llm_rows)})
    log(f"  tools: {len(by_tool)} tools, {len(rows)} logged calls")


RUNNERS = {"planner": eval_planner, "summary": eval_summary, "medical_qa": eval_medical_qa,
           "retrieval": eval_retrieval, "booking": eval_booking, "tools": eval_tools}


def run_evaluation(modules: list[str] | None = None, verbose: bool = True) -> dict[str, Any]:
    from ..build import ensure_built
    ensure_built()
    log = print if verbose else (lambda *a, **k: None)
    eval_id = "eval-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    rec = Recorder(eval_id)
    tok = current_run_id.set(eval_id)
    timings = {}
    try:
        for m in modules or ALL_MODULES:
            t0 = time.perf_counter()
            log(f"[{m}]")
            RUNNERS[m](rec, log)
            timings[m] = round(time.perf_counter() - t0, 1)
    finally:
        current_run_id.reset(tok)
    report = {"eval_run_id": eval_id, "llm": get_llm().label, "timings_s": timings,
              "summary": rec.summary(), "rows": rec.rows}
    out = get_llm().settings.reports_dir
    (out / f"{eval_id}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "latest_evaluation.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(report: dict) -> str:
    lines = [f"# Evaluation report `{report['eval_run_id']}`", "", f"LLM: **{report['llm']}**", ""]
    for module, mets in report["summary"].items():
        lines += [f"## {module}", "", "| metric | mean |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in mets.items()]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", nargs="*", choices=ALL_MODULES)
    rep = run_evaluation(ap.parse_args().modules)
    print(json.dumps(rep["summary"], indent=1))
