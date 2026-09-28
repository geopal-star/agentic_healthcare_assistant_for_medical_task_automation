"""Response synthesis: turn executed step results into the user-facing reply."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from ..llm import get_llm
from ..prompts import SYNTHESIS_SYSTEM, SYNTHESIS_USER

DISCLAIMER = "_This assistant supports scheduling and information access; it does not replace advice from your doctor._"


def _when(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%a %d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return str(ts)


def _appt_lines(a: dict) -> str:
    return (f"- **Doctor:** {a.get('doctor_name')} ({a.get('specialty')})\n- **When:** {_when(a.get('start_ts'))}\n"
            f"- **Where:** {a.get('location', 'see clinic details')}\n- **Appointment ID:** `{a.get('appointment_id')}`")


def _alts(res: dict) -> str:
    alts = res.get("alternatives") or []
    return ("\n  Nearest alternatives: " + "; ".join(f"{s['doctor_name']} - {_when(s['start_ts'])}" for s in alts)) if alts else ""


def template_response(query: str, safety: dict, plan: dict, results: list[dict]) -> str:
    """Deterministic rendering used offline or when the LLM fails."""
    out: list[str] = []
    if safety.get("message"):
        out.append(f"> {safety['message']}\n")
    sources: list[dict] = []
    for r in results:
        tool, res, status = r["tool"], r.get("result") or {}, r["status"]
        if status == "skipped":
            out.append(f"**{r['goal']}** - skipped: {r.get('reason')}.")
            continue
        if not res.get("ok"):
            extra = ""
            if res.get("candidates"):
                extra = " Candidates: " + ", ".join(f"{c['name']} ({c['patient_id']})" for c in res["candidates"])
            out.append(f"**Could not complete: {r['goal']}** - {res.get('error', 'unknown error')}{extra}{_alts(res)}")
            if res.get("hint"):
                out.append(f"  _{res['hint']}_")
            continue
        if tool == "identify_patient":
            p = res["patient"]
            out.append(f"**Patient:** {p['name']} ({p['patient_id']}), {p.get('age') or '?'} y, {p.get('gender') or ''}")
        elif tool == "get_patient_history":
            out.append(f"### Medical history - {res['patient']['name']}\n{res['summary']}")
        elif tool == "book_appointment":
            a = res["appointment"]
            title = "Existing appointment found (not double-booked)" if res.get("already_booked") else "Appointment booked"
            out.append(f"### {title}\n{_appt_lines(a)}")
        elif tool == "reschedule_appointment":
            out.append(f"### Appointment rescheduled\nPrevious slot {_when(res['old_start_ts'])} was released.\n"
                       f"{_appt_lines(res['appointment'])}")
        elif tool == "cancel_appointment":
            c = res["cancelled"]
            out.append(f"### Appointment cancelled\n`{c['appointment_id']}` with {c['doctor_name']} on {_when(c['start_ts'])}.")
        elif tool == "list_appointments":
            rows = res["appointments"]
            out.append("### Upcoming appointments\n" + ("\n".join(
                f"- {_when(a['start_ts'])} - {a['doctor_name']} ({a['specialty']}) for {a['patient_name']} `{a['appointment_id']}`"
                for a in rows) or "No upcoming appointments."))
        elif tool in ("check_availability",):
            out.append("### Available slots\n" + "\n".join(
                f"- {s['doctor_name']} ({s['specialty']}): {_when(s['start_ts'])}" for s in res["slots"]))
        elif tool == "find_doctors":
            out.append("### Doctors\n" + "\n".join(f"- {d['name']} - {d['specialty']}, {d['location']}" for d in res["doctors"]))
        elif tool == "medical_info_search":
            out.append(f"### Medical information: {r['args'].get('query', '')}\n{res['answer']}")
            sources += res.get("sources", [])
        elif tool == "search_patient_notes":
            out.append("### Matching notes\n" + "\n".join(f"- {m['text'][:200]}" for m in res["matches"]))
        elif tool in ("update_patient_record", "register_patient", "add_clinical_note"):
            out.append(f"### Record updated\n```json\n{json.dumps({k: v for k, v in res.items() if k != 'ok'}, indent=1, default=str)[:800]}\n```")
    if plan.get("needs_clarification"):
        out.append(f"**Need more information:** {plan['needs_clarification']}")
    if sources:
        out.append("**Sources:**\n" + "\n".join(f"{s['n']}. [{s['title']}]({s['url']}) - {s['source']}" for s in sources))
    out.append(DISCLAIMER)
    return "\n\n".join(out)


def _compact(results: list[dict]) -> str:
    slim = []
    for r in results:
        res = dict(r.get("result") or {})
        for bulky in ("retrieved", "contexts", "memories_used", "fetch"):
            res.pop(bulky, None)
        slim.append({"step": r["step"], "goal": r["goal"], "tool": r["tool"], "status": r["status"],
                     "reason": r.get("reason"), "result": res})
    return json.dumps(slim, default=str, indent=1)[:12000]


def synthesize(query: str, safety: dict, plan: dict, results: list[dict]) -> tuple[str, str]:
    fallback = lambda: template_response(query, safety, plan, results)  # noqa: E731
    answer, mode = get_llm().text(
        "synthesis", SYNTHESIS_SYSTEM,
        SYNTHESIS_USER.format(query=query, safety=safety.get("message") or "none",
                              plan=json.dumps([{k: s[k] for k in ("step", "goal", "tool")} for s in plan.get("steps", [])]),
                              results=_compact(results), clarification=plan.get("needs_clarification") or "none"),
        fallback=fallback)
    if mode == "live":
        answer = re.sub(r"(?m)^#{1,2} ", "### ", answer)      # keep headings chat-sized whatever the model emits
    if mode == "live" and safety.get("level") == "emergency" and safety["message"][:40] not in answer:
        answer = f"> {safety['message']}\n\n{answer}"          # never let the LLM drop the emergency notice
    return answer, mode


def summarize_outcome(results: list[dict[str, Any]]) -> str:
    """One-line outcome used for interaction memory."""
    bits = []
    for r in results:
        res = r.get("result") or {}
        if r["tool"] == "book_appointment" and res.get("ok"):
            a = res["appointment"]
            bits.append(f"booked {a['specialty']} with {a['doctor_name']} on {a['start_ts'][:16]} ({a['appointment_id']})")
        elif r["tool"] == "medical_info_search" and res.get("ok"):
            bits.append(f"looked up '{r['args'].get('query')}'")
        elif r["tool"] == "get_patient_history" and res.get("ok"):
            bits.append("reviewed medical history")
        elif r["tool"] in ("cancel_appointment", "reschedule_appointment") and res.get("ok"):
            bits.append(r["tool"].replace("_", " "))
        elif r["status"] != "success":
            bits.append(f"{r['tool']} {r['status']}")
    return "; ".join(bits) or "no actions"
