"""EHR tools: identify patients, summarise history, and (for staff) add or
update structured and unstructured records."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .. import memory, records
from ..context import get_actor
from ..ingest import extract_medications
from ..llm import get_llm
from ..observability import traced_tool
from ..prompts import (HISTORY_SUMMARY_SYSTEM, HISTORY_SUMMARY_USER,
                       NOTE_EXTRACTION_SYSTEM)

SELF_WORDS = {"self", "me", "myself", "i", "my"}


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------

def authorize(patient_id: str) -> str | None:
    """Return an error message if the acting user may not access this patient."""
    actor = get_actor()
    if actor.role in {"attendant", "doctor"} or actor.user_id == patient_id:
        return None
    if any(d["patient_id"] == patient_id for d in records.get_dependents(actor.user_id)):
        return None
    return f"{actor.name} is not authorised to access this patient's records (not self or a linked dependent)."


def _brief(p: dict) -> dict:
    return {k: p.get(k) for k in ("patient_id", "name", "age", "gender", "phone", "address")}


# ---------------------------------------------------------------------------
# identify_patient
# ---------------------------------------------------------------------------

@traced_tool("identify_patient")
def identify_patient(name: str | None = None, relation: str | None = None, age: int | None = None) -> dict[str, Any]:
    """Resolve who the request is about: by relation to the acting user, by name, or 'self'."""
    actor = get_actor()
    rel = records.normalize_relation(relation) if relation else None
    if rel in SELF_WORDS or (not name and not rel and actor.role == "patient"):
        p = records.get_patient(actor.user_id)
        if p:
            return {"ok": True, "patient": _brief(p), "method": "self"}
        return {"ok": False, "error": "The acting user has no patient record."}
    if rel:
        deps = [d for d in records.get_dependents(actor.user_id) if d["relation"] == rel]
        if len(deps) == 1:
            return {"ok": True, "patient": _brief(deps[0]), "method": f"relation:{rel}"}
        if not name:
            return {"ok": False, "error": f"No {rel} is linked to {actor.name}'s account.",
                    "hint": "Provide the patient's full name, or ask an attendant to link the family member."}
    if name:
        matches = records.find_patients(name)
        if age:
            matches = [m for m in matches if not m.get("age") or abs(m["age"] - int(age)) <= 2] or matches
        if not matches:
            return {"ok": False, "error": f"No patient named '{name}' found."}
        top = matches[0]
        if len(matches) > 1 and matches[1]["match_score"] >= top["match_score"] - 0.03 and top["match_score"] < 1:
            return {"ok": False, "error": "Ambiguous patient name.", "candidates": [_brief(m) for m in matches[:3]]}
        denied = authorize(top["patient_id"])
        if denied:
            return {"ok": False, "error": denied}
        return {"ok": True, "patient": _brief(top), "method": "name", "match_score": top["match_score"]}
    return {"ok": False, "error": "Could not tell which patient this is about."}


# ---------------------------------------------------------------------------
# get_patient_history
# ---------------------------------------------------------------------------

def _offline_summary(rec: dict, alerts: list[dict]) -> str:
    p = rec["patient"]
    active = [d for d in rec["diagnoses"] if d["status"] == "active"]
    hist = [d for d in rec["diagnoses"] if d["status"] != "active"]
    meds = [m for m in rec["medications"] if m["status"] == "active"]
    lines = [f"**Overview**: {p['name']}, {p.get('age') or '?'}-year-old {(p.get('gender') or '').lower()}."
             + (f" Active problems: {', '.join(d['description'] for d in active)}." if active else "")]
    if rec["encounters"]:
        lines.append("**Diagnoses & treatments**:")
        for e in rec["encounters"][:4]:
            plan = (e["plan"] or "")[:220].rstrip() + ("..." if len(e["plan"] or "") > 220 else "")
            lines.append(f"- {e['visit_date']}: {e['chief_complaint'] or 'visit'} - {plan}")
    if hist:
        lines.append(f"**Past history**: {', '.join(d['description'] for d in hist)}.")
    lines.append("**Current medications**: " + (", ".join(f"{m['name']} {m['dosage'] or ''}".strip() for m in meds) or "none recorded") + ".")
    lines.append("**Alerts**: " + ("; ".join(a["text"] for a in alerts) if alerts else "none") + ".")
    if p.get("summary"):
        lines.append(f"**Recorded summary**: {p['summary']}")
    return "\n".join(lines)


def summarize_history(patient_id: str, focus: str | None = None) -> dict[str, Any]:
    rec = records.get_full_record(patient_id)
    if not rec:
        return {"ok": False, "error": f"unknown patient {patient_id}"}
    alerts = records.compute_alerts(rec)
    hits = memory.recall(focus or "medical history diagnosis treatment", [patient_id], k=4)
    mem_text = "\n".join(f"- {h['text'][:300]}" for h in hits) or "none"
    summary, mode = get_llm().text(
        "history_summary", HISTORY_SUMMARY_SYSTEM,
        HISTORY_SUMMARY_USER.format(record=records.record_to_text(rec),
                                    alerts="\n".join(f"- [{a['severity']}] {a['text']}" for a in alerts) or "none",
                                    memories=mem_text, focus=focus or "general"),
        fallback=lambda: _offline_summary(rec, alerts))
    p = rec["patient"]
    return {
        "ok": True, "patient": _brief(p), "summary": summary, "mode": mode, "alerts": alerts,
        "active_diagnoses": [f"{d['description']}{' (' + d['icd10'] + ')' if d['icd10'] else ''}"
                             for d in rec["diagnoses"] if d["status"] == "active"],
        "active_medications": [f"{m['name']} {m['dosage'] or ''}".strip() for m in rec["medications"] if m["status"] == "active"],
        "last_visit": rec["encounters"][0]["visit_date"] if rec["encounters"] else None,
        "memories_used": [h["text"][:200] for h in hits],
    }


@traced_tool("get_patient_history")
def get_patient_history(patient_id: str, focus: str | None = None) -> dict[str, Any]:
    """Summarise past diagnoses, treatments, medications and alerts."""
    if denied := authorize(patient_id):
        return {"ok": False, "error": denied}
    return summarize_history(patient_id, focus)


@traced_tool("search_patient_notes")
def search_patient_notes(patient_id: str, query: str) -> dict[str, Any]:
    """Semantic search over one patient's notes and memories (FAISS)."""
    if denied := authorize(patient_id):
        return {"ok": False, "error": denied}
    hits = memory.recall(query, [patient_id], k=5, min_score=0.3)
    return {"ok": True, "matches": [{"text": h["text"], "score": round(h["score"], 3), "kind": h["metadata"].get("kind")}
                                    for h in hits]}


# ---------------------------------------------------------------------------
# Record management (staff only)
# ---------------------------------------------------------------------------

def _require_staff() -> str | None:
    actor = get_actor()
    return None if actor.can_edit_records else (
        f"Only attendants or doctors can modify medical records (current role: {actor.role}).")


@traced_tool("register_patient")
def register_patient(name: str, age: int | None = None, gender: str | None = None, phone: str | None = None,
                     email: str | None = None, address: str | None = None, dob: str | None = None,
                     allergies: str | None = None) -> dict[str, Any]:
    if err := _require_staff():
        return {"ok": False, "error": err}
    existing = records.find_patients(name, cutoff=0.95)
    if existing:
        return {"ok": False, "error": f"A patient named {existing[0]['name']} already exists ({existing[0]['patient_id']})."}
    pid = records.create_patient(dict(name=name, age=age, gender=gender, phone=phone, email=email,
                                      address=address, dob=dob, allergies=allergies))
    memory.index_patient(pid)
    return {"ok": True, "patient_id": pid, "patient": _brief(records.get_patient(pid))}


class FieldUpdate(BaseModel):
    field: str = Field(description="one of: name, dob, age, gender, phone, email, address, allergies, summary")
    value: str


@traced_tool("update_patient_record")
def update_patient_record(patient_id: str, updates: list[dict] | dict | None = None) -> dict[str, Any]:
    """Update demographic / structured fields, e.g. phone, address, allergies."""
    if err := _require_staff():
        return {"ok": False, "error": err}
    if isinstance(updates, dict):
        updates = [{"field": k, "value": v} for k, v in updates.items()]
    fields = {u["field"].lower(): u["value"] for u in (updates or []) if u.get("field")}
    changed = records.update_patient(patient_id, fields)
    memory.index_patient(patient_id)
    return {"ok": True, "patient_id": patient_id, "updated": changed}


class NoteDiagnosis(BaseModel):
    description: str
    icd10: str | None = None


class NoteMedication(BaseModel):
    name: str
    dosage: str | None = None


class ExtractedNote(BaseModel):
    visit_date: str | None = Field(None, description="YYYY-MM-DD if stated")
    chief_complaint: str | None = None
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    diagnoses: list[NoteDiagnosis] = Field(default_factory=list)
    medications: list[NoteMedication] = Field(default_factory=list)
    allergies: str | None = None


def _heuristic_note(text: str) -> ExtractedNote:
    dx = [{"description": d.strip(), "icd10": c} for d, c in
          re.findall(r"(?:diagnos(?:is|ed with)|dx)[:\s]+([a-z][\w ,'-]+?)\s*\(([A-Z]\d{2}(?:\.\d+)?)\)", text, re.I)]
    if not dx and (m := re.search(r"(?:diagnos(?:is|ed with)|dx)[:\s]+([a-z][\w '-]{3,60})", text, re.I)):
        dx = [{"description": m.group(1).strip().rstrip("."), "icd10": None}]
    allergy = m.group(0) if (m := re.search(r"allerg\w*\s+(?:to\s+)?[a-z][^.;]*", text, re.I)) else None
    date_m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    meds = [NoteMedication(name=m["name"], dosage=m["dosage"]) for m in extract_medications(text)]
    return ExtractedNote(visit_date=date_m.group(0) if date_m else None, chief_complaint=text.split(".")[0][:120],
                         subjective=text, diagnoses=[NoteDiagnosis(**d) for d in dx], medications=meds,
                         allergies=allergy)


@traced_tool("add_clinical_note")
def add_clinical_note(patient_id: str, note_text: str, visit_date: str | None = None) -> dict[str, Any]:
    """Add an unstructured note; the LLM extracts structure (diagnoses, meds, allergies)."""
    if err := _require_staff():
        return {"ok": False, "error": err}
    if not records.get_patient(patient_id):
        return {"ok": False, "error": f"unknown patient {patient_id}"}
    note, mode = get_llm().structured("note_extraction", NOTE_EXTRACTION_SYSTEM, f"NOTE:\n{note_text}",
                                      ExtractedNote, fallback=lambda: _heuristic_note(note_text))
    data = note.model_dump()
    data["visit_date"] = visit_date or data.get("visit_date") or records.db.now_iso()[:10]
    data["provider"] = get_actor().name
    data["diagnoses"] = [d for d in data["diagnoses"] if d.get("description")]
    data["medications"] = [m for m in data["medications"] if m.get("name")]
    enc_id = records.add_encounter(patient_id, data, source="attendant_note")
    if note.allergies:
        current = records.get_patient(patient_id).get("allergies")
        if not current or note.allergies.lower() not in current.lower():
            records.update_patient(patient_id, {"allergies": "; ".join(filter(None, [current, note.allergies]))})
    memory.index_patient(patient_id)
    return {"ok": True, "encounter_id": enc_id, "extraction_mode": mode,
            "extracted": {k: data[k] for k in ("visit_date", "diagnoses", "medications")} | {"allergies": note.allergies}}
