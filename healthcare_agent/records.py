"""EHR repository: patient CRUD, lookup, relationships and clinical alerts."""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

from . import db
from .ingest import ParsedDocument, ParsedEncounter, age_from_dob, parse_range_flag

EDITABLE_FIELDS = {"name", "dob", "age", "gender", "phone", "email", "address", "allergies", "summary", "mrn"}


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def next_patient_id() -> str:
    row = db.query_one("SELECT patient_id FROM patients WHERE patient_id LIKE 'P%' ORDER BY patient_id DESC LIMIT 1")
    n = int(row["patient_id"][1:]) + 1 if row else 1
    return f"P{n:03d}"


def create_patient(fields: dict[str, Any], source: str = "manual") -> str:
    data = {k: v for k, v in fields.items() if k in EDITABLE_FIELDS and v not in (None, "")}
    if not data.get("name"):
        raise ValueError("patient name is required")
    if data.get("dob") and not data.get("age"):
        data["age"] = age_from_dob(data["dob"])
    pid = next_patient_id()
    db.insert("patients", {"patient_id": pid, **data, "source": source,
                           "created_at": db.now_iso(), "updated_at": db.now_iso()})
    return pid


def update_patient(patient_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    data = {k: v for k, v in fields.items() if k in EDITABLE_FIELDS}
    if not data:
        raise ValueError(f"no editable fields supplied (allowed: {sorted(EDITABLE_FIELDS)})")
    if "dob" in data and "age" not in data:
        data["age"] = age_from_dob(data["dob"])
    sets = ", ".join(f"{k} = ?" for k in data)
    n = db.execute(f"UPDATE patients SET {sets}, updated_at = ? WHERE patient_id = ?",
                   (*data.values(), db.now_iso(), patient_id))
    if n == 0:
        raise KeyError(f"unknown patient {patient_id}")
    return data


def get_patient(patient_id: str) -> dict[str, Any] | None:
    return db.query_one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))


def list_patients() -> list[dict[str, Any]]:
    return db.query("SELECT * FROM patients ORDER BY patient_id")


def find_patients(name: str, limit: int = 5, cutoff: float = 0.55) -> list[dict[str, Any]]:
    """Fuzzy name search (handles typos, first-name-only and last-name-only)."""
    name = (name or "").strip().lower()
    if not name:
        return []
    scored = []
    for p in list_patients():
        full = p["name"].lower()
        parts = full.split()
        score = max(
            difflib.SequenceMatcher(None, name, full).ratio(),
            max((difflib.SequenceMatcher(None, name, part).ratio() for part in parts), default=0) * 0.95,
            1.0 if name == full else 0,
            0.9 if name in full and len(name) >= 3 else 0,
        )
        if score >= cutoff:
            scored.append({**p, "match_score": round(score, 3)})
    return sorted(scored, key=lambda r: -r["match_score"])[:limit]


# ---------------------------------------------------------------------------
# Relationships (caregiver -> dependent)
# ---------------------------------------------------------------------------

RELATION_SYNONYMS = {
    "dad": "father", "papa": "father", "daddy": "father", "mom": "mother", "mum": "mother", "mummy": "mother",
    "wife": "spouse", "husband": "spouse", "son": "son", "daughter": "daughter",
    "grandpa": "grandfather", "grandma": "grandmother", "brother": "brother", "sister": "sister",
}


def normalize_relation(rel: str) -> str:
    rel = (rel or "").lower().strip()
    return RELATION_SYNONYMS.get(rel, rel)


def link_relationship(user_id: str, patient_id: str, relation: str) -> None:
    db.insert("relationships", {"user_id": user_id, "patient_id": patient_id,
                                "relation": normalize_relation(relation)}, or_replace=True)


def get_dependents(user_id: str) -> list[dict[str, Any]]:
    return db.query("""SELECT r.relation, p.* FROM relationships r JOIN patients p ON p.patient_id = r.patient_id
                       WHERE r.user_id = ?""", (user_id,))


# ---------------------------------------------------------------------------
# Clinical data
# ---------------------------------------------------------------------------

def add_encounter(patient_id: str, enc: ParsedEncounter | dict, source: str = "manual") -> str:
    e = asdict(enc) if isinstance(enc, ParsedEncounter) else dict(enc)
    enc_id = e.get("encounter_id") or f"ENC-{patient_id}-{(e.get('visit_date') or date.today().isoformat()).replace('-', '')}"
    # Avoid collisions when two notes share a date.
    while db.query_one("SELECT 1 FROM encounters WHERE encounter_id = ?", (enc_id,)):
        enc_id += "b"
    db.insert("encounters", {
        "encounter_id": enc_id, "patient_id": patient_id, "visit_date": e.get("visit_date"),
        "location": e.get("location"), "provider": e.get("provider"),
        "chief_complaint": e.get("chief_complaint"), "subjective": e.get("subjective", ""),
        "objective": e.get("objective", ""), "assessment": e.get("assessment", ""), "plan": e.get("plan", ""),
        "source": source, "created_at": db.now_iso(),
    })
    for d in e.get("diagnoses", []):
        add_diagnosis(patient_id, d["description"], d.get("icd10"), d.get("status", "active"),
                      e.get("visit_date"), enc_id)
    for m in e.get("medications", []):
        add_medication(patient_id, m["name"], m.get("dosage"), m.get("status", "active"),
                       m.get("noted_on") or e.get("visit_date"), enc_id)
    for v in e.get("vitals", []):
        db.insert("vitals", {"patient_id": patient_id, "taken_on": v.get("taken_on") or e.get("visit_date"),
                             "name": v["name"], "value": str(v["value"]), "unit": v.get("unit")})
    return enc_id


def add_diagnosis(patient_id: str, description: str, icd10: str | None = None, status: str = "active",
                  noted_on: str | None = None, encounter_id: str | None = None) -> None:
    dup = db.query_one("SELECT id FROM diagnoses WHERE patient_id=? AND lower(description)=lower(?)",
                       (patient_id, description))
    if dup:
        db.execute("UPDATE diagnoses SET status=?, icd10=COALESCE(?, icd10) WHERE id=?", (status, icd10, dup["id"]))
        return
    db.insert("diagnoses", {"patient_id": patient_id, "encounter_id": encounter_id, "description": description,
                            "icd10": icd10, "status": status, "noted_on": noted_on})


def add_medication(patient_id: str, name: str, dosage: str | None = None, status: str = "active",
                   noted_on: str | None = None, encounter_id: str | None = None) -> None:
    # A new prescription of the same drug supersedes the old one.
    db.execute("UPDATE medications SET status='superseded' WHERE patient_id=? AND lower(name)=lower(?) AND status='active'",
               (patient_id, name))
    db.insert("medications", {"patient_id": patient_id, "encounter_id": encounter_id, "name": name,
                              "dosage": dosage, "status": status, "noted_on": noted_on})


def add_labs(patient_id: str, labs: list[dict]) -> int:
    for lab in labs:
        flag = lab.get("flag")
        if flag is None and lab.get("value") is not None:
            flag = parse_range_flag(float(lab["value"]), lab.get("ref_range"))
        db.insert("labs", {"patient_id": patient_id, "taken_on": lab.get("taken_on"), "test": lab["test"],
                           "value": lab.get("value"), "unit": lab.get("unit"),
                           "ref_range": lab.get("ref_range"), "flag": flag})
    return len(labs)


def ingest_document(doc: ParsedDocument, patient_id: str | None = None, source: str = "pdf") -> dict[str, Any]:
    """Attach a parsed PDF to an existing patient (matched by name) or create one."""
    created = False
    if patient_id is None:
        name = doc.patient.get("name")
        match = next((p for p in find_patients(name or "", cutoff=0.92)), None) if name else None
        if match:
            patient_id = match["patient_id"]
        elif name:
            patient_id, created = create_patient(doc.patient, source=source), True
        else:
            raise ValueError("could not determine which patient this document belongs to")
    # Fill demographic gaps from the document without overwriting curated values.
    current = get_patient(patient_id) or {}
    gaps = {k: v for k, v in doc.patient.items() if k in EDITABLE_FIELDS and v and not current.get(k)}
    if gaps:
        update_patient(patient_id, gaps)
    enc_ids = [add_encounter(patient_id, e, source=source) for e in doc.encounters]
    for prob in doc.history_problems:
        add_diagnosis(patient_id, prob, None, "history")
    n_labs = add_labs(patient_id, doc.labs)
    return {"patient_id": patient_id, "created_patient": created, "encounters": enc_ids, "labs": n_labs}


def get_full_record(patient_id: str) -> dict[str, Any] | None:
    p = get_patient(patient_id)
    if not p:
        return None
    q = lambda sql: db.query(sql, (patient_id,))  # noqa: E731
    return {
        "patient": p,
        "encounters": q("SELECT * FROM encounters WHERE patient_id=? ORDER BY visit_date DESC"),
        "diagnoses": q("SELECT * FROM diagnoses WHERE patient_id=? ORDER BY noted_on DESC"),
        "medications": q("SELECT * FROM medications WHERE patient_id=? ORDER BY noted_on DESC"),
        "labs": q("SELECT * FROM labs WHERE patient_id=? ORDER BY taken_on DESC, test"),
        "vitals": q("SELECT * FROM vitals WHERE patient_id=? ORDER BY taken_on DESC"),
        "caregivers": q("""SELECT r.relation, p.patient_id, p.name FROM relationships r
                           JOIN patients p ON p.patient_id=r.user_id WHERE r.patient_id=?"""),
        "appointments": q("""SELECT a.*, d.name AS doctor_name, d.specialty FROM appointments a
                             JOIN doctors d ON d.doctor_id=a.doctor_id WHERE a.patient_id=? ORDER BY a.start_ts DESC"""),
    }


def latest_labs(record: dict) -> list[dict]:
    seen, out = set(), []
    for lab in record["labs"]:                      # already sorted newest first
        if lab["test"] not in seen:
            seen.add(lab["test"])
            out.append(lab)
    return out


def compute_alerts(record: dict, today: date | None = None) -> list[dict[str, str]]:
    """Rule-based clinical alerts (deterministic - never delegated to the LLM)."""
    today = today or date.today()
    p, alerts = record["patient"], []
    if p.get("allergies"):
        alerts.append({"severity": "high", "type": "allergy", "text": f"Allergy: {p['allergies'].rstrip('.')}"})
    for lab in latest_labs(record):
        if lab["flag"]:
            word = "HIGH" if lab["flag"] == "H" else "LOW"
            alerts.append({"severity": "medium", "type": "lab",
                           "text": f"{lab['test']} {word}: {lab['value']:g} {lab['unit'] or ''} (ref {lab['ref_range']}, {lab['taken_on']})"})
    for v in record["vitals"]:
        if v["name"] == "BP" and re.fullmatch(r"\d+/\d+", v["value"] or ""):
            sys_, dia = map(int, v["value"].split("/"))
            if sys_ >= 140 or dia >= 90:
                alerts.append({"severity": "medium", "type": "vital",
                               "text": f"Elevated blood pressure {v['value']} mmHg on {v['taken_on']}"})
            break
    active_meds = [m for m in record["medications"] if m["status"] == "active"]
    if len(active_meds) >= 5:
        alerts.append({"severity": "low", "type": "medication", "text": f"Polypharmacy: {len(active_meds)} active medications"})
    if (p.get("age") or 0) >= 65:
        alerts.append({"severity": "low", "type": "risk", "text": f"Geriatric patient (age {p['age']}): review dosing and fall risk"})
    # Follow-up due: "Return in N months/weeks/days" in the latest plan, with no upcoming appointment.
    if record["encounters"]:
        last = record["encounters"][0]
        m = re.search(r"(?:return|follow[- ]?up)\s+in\s+(\d+)\s+(day|week|month)s?", last["plan"] or "", re.I)
        if m and last["visit_date"]:
            n, unit = int(m.group(1)), m.group(2).lower()
            due = date.fromisoformat(last["visit_date"]) + timedelta(days=n * {"day": 1, "week": 7, "month": 30}[unit])
            upcoming = [a for a in record["appointments"] if a["status"] == "booked" and a["start_ts"][:10] >= today.isoformat()]
            if due < today and not upcoming:
                alerts.append({"severity": "medium", "type": "follow-up",
                               "text": f"Follow-up was due {due.isoformat()} (per {last['visit_date']} plan); none booked"})
    return alerts


def record_to_text(record: dict) -> str:
    """Compact plain-text rendering of a record, used as LLM context."""
    p = record["patient"]
    lines = [f"Patient: {p['name']} (ID {p['patient_id']}), {p.get('age') or '?'}y {p.get('gender') or ''}, DOB {p.get('dob') or 'unknown'}"]
    if p.get("allergies"):
        lines.append(f"Allergies: {p['allergies']}")
    if record["diagnoses"]:
        lines.append("Diagnoses: " + "; ".join(
            f"{d['description']}{' (' + d['icd10'] + ')' if d['icd10'] else ''} [{d['status']}]" for d in record["diagnoses"]))
    active = [m for m in record["medications"] if m["status"] == "active"]
    if active:
        lines.append("Active medications: " + "; ".join(f"{m['name']} {m['dosage'] or ''}".strip() for m in active))
    abnormal = [lab for lab in latest_labs(record) if lab["flag"]]
    if abnormal:
        lines.append("Abnormal labs: " + "; ".join(f"{lab['test']} {lab['value']:g} {lab['unit'] or ''} ({lab['flag']})" for lab in abnormal))
    for e in record["encounters"][:4]:
        lines.append(f"\nEncounter {e['visit_date']} at {e['location'] or 'n/a'} ({e['provider'] or 'n/a'}):"
                     f"\n  Complaint: {e['chief_complaint'] or ''}\n  Subjective: {e['subjective'][:600]}"
                     f"\n  Assessment: {e['assessment'][:400]}\n  Plan: {e['plan'][:500]}")
    if p.get("summary"):
        lines.append(f"\nRecorded summary: {p['summary']}")
    return "\n".join(lines)
