"""One-shot data pipeline: dataset -> SQLite EHR -> FAISS indexes.

    python -m healthcare_agent.build            # rebuild everything
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta

import pandas as pd

from . import db, memory, records, seed
from .config import get_settings
from .ingest import age_from_dob, parse_pdf
from .tools.medical_search import load_seed
from .vectorstore import reset_stores


def _clean(v):
    return None if pd.isna(v) else str(v).strip()


def load_records_xlsx(path) -> list[str]:
    """records.xlsx has one row per encounter; de-duplicate patients by (name, phone)."""
    df = pd.read_excel(path)
    df = df.drop_duplicates(subset=["Name", "Phone_number"]).reset_index(drop=True)
    ids = []
    for _, row in df.iterrows():
        pid = records.create_patient({
            "name": _clean(row["Name"]), "age": int(row["Age"]) if not pd.isna(row["Age"]) else None,
            "gender": _clean(row["Gender"]), "phone": _clean(row["Phone_number"]), "email": _clean(row["Email"]),
            "address": _clean(row["Address"]), "summary": _clean(row["Summary"]),
        }, source="records.xlsx")
        ids.append(pid)
    return ids


def add_demo_family() -> str:
    """Synthetic 70-year-old CKD father of Rahul Negi (sample scenario)."""
    demo = seed.demo_family_records()
    pid = records.create_patient({**demo["patient"], "allergies": demo["allergy_note"].replace("Allergy: ", "")},
                                 source="synthetic_demo")
    for enc in demo["encounters"]:
        records.add_encounter(pid, enc, source="synthetic_demo")
    for desc, icd, status, noted in demo["diagnoses"]:
        records.add_diagnosis(pid, desc, icd, status, noted)
    for name, dose, status, noted in demo["medications"]:
        records.add_medication(pid, name, dose, status, noted)
    records.add_labs(pid, [{"taken_on": d, "test": t, "value": v, "unit": u, "ref_range": r}
                           for d, t, v, u, r in demo["labs"]])
    rahul = next(iter(records.find_patients("Rahul Negi", cutoff=0.95)), None)
    if rahul:
        records.link_relationship(rahul["patient_id"], pid, "father")
    return pid


def add_demo_history() -> None:
    """A few past appointments so dashboards have history on first launch."""
    past = date.today() - timedelta(days=20)
    rows = [("Anjali Mehra", "D05", "Upper respiratory infection follow-up"), ("David Thompson", "D04", "Diabetes review")]
    for i, (name, doc, reason) in enumerate(rows):
        start = f"{(past + timedelta(days=i)).isoformat()} 10:00:00"
        if match := records.find_patients(name, cutoff=0.95):
            db.insert("appointments", {"appointment_id": f"APT-HIST{i:02d}", "patient_id": match[0]["patient_id"], "doctor_id": doc,
                                       "slot_id": f"{doc}-hist{i}", "start_ts": start, "reason": reason,
                                       "status": "completed", "booked_by": "attendant-01",
                                       "created_at": start, "updated_at": start})


def build(reset: bool = True, verbose: bool = True) -> dict:
    s = get_settings()
    log = print if verbose else (lambda *a, **k: None)
    if reset:
        if s.db_path.exists():
            s.db_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = s.db_path.with_name(s.db_path.name + suffix)
            if p.exists():
                p.unlink()
        shutil.rmtree(s.vector_dir, ignore_errors=True)
        s.vector_dir.mkdir(parents=True, exist_ok=True)
        reset_stores()
    db.init_db()

    ids = load_records_xlsx(s.raw_dir / "records.xlsx")
    log(f"records.xlsx -> {len(ids)} unique patients")
    for pdf in sorted(s.raw_dir.glob("*.pdf")):
        doc = parse_pdf(pdf)
        res = records.ingest_document(doc, source=pdf.name)
        log(f"{pdf.name} [{doc.layout}] -> {res['patient_id']}: {len(res['encounters'])} encounters, {res['labs']} labs")
    # Current age from DOB where known (records.xlsx ages were captured at visit time).
    for p in records.list_patients():
        if p["dob"]:
            records.update_patient(p["patient_id"], {"age": age_from_dob(p["dob"])})

    demo_pid = add_demo_family()
    log(f"synthetic demo patient Suresh Negi -> {demo_pid} (father of Rahul Negi)")
    seed.seed_doctors()
    n_slots = seed.ensure_slots()
    add_demo_history()
    log(f"doctors: {len(seed.DOCTORS)}, slots created: {n_slots}")

    n_chunks = sum(memory.index_patient(p["patient_id"]) for p in records.list_patients())
    log(f"patient_memory index: {n_chunks} chunks")
    n_med = load_seed(s.knowledge_seed)
    log(f"medical_knowledge index: {n_med} chunks from seed corpus")
    return {"patients": len(records.list_patients()), "slots": n_slots, "patient_chunks": n_chunks, "medical_chunks": n_med}


def ensure_built() -> None:
    """Build on first use (e.g. first Streamlit launch)."""
    s = get_settings()
    if not s.db_path.exists() or not db.query_one("SELECT name FROM sqlite_master WHERE name='patients'") \
            or not db.query_one("SELECT 1 FROM patients LIMIT 1"):
        build(reset=True, verbose=False)
    else:
        db.init_db()            # apply any new tables
        seed.ensure_slots()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-reset", action="store_true")
    build(reset=not ap.parse_args().no_reset)
