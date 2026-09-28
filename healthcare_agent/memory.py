"""Long-term patient memory.

* Short-term memory (the running conversation, the "active patient") lives in
  the LangGraph state and is persisted per chat thread by the checkpointer.
* Long-term memory lives here: patient profiles, encounter notes and facts
  learned from conversations are embedded into the ``patient_memory`` FAISS
  collection (and mirrored in SQLite), so later sessions can recall them with
  semantic search.
"""
from __future__ import annotations

from typing import Any

from . import db, records
from .observability import log_memory
from .vectorstore import chunk_text, get_store

STORE = "patient_memory"


def index_patient(patient_id: str) -> int:
    """(Re)index a patient's record into the vector store. Idempotent."""
    rec = records.get_full_record(patient_id)
    if not rec:
        return 0
    p = rec["patient"]
    base = {"patient_id": patient_id, "patient_name": p["name"]}
    texts, metas = [], []

    profile = records.record_to_text({**rec, "encounters": []})
    texts.append(profile)
    metas.append({**base, "kind": "profile"})
    for e in rec["encounters"]:
        note = (f"Visit {e['visit_date']} ({e['location'] or 'clinic'}). Complaint: {e['chief_complaint'] or ''}. "
                f"{e['subjective']} Assessment: {e['assessment']} Plan: {e['plan']}")
        for chunk in chunk_text(note):
            texts.append(f"[{p['name']}] {chunk}")
            metas.append({**base, "kind": "encounter", "encounter_id": e["encounter_id"], "date": e["visit_date"]})
    for m in db.query("SELECT * FROM memories WHERE subject_id=?", (patient_id,)):
        texts.append(f"[{p['name']}] {m['content']}")
        metas.append({**base, "kind": f"memory:{m['kind']}", "date": m["created_at"][:10]})
    return get_store(STORE).add(texts, metas)


def remember(subject_id: str, content: str, kind: str = "fact") -> None:
    """Persist a long-term memory about a patient (or user)."""
    dup = db.query_one("SELECT id FROM memories WHERE subject_id=? AND content=?", (subject_id, content))
    if dup:
        return
    db.insert("memories", {"subject_id": subject_id, "kind": kind, "content": content, "created_at": db.now_iso()})
    p = records.get_patient(subject_id)
    name = p["name"] if p else subject_id
    get_store(STORE).add([f"[{name}] {content}"],
                         [{"patient_id": subject_id, "patient_name": name, "kind": f"memory:{kind}", "date": db.now_iso()[:10]}])
    log_memory("write", subject_id, {"kind": kind, "content": content})


def recall(query: str, patient_ids: list[str] | None = None, k: int = 4, min_score: float = 0.35) -> list[dict[str, Any]]:
    """Semantic recall, optionally restricted to a set of patients."""
    where = (lambda md: md.get("patient_id") in set(patient_ids)) if patient_ids else None
    hits = get_store(STORE).search(query, k=k, where=where, min_score=min_score)
    log_memory("recall", ",".join(patient_ids or []) or None,
               {"query": query, "hits": [{"text": h["text"][:160], "score": round(h["score"], 3)} for h in hits]})
    return hits


def list_memories(subject_id: str | None = None) -> list[dict[str, Any]]:
    if subject_id:
        return db.query("SELECT * FROM memories WHERE subject_id=? ORDER BY created_at DESC", (subject_id,))
    return db.query("SELECT m.*, p.name FROM memories m LEFT JOIN patients p ON p.patient_id=m.subject_id ORDER BY m.created_at DESC")
