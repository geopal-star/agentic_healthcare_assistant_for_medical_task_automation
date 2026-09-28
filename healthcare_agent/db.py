"""SQLite persistence layer: the EHR (patients, encounters, diagnoses, meds, labs,
vitals), the doctor schedule (doctors, slots, appointments) and the LLMOps
tables (agent runs, tool logs, memory events, evaluation results)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    mrn          TEXT,
    name         TEXT NOT NULL,
    dob          TEXT,
    age          INTEGER,
    gender       TEXT,
    phone        TEXT,
    email        TEXT,
    address      TEXT,
    allergies    TEXT,
    summary      TEXT,
    source       TEXT,
    created_at   TEXT,
    updated_at   TEXT
);
CREATE TABLE IF NOT EXISTS relationships (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,     -- the caregiver / account holder (a patient_id)
    patient_id   TEXT NOT NULL,     -- the dependent
    relation     TEXT NOT NULL,     -- father, mother, spouse, son ...
    UNIQUE(user_id, patient_id)
);
CREATE TABLE IF NOT EXISTS encounters (
    encounter_id    TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    visit_date      TEXT,
    location        TEXT,
    provider        TEXT,
    chief_complaint TEXT,
    subjective      TEXT,
    objective       TEXT,
    assessment      TEXT,
    plan            TEXT,
    source          TEXT,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS diagnoses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    encounter_id TEXT,
    description  TEXT NOT NULL,
    icd10        TEXT,
    status       TEXT DEFAULT 'active',
    noted_on     TEXT
);
CREATE TABLE IF NOT EXISTS medications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    encounter_id TEXT,
    name         TEXT NOT NULL,
    dosage       TEXT,
    status       TEXT DEFAULT 'active',
    noted_on     TEXT
);
CREATE TABLE IF NOT EXISTS labs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    taken_on     TEXT,
    test         TEXT NOT NULL,
    value        REAL,
    unit         TEXT,
    ref_range    TEXT,
    flag         TEXT               -- H / L / NULL
);
CREATE TABLE IF NOT EXISTS vitals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    taken_on     TEXT,
    name         TEXT NOT NULL,
    value        TEXT,
    unit         TEXT
);
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    specialty    TEXT NOT NULL,
    location     TEXT,
    experience_years INTEGER,
    languages    TEXT,
    work_start   TEXT,
    work_end     TEXT,
    work_days    TEXT               -- e.g. "0,1,2,3,4" (Mon=0)
);
CREATE TABLE IF NOT EXISTS slots (
    slot_id      TEXT PRIMARY KEY,
    doctor_id    TEXT NOT NULL,
    start_ts     TEXT NOT NULL,
    end_ts       TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'available'   -- available / booked / blocked
);
CREATE INDEX IF NOT EXISTS idx_slots_doc ON slots(doctor_id, start_ts);
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    patient_id   TEXT NOT NULL,
    doctor_id    TEXT NOT NULL,
    slot_id      TEXT NOT NULL,
    start_ts     TEXT NOT NULL,
    reason       TEXT,
    status       TEXT NOT NULL DEFAULT 'booked',     -- booked / cancelled / completed
    booked_by    TEXT,
    created_at   TEXT,
    updated_at   TEXT
);
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id   TEXT NOT NULL,     -- patient_id or user_id the memory is about
    kind         TEXT NOT NULL,     -- fact / preference / interaction / summary
    content      TEXT NOT NULL,
    created_at   TEXT
);
CREATE TABLE IF NOT EXISTS memory_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    run_id       TEXT,
    op           TEXT,              -- recall / write / short_term
    subject_id   TEXT,
    detail       TEXT
);
CREATE TABLE IF NOT EXISTS medical_docs (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT,
    title        TEXT,
    url          TEXT,
    content      TEXT,
    query        TEXT,
    fetched_at   TEXT
);
CREATE TABLE IF NOT EXISTS medical_answers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    run_id       TEXT,
    patient_id   TEXT,
    question     TEXT,
    answer       TEXT,
    sources      TEXT,              -- JSON
    live_fetch   INTEGER,
    mode         TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id       TEXT PRIMARY KEY,
    ts           TEXT,
    session_id   TEXT,
    user_id      TEXT,
    query        TEXT,
    plan         TEXT,              -- JSON
    trace        TEXT,              -- JSON
    answer       TEXT,
    success      INTEGER,
    latency_ms   INTEGER,
    llm_mode     TEXT
);
CREATE TABLE IF NOT EXISTS tool_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    run_id       TEXT,
    tool         TEXT,
    args         TEXT,
    result       TEXT,
    success      INTEGER,
    error        TEXT,
    latency_ms   INTEGER
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    run_id       TEXT,
    purpose      TEXT,
    mode         TEXT,              -- live / offline / fallback
    success      INTEGER,
    latency_ms   INTEGER,
    error        TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id  TEXT,
    ts           TEXT,
    module       TEXT,
    case_id      TEXT,
    metric       TEXT,
    score        REAL,
    details      TEXT
);
"""


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _connect() -> sqlite3.Connection:
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def query(sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple | dict = ()) -> int:
    """Run a write statement; returns rowcount."""
    with get_conn() as conn:
        return conn.execute(sql, params).rowcount


def insert(table: str, row: dict[str, Any], or_replace: bool = False) -> int:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    verb = "INSERT OR REPLACE" if or_replace else "INSERT"
    with get_conn() as conn:
        cur = conn.execute(f"{verb} INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        return cur.lastrowid


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
