"""Doctor Schedule API: doctor discovery, slot availability, booking,
cancellation and rescheduling. Booking is atomic (conditional UPDATE on the
slot row), idempotent (an existing upcoming booking with the same specialty is
returned instead of double-booking) and permission-checked."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any

from .. import db, records
from ..context import get_actor
from ..domain import normalize_specialty, parse_date_window, specialty_from_text, time_window
from ..observability import traced_tool
from ..seed import ensure_slots
from .patient import authorize


def _resolve_specialty(specialty: str | None, condition: str | None, patient_id: str | None = None) -> str | None:
    spec = normalize_specialty(specialty) or specialty_from_text(condition or "")
    if not spec and patient_id:          # fall back to the patient's primary active diagnosis
        for d in db.query("SELECT description FROM diagnoses WHERE patient_id=? AND status='active' ORDER BY noted_on DESC",
                          (patient_id,)):
            if spec := specialty_from_text(d["description"]):
                break
    return spec


def _available_slots(specialty: str | None, doctor_id: str | None, start: date | None, end: date | None,
                     hours: tuple[int, int], limit: int) -> list[dict[str, Any]]:
    ensure_slots()
    earliest = (datetime.now() + timedelta(hours=1)).isoformat(sep=" ", timespec="seconds")
    sql = ["""SELECT s.slot_id, s.start_ts, s.end_ts, d.doctor_id, d.name AS doctor_name, d.specialty, d.location
              FROM slots s JOIN doctors d ON d.doctor_id = s.doctor_id
              WHERE s.status = 'available' AND s.start_ts >= ?"""]
    params: list[Any] = [earliest]
    if doctor_id:
        sql.append("AND d.doctor_id = ?"); params.append(doctor_id)
    elif specialty:
        sql.append("AND d.specialty = ?"); params.append(specialty)
    if start:
        sql.append("AND date(s.start_ts) >= ?"); params.append(start.isoformat())
    if end:
        sql.append("AND date(s.start_ts) <= ?"); params.append(end.isoformat())
    sql.append("AND CAST(strftime('%H', s.start_ts) AS INTEGER) >= ? AND CAST(strftime('%H', s.start_ts) AS INTEGER) < ?")
    params += [hours[0], hours[1]]
    sql.append("ORDER BY s.start_ts LIMIT ?"); params.append(limit)
    return db.query(" ".join(sql), tuple(params))


@traced_tool("find_doctors")
def find_doctors(specialty: str | None = None, condition: str | None = None, location: str | None = None) -> dict[str, Any]:
    spec = _resolve_specialty(specialty, condition)
    sql, params = "SELECT * FROM doctors WHERE 1=1", []
    if spec:
        sql += " AND specialty = ?"; params.append(spec)
    if location:
        sql += " AND lower(location) LIKE ?"; params.append(f"%{location.lower()}%")
    docs = db.query(sql, tuple(params))
    if not docs:
        return {"ok": False, "error": f"No doctors found for specialty={spec!r} location={location!r}."}
    return {"ok": True, "specialty": spec, "doctors": docs}


@traced_tool("check_availability")
def check_availability(specialty: str | None = None, condition: str | None = None, doctor_id: str | None = None,
                       preferred_date: str | None = None, time_of_day: str | None = None, limit: int = 6) -> dict[str, Any]:
    spec = _resolve_specialty(specialty, condition)
    if not spec and not doctor_id:
        return {"ok": False, "error": "Need a specialty, condition or doctor to check availability."}
    start, end = parse_date_window(preferred_date)
    slots = _available_slots(spec, doctor_id, start, end, time_window(time_of_day), limit)
    if not slots and (start or time_of_day):
        alts = _available_slots(spec, doctor_id, None, None, (0, 24), 3)
        return {"ok": False, "error": "No free slots in the requested window.", "specialty": spec, "alternatives": alts}
    return {"ok": bool(slots), "specialty": spec, "window": [str(start), str(end)], "slots": slots,
            **({} if slots else {"error": "No free slots found."})}


def _upcoming_booking(patient_id: str, specialty: str | None = None) -> dict | None:
    sql = """SELECT a.*, d.name AS doctor_name, d.specialty, d.location FROM appointments a
             JOIN doctors d ON d.doctor_id = a.doctor_id
             WHERE a.patient_id = ? AND a.status = 'booked' AND a.start_ts >= ?"""
    params: list[Any] = [patient_id, datetime.now().isoformat(sep=" ", timespec="seconds")]
    if specialty:
        sql += " AND d.specialty = ?"; params.append(specialty)
    return db.query_one(sql + " ORDER BY a.start_ts LIMIT 1", tuple(params))


def _claim_slot(slot: dict, patient_id: str, reason: str | None) -> dict | None:
    """Atomically claim a slot; returns the appointment or None if someone else got it."""
    appt_id = "APT-" + uuid.uuid4().hex[:8].upper()
    with db.get_conn() as conn:
        cur = conn.execute("UPDATE slots SET status='booked' WHERE slot_id=? AND status='available'", (slot["slot_id"],))
        if cur.rowcount != 1:
            return None
        conn.execute("""INSERT INTO appointments (appointment_id, patient_id, doctor_id, slot_id, start_ts, reason,
                        status, booked_by, created_at, updated_at) VALUES (?,?,?,?,?,?, 'booked', ?,?,?)""",
                     (appt_id, patient_id, slot["doctor_id"], slot["slot_id"], slot["start_ts"], reason,
                      get_actor().user_id, db.now_iso(), db.now_iso()))
    return {"appointment_id": appt_id, "patient_id": patient_id, "doctor_id": slot["doctor_id"],
            "doctor_name": slot["doctor_name"], "specialty": slot["specialty"], "location": slot["location"],
            "start_ts": slot["start_ts"], "reason": reason, "status": "booked"}


@traced_tool("book_appointment")
def book_appointment(patient_id: str, specialty: str | None = None, condition: str | None = None,
                     doctor_id: str | None = None, slot_id: str | None = None, preferred_date: str | None = None,
                     time_of_day: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Book the earliest slot that matches the patient's intent and preferences."""
    if denied := authorize(patient_id):
        return {"ok": False, "error": denied}
    patient = records.get_patient(patient_id)
    if not patient:
        return {"ok": False, "error": f"unknown patient {patient_id}"}
    if doctor_id:
        doc = db.query_one("SELECT * FROM doctors WHERE doctor_id=?", (doctor_id,))
        if not doc:
            return {"ok": False, "error": f"unknown doctor {doctor_id}"}
        spec = doc["specialty"]
    else:
        spec = _resolve_specialty(specialty, condition or reason, patient_id)
    if not spec:
        return {"ok": False, "error": "Could not determine which specialty to book."}

    if existing := _upcoming_booking(patient_id, spec):
        return {"ok": True, "already_booked": True, "appointment": existing,
                "note": f"{patient['name']} already has an upcoming {spec} appointment; not double-booking."}

    if slot_id:
        candidates = db.query("""SELECT s.slot_id, s.start_ts, d.doctor_id, d.name AS doctor_name, d.specialty, d.location
                                 FROM slots s JOIN doctors d ON d.doctor_id=s.doctor_id
                                 WHERE s.slot_id=? AND s.status='available'""", (slot_id,))
    else:
        start, end = parse_date_window(preferred_date)
        candidates = _available_slots(spec, doctor_id, start, end, time_window(time_of_day), 5)
    if not candidates:
        alts = _available_slots(spec, doctor_id, None, None, (0, 24), 3)
        return {"ok": False, "error": f"No available {spec} slot matching the request "
                                      f"(date={preferred_date or 'any'}, time={time_of_day or 'any'}).",
                "specialty": spec, "alternatives": alts}
    reason = reason or condition or f"{spec} consultation"
    for slot in candidates:              # retry on a lost race
        if appt := _claim_slot(slot, patient_id, reason):
            return {"ok": True, "appointment": appt, "patient": {"patient_id": patient_id, "name": patient["name"]}}
    return {"ok": False, "error": "Slots were taken concurrently; please retry."}


def _find_appointment(appointment_id: str | None, patient_id: str | None, specialty: str | None) -> dict | None:
    if appointment_id:
        return db.query_one("""SELECT a.*, d.name AS doctor_name, d.specialty FROM appointments a
                               JOIN doctors d ON d.doctor_id=a.doctor_id WHERE a.appointment_id=?""", (appointment_id,))
    if patient_id:
        return _upcoming_booking(patient_id, normalize_specialty(specialty))
    return None


@traced_tool("cancel_appointment")
def cancel_appointment(appointment_id: str | None = None, patient_id: str | None = None,
                       specialty: str | None = None) -> dict[str, Any]:
    appt = _find_appointment(appointment_id, patient_id, specialty)
    if not appt or appt["status"] != "booked":
        return {"ok": False, "error": "No matching active appointment to cancel."}
    if denied := authorize(appt["patient_id"]):
        return {"ok": False, "error": denied}
    with db.get_conn() as conn:
        conn.execute("UPDATE appointments SET status='cancelled', updated_at=? WHERE appointment_id=?",
                     (db.now_iso(), appt["appointment_id"]))
        conn.execute("UPDATE slots SET status='available' WHERE slot_id=?", (appt["slot_id"],))
    return {"ok": True, "cancelled": {k: appt[k] for k in ("appointment_id", "doctor_name", "specialty", "start_ts")}}


@traced_tool("reschedule_appointment")
def reschedule_appointment(appointment_id: str | None = None, patient_id: str | None = None,
                           specialty: str | None = None, preferred_date: str | None = None,
                           time_of_day: str | None = None) -> dict[str, Any]:
    """Book the new slot first, then release the old one (never leaves the patient with nothing)."""
    appt = _find_appointment(appointment_id, patient_id, specialty)
    if not appt or appt["status"] != "booked":
        return {"ok": False, "error": "No matching active appointment to reschedule."}
    if denied := authorize(appt["patient_id"]):
        return {"ok": False, "error": denied}
    start, end = parse_date_window(preferred_date)
    hours = time_window(time_of_day)
    candidates = [s for s in _available_slots(None, appt["doctor_id"], start, end, hours, 5) if s["start_ts"] != appt["start_ts"]]
    if not candidates:   # same specialty, any doctor
        candidates = _available_slots(appt["specialty"], None, start, end, hours, 5)
    if not candidates:
        return {"ok": False, "error": "No alternative slot found in the requested window.",
                "alternatives": _available_slots(appt["specialty"], None, None, None, (0, 24), 3)}
    for slot in candidates:
        if new := _claim_slot(slot, appt["patient_id"], appt["reason"]):
            with db.get_conn() as conn:
                conn.execute("UPDATE appointments SET status='cancelled', updated_at=? WHERE appointment_id=?",
                             (db.now_iso(), appt["appointment_id"]))
                conn.execute("UPDATE slots SET status='available' WHERE slot_id=?", (appt["slot_id"],))
            return {"ok": True, "old_appointment_id": appt["appointment_id"], "old_start_ts": appt["start_ts"],
                    "appointment": new}
    return {"ok": False, "error": "Slots were taken concurrently; please retry."}


@traced_tool("list_appointments")
def list_appointments(patient_id: str | None = None, doctor_id: str | None = None,
                      upcoming_only: bool = True) -> dict[str, Any]:
    if patient_id and (denied := authorize(patient_id)):
        return {"ok": False, "error": denied}
    sql = ["""SELECT a.appointment_id, a.start_ts, a.status, a.reason, p.name AS patient_name, a.patient_id,
                     d.name AS doctor_name, d.specialty, d.location
              FROM appointments a JOIN doctors d ON d.doctor_id=a.doctor_id
              JOIN patients p ON p.patient_id=a.patient_id WHERE 1=1"""]
    params: list[Any] = []
    if patient_id:
        sql.append("AND a.patient_id=?"); params.append(patient_id)
    if doctor_id:
        sql.append("AND a.doctor_id=?"); params.append(doctor_id)
    if upcoming_only:
        sql.append("AND a.status='booked' AND a.start_ts >= ?")
        params.append(datetime.now().isoformat(sep=" ", timespec="seconds"))
    rows = db.query(" ".join(sql) + " ORDER BY a.start_ts", tuple(params))
    return {"ok": True, "appointments": rows, "count": len(rows)}
