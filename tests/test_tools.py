from datetime import datetime

from healthcare_agent import db, memory, records
from healthcare_agent.tools.medical_search import medical_info_search
from healthcare_agent.tools.patient import (add_clinical_note, get_patient_history, identify_patient,
                                            update_patient_record)
from healthcare_agent.tools.scheduling import (_claim_slot, book_appointment, cancel_appointment,
                                               check_availability, reschedule_appointment)


def pid(name):
    return records.find_patients(name)[0]["patient_id"]


def test_identify_by_relation(as_actor):
    as_actor(pid("Rahul Negi"), "Rahul Negi")
    res = identify_patient(relation="dad")
    assert res["ok"] and res["patient"]["name"] == "Suresh Negi"


def test_privacy_guard_blocks_unrelated_patient(as_actor):
    as_actor(pid("Anjali Mehra"), "Anjali Mehra")
    res = identify_patient(name="Ramesh Kulkarni")
    assert not res["ok"] and "not authorised" in res["error"]


def test_history_summary_contains_alerts(as_actor):
    as_actor("attendant-01", "Asha", "attendant")
    res = get_patient_history(patient_id=pid("Suresh Negi"))
    assert res["ok"] and "Chronic kidney disease stage 3b (N18.32)" in res["active_diagnoses"]
    assert any("eGFR" in a["text"] for a in res["alerts"])


def test_booking_is_atomic_idempotent_and_cancellable(as_actor):
    as_actor("attendant-01", "Asha", "attendant")
    p = pid("Ramesh Kulkarni")
    first = book_appointment(patient_id=p, specialty="Cardiology", time_of_day="morning")
    assert first["ok"] and first["appointment"]["specialty"] == "Cardiology"
    assert datetime.fromisoformat(first["appointment"]["start_ts"]).hour < 12
    again = book_appointment(patient_id=p, specialty="Cardiology")
    assert again["already_booked"] and again["appointment"]["appointment_id"] == first["appointment"]["appointment_id"]
    slot_status = db.query_one("SELECT status FROM slots WHERE slot_id=?", (first["appointment"]["doctor_id"] + "-" +
                                datetime.fromisoformat(first["appointment"]["start_ts"]).strftime("%Y%m%d%H%M"),))
    assert slot_status["status"] == "booked"
    moved = reschedule_appointment(patient_id=p, specialty="Cardiology", preferred_date="next week", time_of_day="afternoon")
    assert moved["ok"] and datetime.fromisoformat(moved["appointment"]["start_ts"]).hour >= 12
    assert cancel_appointment(appointment_id=moved["appointment"]["appointment_id"])["ok"]


def test_race_on_same_slot_has_one_winner():
    slot = db.query_one("""SELECT s.slot_id, s.start_ts, d.doctor_id, d.name AS doctor_name, d.specialty, d.location
                           FROM slots s JOIN doctors d ON d.doctor_id=s.doctor_id WHERE s.status='available' LIMIT 1""")
    p = pid("Rahul Negi")
    assert _claim_slot(slot, p, "t") is not None
    assert _claim_slot(slot, p, "t") is None


def test_unavailable_window_returns_alternatives(as_actor):
    as_actor("attendant-01", "Asha", "attendant")
    res = check_availability(specialty="Pulmonology", time_of_day="evening")     # clinic runs 10:00-14:00
    assert not res["ok"] and res["alternatives"]


def test_patient_cannot_edit_records(as_actor):
    as_actor(pid("Anjali Mehra"), "Anjali Mehra")
    res = update_patient_record(patient_id=pid("Anjali Mehra"), updates=[{"field": "phone", "value": "1"}])
    assert not res["ok"] and "attendants or doctors" in res["error"]


def test_unstructured_note_extraction(as_actor):
    as_actor("attendant-01", "Asha", "attendant")
    p = pid("David Thompson")
    res = add_clinical_note(patient_id=p, note_text="2026-09-20 review. Diagnosis: Diabetic neuropathy (E11.40). "
                                                   "Started Pregabalin 75mg BID. Allergic to penicillin.")
    assert res["ok"]
    assert res["extracted"]["diagnoses"][0]["icd10"] == "E11.40"
    assert any(m["name"] == "Pregabalin" for m in res["extracted"]["medications"])
    assert "penicillin" in records.get_patient(p)["allergies"].lower()


def test_medical_search_is_grounded_and_cited():
    res = medical_info_search(query="chronic kidney disease treatment")
    assert res["ok"] and res["sources"]
    assert all(s["url"].startswith("https://") for s in res["sources"])
    assert "[1]" in res["answer"]


def test_memory_write_and_recall():
    p = pid("Rebeca Nagle")
    memory.remember(p, "Prefers early-morning appointments on weekdays", kind="preference")
    hits = memory.recall("what time does she like appointments", [p], k=3, min_score=0.0)
    assert any("early-morning" in h["text"] for h in hits)
