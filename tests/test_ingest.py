from datetime import date

from healthcare_agent import records
from healthcare_agent.config import get_settings
from healthcare_agent.domain import parse_date_window, specialty_from_text
from healthcare_agent.ingest import parse_pdf, parse_range_flag


def raw(name):
    return get_settings().raw_dir / name


def test_simple_note_parsed():
    doc = parse_pdf(raw("sample_report_david.pdf"))
    assert doc.layout == "hp_note"
    assert doc.patient["name"] == "David Thompson" and doc.patient["dob"] == "1972-03-14"
    enc = doc.encounters[0]
    assert enc.diagnoses == [{"description": "Type 2 Diabetes Mellitus", "icd10": "E11.9", "status": "active"}]
    assert enc.medications[0]["name"] == "Metformin" and "1000mg BID" in enc.medications[0]["dosage"]
    assert {v["name"] for v in enc.vitals} == {"BP", "Pulse", "Temperature"}


def test_ehr_export_parsed():
    doc = parse_pdf(raw("sample_patient.pdf"))
    assert doc.layout == "ehr_export"
    assert doc.patient["mrn"] == "D2147783321-907568"
    assert [e.encounter_id for e in doc.encounters] == ["727374050", "730177633"]
    assert any(d["icd10"] == "M94.0" for d in doc.encounters[1].diagnoses)
    flags = {lab["test"]: lab["flag"] for lab in doc.labs}
    assert flags["ALT"] == "H" and flags["HDL cholesterol"] == "L" and flags["Sodium"] is None
    assert "PCOS" in doc.history_problems


def test_range_flags():
    assert parse_range_flag(52, "6-29") == "H"
    assert parse_range_flag(35, "> OR = 50") == "L"
    assert parse_range_flag(190, "<150") == "H"
    assert parse_range_flag(149, "<150") is None
    assert parse_range_flag(60, "> OR = 60") is None


def test_database_built():
    names = {p["name"] for p in records.list_patients()}
    assert {"Rahul Negi", "Rebeca Nagle", "Ramesh Kulkarni", "Anjali Mehra", "David Thompson", "Suresh Negi"} <= names
    assert len(names) == 6                     # the 3 duplicated Rebeca rows were de-duplicated
    rahul = records.find_patients("Rahul Negi")[0]
    assert records.get_dependents(rahul["patient_id"])[0]["name"] == "Suresh Negi"


def test_fuzzy_lookup_handles_typos():
    assert records.find_patients("Rebecca Nagel")[0]["name"] == "Rebeca Nagle"
    assert records.find_patients("Kulkarni")[0]["name"] == "Ramesh Kulkarni"


def test_alerts_for_ckd_patient():
    pid = records.find_patients("Suresh Negi")[0]["patient_id"]
    texts = " | ".join(a["text"] for a in records.compute_alerts(records.get_full_record(pid)))
    assert "eGFR LOW" in texts and "Potassium HIGH" in texts and "sulfa" in texts


def test_domain_helpers():
    assert specialty_from_text("book a nephrologist") == "Nephrology"
    assert specialty_from_text("he has high blood pressure") == "Cardiology"
    today = date(2026, 9, 27)                  # a Sunday
    assert parse_date_window("tomorrow", today) == (date(2026, 9, 28), date(2026, 9, 28))
    assert parse_date_window("next week", today) == (date(2026, 9, 28), date(2026, 10, 4))
    assert parse_date_window("friday", today)[0] == date(2026, 10, 2)
    assert parse_date_window("asap", today) == (None, None)
