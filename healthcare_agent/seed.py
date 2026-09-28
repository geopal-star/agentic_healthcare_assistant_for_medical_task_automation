"""Synthetic data that the provided dataset does not contain but the use case
needs: a doctor directory with working hours (the "Doctor Schedule API"
backend), a rolling calendar of appointment slots, and one demo family so the
capstone's reference scenario ("my 70-year-old father has chronic kidney
disease...") can run end to end. Every synthetic row carries
``source='synthetic_demo'`` so it is never confused with the provided data."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta

from . import db
from .config import get_settings

DOCTORS = [
    # id, name, specialty, location, years, languages, start, end, weekdays(Mon=0)
    ("D01", "Dr. Priya Raman", "Nephrology", "Sunrise Multispeciality Clinic, Chennai", 16, "English, Tamil", "09:00", "13:00", "0,1,2,3,4"),
    ("D02", "Dr. Vikram Sethi", "Nephrology", "Lakeside Health Centre, Bangalore", 11, "English, Hindi, Kannada", "14:00", "18:00", "0,2,4,5"),
    ("D03", "Dr. Ananya Iyer", "Cardiology", "Sunrise Multispeciality Clinic, Chennai", 18, "English, Tamil", "10:00", "16:00", "0,1,3,4"),
    ("D04", "Dr. Rohan Kapoor", "Endocrinology", "Lakeside Health Centre, Bangalore", 9, "English, Hindi", "09:00", "15:00", "1,2,3,4"),
    ("D05", "Dr. Meera Nair", "General Medicine", "CityCare Family Clinic, Pune", 12, "English, Hindi, Marathi", "08:30", "17:00", "0,1,2,3,4,5"),
    ("D06", "Dr. Sameer Khan", "General Medicine", "Lakeside Health Centre, Bangalore", 7, "English, Hindi, Urdu", "09:00", "17:00", "0,1,2,3,4"),
    ("D07", "Dr. Kavya Reddy", "Pulmonology", "CityCare Family Clinic, Pune", 10, "English, Telugu", "10:00", "14:00", "0,2,4"),
    ("D08", "Dr. Elena Brooks", "Obstetrics & Gynecology", "Bridport Family Medicine, Portland", 15, "English, Spanish", "08:00", "12:00", "0,1,2,3,4"),
    ("D09", "Dr. Arvind Menon", "Neurology", "Sunrise Multispeciality Clinic, Chennai", 20, "English, Malayalam", "13:00", "17:00", "1,3,5"),
    ("D10", "Dr. Farah Siddiqui", "Rheumatology", "Lakeside Health Centre, Bangalore", 8, "English, Hindi", "09:00", "13:00", "0,1,3"),
    ("D11", "Dr. Thomas Walker", "Orthopedics", "Bridport Family Medicine, Portland", 13, "English", "09:00", "15:00", "0,2,3,4"),
    ("D12", "Dr. Nisha Verma", "Dermatology", "CityCare Family Clinic, Pune", 6, "English, Hindi", "11:00", "17:00", "1,2,4,5"),
    ("D13", "Dr. Karthik Subramanian", "Gastroenterology", "Sunrise Multispeciality Clinic, Chennai", 14, "English, Tamil", "09:00", "13:00", "0,2,4"),
    ("D14", "Dr. Sarah Kim", "Psychiatry", "Bridport Family Medicine, Portland", 10, "English, Korean", "12:00", "18:00", "0,1,2,3"),
    ("D15", "Dr. Aditya Rao", "Nutrition & Dietetics", "Lakeside Health Centre, Bangalore", 5, "English, Kannada", "09:00", "13:00", "0,1,2,3,4"),
]

# Staff accounts used by the UI for role-based access (not patients).
STAFF = [
    {"user_id": "attendant-01", "name": "Front-desk Attendant (Asha)", "role": "attendant"},
    {"user_id": "D01", "name": "Dr. Priya Raman (Nephrology)", "role": "doctor"},
    {"user_id": "D03", "name": "Dr. Ananya Iyer (Cardiology)", "role": "doctor"},
    {"user_id": "D05", "name": "Dr. Meera Nair (General Medicine)", "role": "doctor"},
]


def _stable_fraction(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def seed_doctors() -> None:
    for row in DOCTORS:
        keys = ["doctor_id", "name", "specialty", "location", "experience_years", "languages",
                "work_start", "work_end", "work_days"]
        db.insert("doctors", dict(zip(keys, row)), or_replace=True)


def ensure_slots(start: date | None = None, horizon_days: int | None = None,
                 prebooked_ratio: float = 0.3) -> int:
    """Create 30-minute slots for each doctor's working hours up to the horizon.
    Idempotent. A deterministic fraction of slots is 'blocked' to simulate
    bookings made through other channels, so availability looks realistic."""
    s = get_settings()
    start = start or date.today()
    horizon = horizon_days or s.slot_horizon_days
    step = timedelta(minutes=s.slot_minutes)
    created = 0
    with db.get_conn() as conn:
        existing = {r[0] for r in conn.execute("SELECT slot_id FROM slots")}
        for doc in conn.execute("SELECT * FROM doctors").fetchall():
            days = {int(d) for d in doc["work_days"].split(",")}
            t_start = time.fromisoformat(doc["work_start"])
            t_end = time.fromisoformat(doc["work_end"])
            for offset in range(horizon):
                day = start + timedelta(days=offset)
                if day.weekday() not in days:
                    continue
                cur = datetime.combine(day, t_start)
                while cur + step <= datetime.combine(day, t_end):
                    slot_id = f"{doc['doctor_id']}-{cur:%Y%m%d%H%M}"
                    if slot_id not in existing:
                        status = "blocked" if _stable_fraction(slot_id) < prebooked_ratio else "available"
                        conn.execute("INSERT INTO slots VALUES (?,?,?,?,?)",
                                     (slot_id, doc["doctor_id"], cur.isoformat(sep=" "),
                                      (cur + step).isoformat(sep=" "), status))
                        created += 1
                    cur += step
    return created


def demo_family_records(today: date | None = None) -> dict:
    """A 70-year-old CKD patient (father of the dataset's Rahul Negi)."""
    t = today or date.today()
    visit = (t - timedelta(days=45)).isoformat()
    prior = (t - timedelta(days=210)).isoformat()
    labs_recent = [
        ("Creatinine", 1.9, "mg/dL", "0.70-1.30"), ("eGFR", 36, "mL/min/1.73m2", "> OR = 60"),
        ("Potassium", 5.4, "mmol/L", "3.5-5.3"), ("HbA1c", 7.4, "%", "<5.7"),
        ("Urine albumin/creatinine ratio", 180, "mg/g", "<30"), ("Hemoglobin", 11.2, "g/dL", "13.5-17.5"),
        ("BUN", 32, "mg/dL", "7-25"),
    ]
    labs_prior = [("Creatinine", 1.6, "mg/dL", "0.70-1.30"), ("eGFR", 44, "mL/min/1.73m2", "> OR = 60"),
                  ("HbA1c", 7.9, "%", "<5.7")]
    return {
        "patient": {
            "name": "Suresh Negi", "dob": f"{t.year - 70}-03-02", "gender": "Male",
            "phone": "+91-98110-45671", "address": "Chattarpur, New Delhi",
        },
        "encounters": [{
            "visit_date": prior, "location": "CityCare Family Clinic, Pune", "provider": "Dr. Meera Nair",
            "chief_complaint": "Routine diabetes and blood pressure review",
            "subjective": "69-year-old male with type 2 diabetes (15 years) and hypertension. Reports mild ankle swelling in the evenings. No chest pain or breathlessness.",
            "objective": "BP 148/90, Pulse 78. Mild bilateral pedal edema. No pallor.",
            "assessment": "Diagnosis: Chronic kidney disease stage 3a (N18.31); Type 2 diabetes mellitus with diabetic CKD (E11.22); Hypertensive CKD (I12.9)",
            "plan": "Start Losartan 50mg OD. Continue Metformin 500mg BID. Low-salt diet. Repeat renal function in 6 months.",
        }, {
            "visit_date": visit, "location": "CityCare Family Clinic, Pune", "provider": "Dr. Meera Nair",
            "chief_complaint": "Follow-up of kidney function; fatigue",
            "subjective": "70-year-old male with diabetic kidney disease. Reports fatigue and reduced appetite over 2 months. Urine output normal. Adherent to losartan and metformin.",
            "objective": "BP 142/88, Pulse 80, weight 71 kg. Trace pedal edema.",
            "assessment": "Diagnosis: Chronic kidney disease stage 3b (N18.32); progression from 3a with albuminuria. Mild hyperkalemia. Anemia of CKD suspected.",
            "plan": "Start Empagliflozin 10mg OD. Continue Losartan 50mg OD; recheck potassium in 2 weeks. Reduce Metformin to 500mg OD given eGFR. Refer to nephrology. Dietitian review for low-potassium diet.",
        }],
        "diagnoses": [
            ("Chronic kidney disease stage 3b", "N18.32", "active", visit),
            ("Type 2 diabetes mellitus with diabetic chronic kidney disease", "E11.22", "active", prior),
            ("Hypertensive chronic kidney disease", "I12.9", "active", prior),
            ("Hyperkalemia", "E87.5", "active", visit),
        ],
        "medications": [
            ("Losartan", "50mg OD", "active", prior), ("Empagliflozin", "10mg OD", "active", visit),
            ("Metformin", "500mg OD", "active", visit), ("Atorvastatin", "20mg HS", "active", prior),
        ],
        "labs": [(visit, *x) for x in labs_recent] + [(prior, *x) for x in labs_prior],
        "allergy_note": "Allergy: sulfa drugs (rash).",
    }
