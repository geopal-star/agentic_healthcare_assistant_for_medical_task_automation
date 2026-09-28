"""Domain knowledge shared by the planner and the tools: condition -> specialty
routing, specialist nouns, and natural-language date/time parsing."""
from __future__ import annotations

import re
from datetime import date, timedelta

from dateutil import parser as dateparser

SPECIALTIES = [
    "Nephrology", "Cardiology", "Endocrinology", "General Medicine", "Pulmonology", "Obstetrics & Gynecology",
    "Neurology", "Rheumatology", "Orthopedics", "Dermatology", "Gastroenterology", "Psychiatry",
    "Nutrition & Dietetics",
]

# Specialist nouns ("nephrologist") map directly to a specialty.
SPECIALIST_WORDS = {
    r"nephrolog\w*|kidney (?:doctor|specialist)|renal specialist": "Nephrology",
    r"cardiolog\w*|heart (?:doctor|specialist)": "Cardiology",
    r"endocrinolog\w*|diabetolog\w*|diabetes (?:doctor|specialist)": "Endocrinology",
    r"pulmonolog\w*|chest physician|lung (?:doctor|specialist)": "Pulmonology",
    r"gyn(?:a)?ecolog\w*|obstetric\w*|ob[- ]?gyn": "Obstetrics & Gynecology",
    r"neurolog\w*": "Neurology",
    r"rheumatolog\w*": "Rheumatology",
    r"orthop(?:a)?edi\w*|bone (?:doctor|specialist)": "Orthopedics",
    r"dermatolog\w*|skin (?:doctor|specialist)": "Dermatology",
    r"gastroenterolog\w*|gastro\b": "Gastroenterology",
    r"psychiatr\w*|psycholog\w*|therapist|counsel\w*": "Psychiatry",
    r"dietitian|dietician|nutritionist": "Nutrition & Dietetics",
    r"general physician|family (?:doctor|physician)|\bgp\b|primary care|physician": "General Medicine",
}

# Conditions / symptoms -> specialty, with a canonical condition name for search.
CONDITIONS = [
    (r"chronic kidney disease|\bckd\b|kidney|renal|dialysis|creatinine|egfr|nephr", "Nephrology", "chronic kidney disease"),
    (r"hypertension|high blood pressure|\bbp\b|blood pressure", "Cardiology", "hypertension"),
    (r"heart|cardiac|palpitation|chest pain|angina|arrhythmia|heart failure|cholesterol", "Cardiology", "heart disease"),
    (r"type 2 diabetes|type 1 diabetes|diabet|blood sugar|hba1c|insulin", "Endocrinology", "diabetes"),
    (r"thyroid|pcos|polycystic|hormone", "Endocrinology", "thyroid and hormonal disorders"),
    (r"asthma|copd|cough|pneumonia|lung|breathing|respiratory|wheez", "Pulmonology", "respiratory infection"),
    (r"pregnan|pap smear|menstrua|period|cervical|gyn", "Obstetrics & Gynecology", "women's health"),
    (r"migraine|headache|seizure|epilep|stroke|parkinson|dementia|alzheimer|numbness", "Neurology", "migraine"),
    (r"rheumatoid|arthritis|lupus|gout|joint pain", "Rheumatology", "arthritis"),
    (r"fracture|back pain|knee|sprain|bone|osteopor|costochondritis", "Orthopedics", "musculoskeletal pain"),
    (r"skin|rash|acne|eczema|psoriasis", "Dermatology", "skin conditions"),
    (r"stomach|abdominal|gastric|acid reflux|gerd|liver|hepatitis|ibs|constipation|diarrh", "Gastroenterology", "digestive disorders"),
    (r"depress|anxiety|mental health|insomnia|panic|stress", "Psychiatry", "depression and anxiety"),
    (r"\bdiet\b|nutrition|weight loss|obesity|meal plan", "Nutrition & Dietetics", "healthy diet"),
    (r"fever|cold|flu|infection|check-?up|routine|wellness|general", "General Medicine", "general health"),
]

# Canonical names used when turning a detected condition into a search query.
CONDITION_CANONICAL = {
    "hypertension": "high blood pressure", "diabetes": "type 2 diabetes",
    "respiratory infection": "upper respiratory infection",
}


def specialty_from_text(text: str) -> str | None:
    t = (text or "").lower()
    for pat, spec in SPECIALIST_WORDS.items():
        if re.search(pat, t):
            return spec
    for pat, spec, _ in CONDITIONS:
        if re.search(pat, t):
            return spec
    for spec in SPECIALTIES:
        if spec.lower() in t:
            return spec
    return None


def conditions_in_text(text: str) -> list[str]:
    t = (text or "").lower()
    out = []
    for pat, _, name in CONDITIONS:
        m = re.search(pat, t)
        if m and name not in out and name != "general health":
            out.append(name)
    return out


def normalize_specialty(s: str | None) -> str | None:
    if not s:
        return None
    for spec in SPECIALTIES:
        if s.strip().lower() == spec.lower():
            return spec
    return specialty_from_text(s)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TIME_WINDOWS = {"morning": (0, 12), "afternoon": (12, 17), "evening": (16, 24)}


def parse_date_window(hint: str | None, today: date | None = None) -> tuple[date | None, date | None]:
    """Turn 'tomorrow', 'next Monday', 'next week', '3 Oct', '2026-10-03' into a date window.
    Returns (None, None) for 'asap' / unknown hints (meaning: earliest available)."""
    today = today or date.today()
    if not hint:
        return None, None
    h = hint.lower().strip()
    if re.search(r"\b(asap|earliest|soonest|as soon as possible|any ?time|whenever)\b", h):
        return None, None
    if "day after tomorrow" in h:
        d = today + timedelta(days=2)
        return d, d
    if "tomorrow" in h:
        d = today + timedelta(days=1)
        return d, d
    if re.search(r"\btoday\b", h):
        return today, today
    if m := re.search(r"in (\d+) (day|week)s?", h):
        d = today + timedelta(days=int(m.group(1)) * (7 if m.group(2) == "week" else 1))
        return (d, d) if m.group(2) == "day" else (d - timedelta(days=d.weekday()), d + timedelta(days=6 - d.weekday()))
    if "next week" in h:
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if "this week" in h:
        return today, today + timedelta(days=6 - today.weekday())
    if "weekend" in h:
        sat = today + timedelta(days=(5 - today.weekday()) % 7)
        return sat, sat + timedelta(days=1)
    for i, wd in enumerate(WEEKDAYS):
        if re.search(rf"\b{wd[:3]}(?:{wd[3:]})?\b", h):
            # "monday" / "next monday" both mean the upcoming occurrence (never today).
            d = today + timedelta(days=(i - today.weekday()) % 7 or 7)
            return d, d
    try:
        d = dateparser.parse(hint, fuzzy=True, default=dateparser.parse(today.isoformat())).date()
        if d < today and not re.search(r"\d{4}", hint):
            d = d.replace(year=d.year + 1)
        return d, d
    except (ValueError, OverflowError):
        return None, None


def time_window(time_of_day: str | None) -> tuple[int, int]:
    if not time_of_day:
        return 0, 24
    t = time_of_day.lower()
    for k, v in TIME_WINDOWS.items():
        if k in t:
            return v
    if m := re.search(r"(\d{1,2})(?::\d{2})?\s*(am|pm)?", t):
        hr = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        return hr, hr + 2
    return 0, 24
