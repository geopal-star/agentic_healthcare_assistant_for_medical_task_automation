"""Parsers that turn the raw dataset (records.xlsx + clinical PDFs) into
structured EHR rows.

Two PDF layouts occur in the dataset:
  1. *Simple H&P note* (sample_report_*.pdf): labelled fields
     (Patient:, DOB:, Subjective Notes:, ..., Diagnosis: X (ICD10)).
  2. *EHR export* (sample_patient.pdf): several progress notes (with page
     footers carrying the encounter id), a past-medical-history page, and
     CCD "History and Physical Note" documents holding medications, lab
     results with reference ranges, and vital signs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pymupdf

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ParsedEncounter:
    encounter_id: str
    visit_date: str | None = None
    location: str | None = None
    provider: str | None = None
    chief_complaint: str | None = None
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    diagnoses: list[dict] = field(default_factory=list)      # {description, icd10, status}
    medications: list[dict] = field(default_factory=list)    # {name, dosage, status, noted_on}
    vitals: list[dict] = field(default_factory=list)         # {name, value, unit, taken_on}


@dataclass
class ParsedDocument:
    """Everything extracted from one PDF (may cover several encounters)."""
    patient: dict[str, Any] = field(default_factory=dict)    # name, dob, gender, phone, address, mrn
    encounters: list[ParsedEncounter] = field(default_factory=list)
    labs: list[dict] = field(default_factory=list)           # {test, value, unit, ref_range, flag, taken_on}
    history_problems: list[str] = field(default_factory=list)
    raw_text: str = ""
    layout: str = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pdf_pages(path_or_bytes: str | Path | bytes) -> list[str]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = pymupdf.open(stream=bytes(path_or_bytes), filetype="pdf")
    else:
        doc = pymupdf.open(str(path_or_bytes))
    with doc:
        return [p.get_text() for p in doc]


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _between(text: str, start: str, ends: list[str]) -> str:
    end_pat = "|".join(ends) if ends else r"\Z"
    m = re.search(rf"{start}\s*:?\s*(.*?)\s*(?:{end_pat}|\Z)", text, re.S | re.I)
    return _clean(m.group(1)) if m else ""


def normalize_date(s: str | None) -> str | None:
    """Accepts 9/7/2022, 03/15/2024, 08-Sep-2022, 'September 20, 2022' -> ISO date."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def age_from_dob(dob_iso: str | None, today: date | None = None) -> int | None:
    if not dob_iso:
        return None
    d = date.fromisoformat(dob_iso)
    t = today or date.today()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def parse_range_flag(value: float, ref: str | None) -> str | None:
    """Return 'H' / 'L' when a lab value is outside its reference range."""
    if value is None or not ref:
        return None
    r = ref.replace("OR =", "=").replace(" ", "")
    try:
        if m := re.fullmatch(r"(-?[\d.]+)-(-?[\d.]+)", r):
            lo, hi = float(m.group(1)), float(m.group(2))
            return "L" if value < lo else "H" if value > hi else None
        if m := re.fullmatch(r"<(=?)(-?[\d.]+)", r):
            limit, inclusive = float(m.group(2)), bool(m.group(1))
            return "H" if (value > limit if inclusive else value >= limit) else None
        if m := re.fullmatch(r">(=?)(-?[\d.]+)", r):
            limit, inclusive = float(m.group(2)), bool(m.group(1))
            return "L" if (value < limit if inclusive else value <= limit) else None
    except ValueError:
        return None
    return None


# Friendly names for the LOINC-style short names used in the CCD export.
LAB_NAMES = {
    "AST": "AST", "ALT": "ALT", "ALP": "Alkaline phosphatase", "Albumin": "Albumin",
    "Calcium": "Calcium", "CO2": "CO2", "Creat": "Creatinine", "Potassium": "Potassium",
    "Prot": "Total protein", "eGFRcr": "eGFR", "BUN": "BUN", "Bilirub": "Bilirubin",
    "Globulin": "Globulin", "Sodium": "Sodium", "Glucose": "Glucose", "Chloride": "Chloride",
    "Platelet": "Platelets", "RDW": "RDW", "WBC": "WBC", "MCHC": "MCHC", "Hct": "Hematocrit",
    "PMV": "MPV", "Hgb": "Hemoglobin", "MCV": "MCV", "MCH": "MCH", "RBC": "RBC",
    "TSH": "TSH", "Trigl": "Triglycerides", "Cholest": "Total cholesterol",
    "NonHDLc": "Non-HDL cholesterol", "HDLc": "HDL cholesterol", "LDLc": "LDL cholesterol",
}


def friendly_lab_name(raw: str) -> str:
    raw = _clean(raw)
    head = raw.split(" ")[0]
    if "/" in head:                       # ratios / differential percentages keep their head
        base = head
    else:
        base = LAB_NAMES.get(head, head)
    if " # " in f" {raw} " and "/" not in head:
        base += " (abs)"
    return base


# ---------------------------------------------------------------------------
# Layout 1: simple H&P note
# ---------------------------------------------------------------------------

_MED_RE = re.compile(
    r"\b([A-Za-z][a-z]{3,})\s+(?:dosage\s+to\s+|dose\s+to\s+)?"
    r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|units?)(?:\s+(?:OD|BID|TID|QID|QD|HS|daily|once daily|twice daily))?)",
    re.I,
)
_DX_RE = re.compile(r"Diagnosis:\s*(.+?)\s*\(([A-Z]\d{2}(?:\.\d{1,4})?)\)", re.I)


def extract_medications(text: str) -> list[dict]:
    meds, seen = [], set()
    for name, dose in _MED_RE.findall(text or ""):
        key = name.lower()
        if key in {"increase", "continue", "start", "take", "reduce", "dosage", "with"} or key in seen:
            continue
        seen.add(key)
        meds.append({"name": name.capitalize(), "dosage": _clean(dose), "status": "active"})
    return meds


def parse_simple_note(text: str) -> ParsedDocument:
    doc = ParsedDocument(raw_text=text, layout="hp_note")
    field_re = lambda label: _clean(m.group(1)) if (m := re.search(rf"{label}:\s*(.+)", text)) else None  # noqa: E731
    dob = normalize_date(field_re("DOB"))
    doc.patient = {
        "name": field_re("Patient"), "dob": dob, "gender": field_re("Gender"),
        "phone": field_re("Phone"), "address": field_re("Address"),
    }
    visit = normalize_date(field_re("Visit Date"))
    sections = ["Subjective Notes", "Objective Notes", "Assessment Notes", "Plan Notes"]
    subj = _between(text, "Subjective Notes", sections[1:])
    obj = _between(text, "Objective Notes", sections[2:])
    assess = _between(text, "Assessment Notes", sections[3:])
    plan = _between(text, "Plan Notes", [])
    enc = ParsedEncounter(
        encounter_id="", visit_date=visit, subjective=subj, objective=obj, assessment=assess, plan=plan,
        chief_complaint=re.split(r"(?<=[.!?])\s", subj)[0] if subj else None,
    )
    for desc, icd in _DX_RE.findall(assess):
        enc.diagnoses.append({"description": _clean(desc), "icd10": icd, "status": "active"})
    enc.medications = [dict(m, noted_on=visit) for m in extract_medications(plan)]
    for name, pat, unit in [("BP", r"BP\s*(\d{2,3}/\d{2,3})", "mmHg"), ("Pulse", r"Pulse\s*(\d+)", "/min"),
                            ("Temperature", r"Temp\s*([\d.]+)", "F")]:
        if m := re.search(pat, obj):
            enc.vitals.append({"name": name, "value": m.group(1), "unit": unit, "taken_on": visit})
    doc.encounters.append(enc)
    return doc


# ---------------------------------------------------------------------------
# Layout 2: multi-encounter EHR export
# ---------------------------------------------------------------------------

_FOOTER_RE = re.compile(r"Patient MRN:\s*(\S+)\s*Encounter ID:\s*(\d+)\s*Page \d+ of \d+", re.S)
_ICD10_RE = re.compile(r"([A-Za-z][^\[\],]*?)\s*\[ICD-10:\s*([A-Z0-9.]+)\]")
_DATE_LINE = re.compile(r"^(\d{1,2}-[A-Z][a-z]{2}-\d{4})\s+\d{1,2}:\d{2}$")
_MEASURE_LINE = re.compile(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s+(\S+)$")


def _parse_progress_note(text: str, encounter_id: str) -> ParsedEncounter:
    head = re.search(r"^\s*(.+?)\s*\n\s*(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*[AP]M", text, re.M)
    enc = ParsedEncounter(encounter_id=encounter_id, visit_date=normalize_date(head.group(2)) if head else None)
    if m := re.search(r"Location:\s*(.+)", text):
        enc.location = _clean(m.group(1))
    if m := re.search(r"Signed electronically by\s*([^,(]+)", text):
        enc.provider = _clean(m.group(1))
    if m := re.search(r"ChiefComplaint:\s*(.+?)(?:\s+-\s+|\s+ACTIVE|\n)", text):
        enc.chief_complaint = _clean(m.group(1))
    enc.subjective = _between(text, "Subjective Notes", ["Objective Notes"])
    enc.objective = _between(text, "Objective Notes", ["Assessment Notes"])
    enc.assessment = _between(text, "Assessment Notes", ["Plan Notes"])
    enc.plan = _between(text, "Plan Notes", [r"\bEncounter\s*\n", "Signed electronically"])
    dx_text = re.sub(r"\[(?:ICD-9|SNOMED):[^\]]*\],?", "", enc.assessment.replace("DIAGNOSIS:", ""))
    for desc, icd in _ICD10_RE.findall(dx_text):
        enc.diagnoses.append({"description": _clean(desc).strip(", "), "icd10": icd, "status": "active"})
    if enc.chief_complaint is None and (m := re.search(r"(?:CC|Concerns today):\s*([^.]+?)(?:\s+[A-Z][a-z]+:|\.)", enc.subjective)):
        enc.chief_complaint = _clean(m.group(1))
    return enc


def _parse_ccd(text: str, doc: ParsedDocument) -> tuple[str | None, list[dict], list[dict]]:
    """Returns (encounter_id, medications, vitals); appends labs to doc."""
    enc_id = m.group(1) if (m := re.search(r"Encounter Id\s*\n\s*(\d+)", text)) else None
    lines = [ln.strip() for ln in text.splitlines()]

    def section(name: str, next_names: list[str]) -> list[str]:
        # Headings also appear in the table of contents; the real section is the last occurrence.
        hits = [k for k, ln in enumerate(lines) if ln == name]
        if not hits:
            return []
        i = hits[-1]
        out = []
        for ln in lines[i + 1:]:
            if ln in next_names:
                break
            out.append(ln)
        return out

    meds = []
    med_lines = section("Medications", ["Procedures", "Immunizations", "Past Medical History"])
    for k, ln in enumerate(med_lines):
        if ln.startswith("Refills:") or ln.startswith("Quantity:"):
            continue
        nxt = med_lines[k + 1] if k + 1 < len(med_lines) else ""
        if ln and (nxt.startswith("Refills:") or nxt.startswith("Quantity:")):
            m = re.match(r"(.*?)\s+(\d[\d.\-]*\s*(?:MG|MCG|MG-MCG|mg|mcg)\b.*)$", ln)
            name, dosage = (m.group(1), m.group(2)) if m else (ln, None)
            block = " ".join(med_lines[k:k + 6])
            ordered = normalize_date(mm.group(1)) if (mm := re.search(r"Ordered:\s*(\S+)", block)) else None
            status = "completed" if re.search(r"Status:\s*Completed", block) else "active"
            meds.append({"name": _clean(name), "dosage": dosage, "status": status, "noted_on": ordered})

    current = None
    last_lab = None
    for ln in section("Results", ["Vital Signs"]):
        if m := _DATE_LINE.match(ln):
            current = normalize_date(m.group(1))
        elif ln.startswith("Range:") and last_lab is not None:
            last_lab["ref_range"] = ln.split(":", 1)[1].strip()
            last_lab["flag"] = parse_range_flag(last_lab["value"], last_lab["ref_range"])
        elif (m := _MEASURE_LINE.match(ln)) and not ln.startswith(("TEST NAME", "RESULT")):
            last_lab = {"test": friendly_lab_name(m.group(1)), "raw_name": _clean(m.group(1)),
                        "value": float(m.group(2)), "unit": m.group(3), "ref_range": None,
                        "flag": None, "taken_on": current}
            doc.labs.append(last_lab)
        else:
            last_lab = None if ln else last_lab

    vitals, current = [], None
    for ln in section("Vital Signs", ["Encounters"]):
        if m := _DATE_LINE.match(ln):
            current = normalize_date(m.group(1))
        elif m := _MEASURE_LINE.match(ln):
            vitals.append({"name": _clean(m.group(1)), "value": m.group(2), "unit": m.group(3), "taken_on": current})
    return enc_id, meds, vitals


def parse_ehr_export(pages: list[str]) -> ParsedDocument:
    full = "\n".join(pages)
    doc = ParsedDocument(raw_text=full, layout="ehr_export")
    if m := re.search(r"Patient\s*\n\s*(.+?)\s*\n\s*Date of birth\s*\n\s*(.+?)\s*\n\s*Sex\s*\n\s*(\w+)", full):
        doc.patient.update(name=_clean(m.group(1)), dob=normalize_date(_clean(m.group(2))), gender=m.group(3))
    if m := re.search(r"Patient #:\s*(\S+)", full):
        doc.patient["mrn"] = m.group(1)
    if m := re.search(r"Home:\s*\n\s*(.+?)\s*\n\s*(.+?)\s*\n\s*Tel:\s*(\S+)", full):
        doc.patient.update(address=f"{_clean(m.group(1))}, {_clean(m.group(2))}", phone=m.group(3))

    # Group footer-bearing pages by encounter id; CCD documents stand alone.
    notes: dict[str, str] = {}
    ccds: list[str] = []
    for page in pages:
        foot = _FOOTER_RE.search(page)
        if page.lstrip().startswith("History and Physical Note"):
            ccds.append(page)
        elif ccds and not foot and not re.search(r"Subjective Notes", page):
            ccds[-1] += "\n" + page                       # continuation of a CCD
        elif foot:
            notes[foot.group(2)] = notes.get(foot.group(2), "") + "\n" + _FOOTER_RE.sub("", page)

    encounters: dict[str, ParsedEncounter] = {}
    for enc_id, text in notes.items():
        note_part = text.split("Past Medical History")[0] if "Subjective Notes" in text else ""
        if note_part:
            encounters[enc_id] = _parse_progress_note(note_part, enc_id)
        if "On-Going Medical Problems" in text:
            probs = _between(text, "On-Going Medical Problems", ["Past Medical History"])
            doc.history_problems += [p for p in [probs] if p]

    for ccd in ccds:
        enc_id, meds, vitals = _parse_ccd(ccd, doc)
        if enc_id and enc_id in encounters:
            encounters[enc_id].medications += meds
            encounters[enc_id].vitals += vitals

    for enc in encounters.values():
        if m := re.search(r"PMH:\s*(.+?)\s+(?:FH|FMH|PSH):", enc.subjective):
            doc.history_problems += [p.strip() for p in m.group(1).split(",") if p.strip()]
    doc.history_problems = list(dict.fromkeys(doc.history_problems))
    doc.encounters = sorted(encounters.values(), key=lambda e: e.visit_date or "")
    return doc


def parse_pdf(path_or_bytes: str | Path | bytes) -> ParsedDocument:
    pages = pdf_pages(path_or_bytes)
    full = "\n".join(pages)
    if _FOOTER_RE.search(full) or "Table of Contents" in full:
        return parse_ehr_export(pages)
    if re.search(r"Patient:\s*\S", full) and re.search(r"Subjective Notes", full):
        return parse_simple_note(full)
    return ParsedDocument(raw_text=full, layout="unstructured")
