"""Deterministic red-flag screen that runs before planning. An administrative
assistant must never let an emergency wait behind appointment booking."""
from __future__ import annotations

import re

EMERGENCY_PATTERNS = [
    (r"chest pain.*(?:breath|sweat|left arm|jaw|faint)|(?:breath|sweat|left arm|jaw).*chest pain", "possible heart attack"),
    (r"can'?t breathe|cannot breathe|unable to breathe|struggling to breathe|choking", "breathing difficulty"),
    (r"face (?:is )?droop|slurred speech|sudden (?:weakness|numbness)|stroke symptoms", "possible stroke"),
    (r"unconscious|not responding|unresponsive|passed out|collapsed", "loss of consciousness"),
    (r"seizure (?:now|right now|ongoing)|having a seizure|fitting now", "active seizure"),
    (r"severe bleeding|bleeding heavily|won'?t stop bleeding", "severe bleeding"),
    (r"overdose|poison(?:ed|ing)", "poisoning / overdose"),
    (r"suicid|kill (?:myself|himself|herself)|end (?:my|his|her) life|self[- ]harm", "risk of self-harm"),
]
URGENT_PATTERNS = [
    (r"chest pain|high fever.*(?:child|baby|infant)|severe (?:pain|headache)|vomiting blood|blood in (?:stool|urine)",
     "symptom needing prompt medical review"),
]


def screen(text: str) -> dict:
    t = (text or "").lower()
    for pat, label in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            msg = (f"**This may be an emergency ({label}).** Please call your local emergency number now "
                   "(India 112 / US 911) or go to the nearest emergency department. Do not wait for a scheduled appointment.")
            if "self-harm" in label:
                msg += " You can also reach a crisis line (India: Tele-MANAS 14416; US: 988)."
            return {"level": "emergency", "label": label, "message": msg}
    for pat, label in URGENT_PATTERNS:
        if re.search(pat, t):
            return {"level": "urgent", "label": label,
                    "message": f"Note: {label} - if symptoms are severe or worsening, seek same-day care rather than a routine appointment."}
    return {"level": "none", "label": None, "message": ""}
