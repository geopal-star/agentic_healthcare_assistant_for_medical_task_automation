"""Tool registry. The planner sees ``catalog_text()``; the executor calls
``TOOLS[name].func(**kwargs)``. Every function is wrapped by ``traced_tool``
so success/failure and latency are logged per call."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .medical_search import medical_info_search
from .patient import (add_clinical_note, get_patient_history, identify_patient,
                      register_patient, search_patient_notes, update_patient_record)
from .scheduling import (book_appointment, cancel_appointment, check_availability,
                         find_doctors, list_appointments, reschedule_appointment)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., dict]
    description: str
    args: str
    module: str               # grouping used by the evaluation dashboard
    needs_patient: bool = False
    staff_only: bool = False

    @property
    def params(self) -> set[str]:
        return set(inspect.signature(inspect.unwrap(self.func)).parameters)


_SPECS = [
    ToolSpec("identify_patient", identify_patient,
             "Resolve which patient the request is about (by relation to the user, by name, or self).",
             "name?, relation? (father/mother/spouse/son/daughter/self), age?", "patient"),
    ToolSpec("get_patient_history", get_patient_history,
             "Retrieve and summarise the patient's diagnoses, treatments, medications and alerts from the EHR.",
             "focus? (e.g. 'kidney disease')", "history", needs_patient=True),
    ToolSpec("search_patient_notes", search_patient_notes,
             "Semantic search over the patient's past notes and remembered facts.",
             "query", "history", needs_patient=True),
    ToolSpec("find_doctors", find_doctors,
             "List doctors for a specialty or condition.", "specialty?, condition?, location?", "scheduling"),
    ToolSpec("check_availability", check_availability,
             "List open appointment slots for a specialty/doctor and optional date/time preference.",
             "specialty?, condition?, doctor_id?, preferred_date?, time_of_day?", "scheduling"),
    ToolSpec("book_appointment", book_appointment,
             "Book the earliest matching slot for the patient (idempotent; will not double-book).",
             "specialty?, condition?, doctor_id?, preferred_date?, time_of_day?, reason?", "scheduling",
             needs_patient=True),
    ToolSpec("cancel_appointment", cancel_appointment,
             "Cancel an appointment by id, or the patient's next appointment (optionally of a specialty).",
             "appointment_id?, specialty?", "scheduling", needs_patient=True),
    ToolSpec("reschedule_appointment", reschedule_appointment,
             "Move an existing appointment to a new date/time preference.",
             "appointment_id?, specialty?, preferred_date?, time_of_day?", "scheduling", needs_patient=True),
    ToolSpec("list_appointments", list_appointments,
             "List the patient's upcoming appointments.", "(none)", "scheduling", needs_patient=True),
    ToolSpec("medical_info_search", medical_info_search,
             "Search trusted sources (MedlinePlus, WHO) and return a cited summary about a disease, treatment, "
             "symptom or prevention topic.", "query", "search"),
    ToolSpec("register_patient", register_patient,
             "Create a new patient record (staff only).", "name, age?, gender?, phone?, email?, address?, dob?, allergies?",
             "records", staff_only=True),
    ToolSpec("update_patient_record", update_patient_record,
             "Update structured patient fields such as phone, address, allergies (staff only).",
             "updates: list of {field, value}", "records", needs_patient=True, staff_only=True),
    ToolSpec("add_clinical_note", add_clinical_note,
             "Add a free-text clinical note; structure is extracted automatically (staff only).",
             "note_text, visit_date?", "records", needs_patient=True, staff_only=True),
]

TOOLS: dict[str, ToolSpec] = {s.name: s for s in _SPECS}


def catalog_text() -> str:
    return "\n".join(f"- {s.name}({s.args}): {s.description}" for s in _SPECS)


def call_tool(name: str, kwargs: dict[str, Any]) -> dict:
    spec = TOOLS[name]
    accepted = spec.params
    clean = {k: v for k, v in kwargs.items() if k in accepted and v not in (None, "", [])}
    return spec.func(**clean)
