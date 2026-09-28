"""Planner: interprets a (possibly multi-part) request and decomposes it into
ordered sub-goals, each mapped to one tool.

* ``llm_plan``        - structured-output LLM planner (primary)
* ``heuristic_plan``  - deterministic keyword planner (offline fallback)
* ``validate_plan``   - guard-rails applied to *either* plan: unknown tools,
                        missing identify step, role permissions, dependencies.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import records
from ..context import Actor
from ..domain import CONDITION_CANONICAL, conditions_in_text, specialty_from_text
from ..llm import get_llm
from ..prompts import PLANNER_SYSTEM, PLANNER_USER
from ..tools import TOOLS, catalog_text

ToolName = Literal[tuple(TOOLS)]  # type: ignore[valid-type]


class FieldUpdate(BaseModel):
    field: str
    value: str


class StepArgs(BaseModel):
    """Union of all tool arguments (flat, so every provider's structured output handles it)."""
    name: str | None = None
    relation: str | None = None
    age: int | None = None
    focus: str | None = None
    query: str | None = None
    specialty: str | None = None
    condition: str | None = None
    doctor_id: str | None = None
    appointment_id: str | None = None
    preferred_date: str | None = None
    time_of_day: str | None = None
    reason: str | None = None
    location: str | None = None
    note_text: str | None = None
    visit_date: str | None = None
    updates: list[FieldUpdate] | None = None
    gender: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    dob: str | None = None
    allergies: str | None = None


class PlanStep(BaseModel):
    step: int
    goal: str = Field(description="the sub-goal in plain language")
    tool: ToolName
    args: StepArgs = Field(default_factory=StepArgs)
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    intent_summary: str = Field(description="one sentence: what the user wants overall")
    patient_name: str | None = Field(None, description="patient's name if stated")
    patient_relation: str | None = Field(None, description="relation to the user if stated (father, self, ...)")
    steps: list[PlanStep]
    needs_clarification: str | None = None


# ---------------------------------------------------------------------------
# Heuristic planner (offline fallback)
# ---------------------------------------------------------------------------

_RELATION_RE = re.compile(r"\bmy\s+(\d{1,3}[- ]year[- ]old\s+)?(father|dad|daddy|papa|mother|mom|mum|mummy|wife|husband|"
                          r"spouse|son|daughter|grandfather|grandmother|grandpa|grandma|brother|sister)\b", re.I)
_INTENTS = {
    "cancel": r"\bcancel\w*\b",
    "reschedule": r"\breschedul\w*|\bmove (?:my|his|her|the) appointment|\bpostpone\b|\bchange (?:my|his|her|the) appointment",
    "list_appts": r"\b(?:show|list|view|check|upcoming|what are|when is|when's|do i have|does (?:he|she) have)\b"
                  r"[^.]*\bappointments?\b",
    "book": r"\b(?:book|schedule|make an appointment|fix an appointment|consult|see an?|visit an?)\b|"
            r"\bneeds? (?:an? |to see )?[\w ]*?(?:appointment|doctor|specialist|\w+(?:ologist|ician|itian))\b|"
            r"\b\w+(?:ologist|ician|itian) appointment\b",
    "history": r"\b(?:medical history|history|past diagnos\w*|records?|summari[sz]e (?:his|her|my|the patient)|"
               r"what medications|current medications|previous visits?|health summary)\b",
    "search": r"\b(?:latest|treatment|treatments|therap\w*|what is|what are|information|info on|symptoms?|causes?|"
              r"prevent\w*|guidelines?|manage(?:ment)?|side effects?|how to treat|tell me about|risk factors?|diet for)\b",
    "add_note": r"\b(?:add|record|log|note)\b.*\b(?:note|visit|observation|diagnos\w*|complain\w*)\b",
    "update": r"\bupdate\b|\bchange (?:the |his |her )?(?:phone|address|email|allerg)|\bnew (?:phone|address|email)\b",
    "register": r"\b(?:register|create|add)\b.*\bnew patient\b",
}


def _detect_patient_ref(query: str, actor: Actor, context: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    q = query.lower()
    age = int(m.group(1)) if (m := re.search(r"(\d{1,3})[- ]year[- ]old", q)) else None
    if m := _RELATION_RE.search(query):
        return None, records.normalize_relation(m.group(2)), age
    for p in records.list_patients():
        full = p["name"].lower()
        first = full.split()[0]
        if p["patient_id"] != actor.user_id and (full in q or (len(first) >= 4 and re.search(rf"\b{first}\b", q))):
            return p["name"], None, age
    if re.search(r"\b(?:i|me|my|myself)\b", q) and actor.role == "patient":
        return None, "self", age
    if re.search(r"\b(?:he|she|him|his|her)\b", q) and not context.get("active_patient"):
        deps = records.get_dependents(actor.user_id)
        if len(deps) == 1:                # "his" with a single linked dependent is unambiguous
            return None, deps[0]["relation"], age
    return None, None, age


def _search_query(query: str, conditions: list[str]) -> str:
    q = query.lower()
    topic = next((CONDITION_CANONICAL.get(c, c) for c in conditions), None)
    aspect = next((w for w in ("latest treatment", "treatment", "symptoms", "causes", "prevention", "diet",
                               "side effects", "risk factors", "management") if w in q), None)
    if topic:
        return f"{topic} {aspect or 'overview treatment'}"
    # No known condition: use the clause that carries the question.
    clauses = re.split(r"[.?!]|\balso\b|\band\b", query)
    best = next((c for c in clauses if re.search(_INTENTS["search"], c, re.I)), query)
    return re.sub(r"\b(?:can you|could you|please|summari[sz]e|tell me about|me)\b", "", best, flags=re.I).strip() or query


def heuristic_plan(query: str, actor: Actor, context: dict[str, Any]) -> Plan:
    q = query.lower()
    intents = {k for k, pat in _INTENTS.items() if re.search(pat, q, re.I)}
    if intents & {"cancel", "reschedule"}:
        intents.discard("book")
    if "list_appts" in intents and not re.search(r"\b(?:book|schedule)\b", q):
        intents.discard("book")
    if intents & {"update", "add_note", "register"}:
        intents -= {"history", "search"} if not re.search(r"\blatest|treatment\b", q) else set()
    name, relation, age = _detect_patient_ref(query, actor, context)
    conditions = conditions_in_text(query)
    specialty = specialty_from_text(query) if "book" in intents else None
    time_of_day = m.group(0) if (m := re.search(r"\b(morning|afternoon|evening)\b", q)) else None
    date_hint = m.group(0) if (m := re.search(
        r"\b(?:today|tomorrow|day after tomorrow|next week|this week|weekend|asap|"
        r"(?:next |this |on )?(?:mon|tues|wednes|thurs|fri|satur|sun)day|in \d+ (?:days?|weeks?)|"
        r"\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)? (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)\b", q)) else None

    steps: list[PlanStep] = []

    def add(tool: str, goal: str, deps: list[int] | None = None, **args: Any) -> int:
        steps.append(PlanStep(step=len(steps) + 1, goal=goal, tool=tool, args=StepArgs(**args), depends_on=deps or []))
        return len(steps)

    patient_needed = bool(intents & {"book", "history", "cancel", "reschedule", "list_appts", "update", "add_note"})
    ident = None
    if patient_needed and "register" not in intents:
        who = name or relation or ("the active patient" if context.get("active_patient") else "the user")
        if name or relation:
            ident = add("identify_patient", f"Identify the patient ({who})", name=name, relation=relation, age=age)
    deps = [ident] if ident else []

    if "register" in intents:
        nm = m.group(1).strip() if (m := re.search(r"(?:named|called|name is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", query)) else None
        add("register_patient", "Register the new patient", name=nm or "Unknown", age=age)
    if "history" in intents or "book" in intents:
        add("get_patient_history", "Retrieve and summarise the patient's medical history", deps,
            focus=conditions[0] if conditions else None)
    if "update" in intents:
        ups = []
        if m := re.search(r"phone\b.*?(\+?\d[\d\- ]{6,}\d)", query, re.I):
            ups.append(FieldUpdate(field="phone", value=m.group(1).strip()))
        if m := re.search(r"address\b.*?(?:\bto\b|:|\bis\b)\s*(.+?)(?:\.\s|\.$|$)", query, re.I):
            ups.append(FieldUpdate(field="address", value=m.group(1).strip()))
        if m := re.search(r"email\b.*?(\S+@\S+)", query, re.I):
            ups.append(FieldUpdate(field="email", value=m.group(1).strip()))
        if m := re.search(r"allerg\w*\s+(?:to\s+)?([^.]+)", query, re.I):
            ups.append(FieldUpdate(field="allergies", value=m.group(1).strip()))
        add("update_patient_record", "Update the patient's record", deps, updates=ups)
    if "add_note" in intents and "update" not in intents:
        note = query.split(":", 1)[1].strip() if ":" in query else query
        add("add_clinical_note", "Add the clinical note to the patient's record", deps, note_text=note)
    if "cancel" in intents:
        add("cancel_appointment", "Cancel the appointment", deps,
            appointment_id=m.group(0) if (m := re.search(r"APT-[A-Z0-9]+", query)) else None,
            specialty=specialty_from_text(query))
    if "reschedule" in intents:
        add("reschedule_appointment", "Reschedule the appointment", deps,
            specialty=specialty_from_text(query), preferred_date=date_hint, time_of_day=time_of_day)
    if "list_appts" in intents:
        add("list_appointments", "List upcoming appointments", deps)
    if "book" in intents:
        add("book_appointment", f"Book a {specialty or 'suitable'} appointment", deps,
            specialty=specialty, condition=conditions[0] if conditions else None,
            preferred_date=date_hint, time_of_day=time_of_day,
            reason=(conditions[0] + " consultation") if conditions else None)
    if "search" in intents and not (intents & {"update", "add_note"}):
        add("medical_info_search", "Search trusted sources and summarise the medical information",
            query=_search_query(query, conditions))
    if not steps and conditions:
        add("medical_info_search", "Search trusted sources", query=_search_query(query, conditions))
    clarification = None if steps else ("I can book appointments, summarise medical history, update records, or look up "
                                         "disease information. Could you tell me what you need?")
    return Plan(intent_summary=", ".join(sorted(intents)) or "unclear", patient_name=name,
                patient_relation=relation, steps=steps, needs_clarification=clarification)


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------

def _fmt_context(actor: Actor, context: dict[str, Any]) -> dict[str, str]:
    deps = context.get("dependents") or []
    ap = context.get("active_patient")
    return {
        "today": date.today().strftime("%A %Y-%m-%d"),
        "actor": f"{actor.name} (id {actor.user_id}, role {actor.role})",
        "dependents": ", ".join(f"{d['relation']}: {d['name']} ({d['patient_id']}, {d.get('age')}y)" for d in deps) or "none",
        "active_patient": f"{ap['name']} ({ap['patient_id']})" if ap else "none",
        "memories": "\n".join(f"  - {m}" for m in context.get("memories", [])) or "none",
        "history": "\n".join(f"  {r}: {c[:300]}" for r, c in context.get("conversation", [])[-6:]) or "none",
    }


def llm_plan(query: str, actor: Actor, context: dict[str, Any]) -> tuple[Plan, str]:
    system = PLANNER_SYSTEM.format(tool_catalog=catalog_text(), **_fmt_context(actor, context))
    return get_llm().structured("planner", system, PLANNER_USER.format(query=query), Plan,
                                fallback=lambda: heuristic_plan(query, actor, context))


# ---------------------------------------------------------------------------
# Validation / repair
# ---------------------------------------------------------------------------

def validate_plan(plan: Plan, actor: Actor, context: dict[str, Any], max_steps: int = 8) -> tuple[Plan, list[str]]:
    """Enforce invariants on any plan; returns (plan, list of repairs made)."""
    repairs: list[str] = []
    steps: list[PlanStep] = []
    seen = set()
    for s in plan.steps:
        spec = TOOLS.get(s.tool)
        if spec is None:
            repairs.append(f"dropped unknown tool {s.tool}")
            continue
        if spec.staff_only and not actor.can_edit_records:
            repairs.append(f"removed {s.tool}: requires attendant/doctor role")
            plan.needs_clarification = (plan.needs_clarification or "") + \
                f" Updating medical records requires an attendant or doctor (you are signed in as {actor.role})."
            continue
        key = (s.tool, s.args.model_dump_json(exclude_none=True))
        if key in seen:
            repairs.append(f"dropped duplicate {s.tool}")
            continue
        seen.add(key)
        steps.append(s)

    needs_patient = any(TOOLS[s.tool].needs_patient for s in steps)
    has_ident = any(s.tool == "identify_patient" for s in steps)
    if needs_patient and not has_ident:
        if plan.patient_name or plan.patient_relation:
            steps.insert(0, PlanStep(step=0, goal="Identify the patient", tool="identify_patient",
                                     args=StepArgs(name=plan.patient_name, relation=plan.patient_relation)))
            repairs.append("inserted identify_patient step")
        elif not context.get("active_patient"):
            steps.insert(0, PlanStep(step=0, goal="Identify the patient (self)", tool="identify_patient",
                                     args=StepArgs(relation="self")))
            repairs.append("inserted identify_patient(self) step")
        else:
            repairs.append(f"using active patient from conversation: {context['active_patient']['name']}")

    # identify must run first; renumber and remap dependencies.
    steps.sort(key=lambda s: 0 if s.tool == "identify_patient" else 1)
    steps = steps[:max_steps]
    old_to_new = {s.step: i + 1 for i, s in enumerate(steps)}
    ident_no = next((i + 1 for i, s in enumerate(steps) if s.tool == "identify_patient"), None)
    for i, s in enumerate(steps):
        s.depends_on = sorted({old_to_new[d] for d in s.depends_on if d in old_to_new and old_to_new[d] < i + 1})
        if ident_no and TOOLS[s.tool].needs_patient and s.tool != "identify_patient" and ident_no not in s.depends_on:
            s.depends_on = sorted({*s.depends_on, ident_no})
        s.step = i + 1
    plan.steps = steps
    if plan.needs_clarification:
        plan.needs_clarification = plan.needs_clarification.strip()
    return plan, repairs
