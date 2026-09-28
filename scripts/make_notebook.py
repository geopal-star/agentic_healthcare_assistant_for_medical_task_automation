"""Generate the capstone walkthrough notebook (Part 1 + Part 2 narrative).

    python scripts/make_notebook.py            # writes notebooks/Agentic_Healthcare_Assistant.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/Agentic_Healthcare_Assistant.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Agentic Healthcare Assistant for Medical Task Automation
**Applied Generative AI Specialisation – Capstone**

A virtual medical assistant that **books appointments**, **manages medical records**, **summarises medical histories**
and **searches trusted medical sources** (MedlinePlus, WHO). It is built as a LangGraph agent with a planner,
tool executor, FAISS-backed long-term memory and a RAG pipeline, plus an LLMOps layer (evaluation, logging) and
a Streamlit dashboard.

This notebook walks through the system in the order of the problem statement:

| Part | Step | Section |
|---|---|---|
| 1 | Agent planning & goal decomposition | §4 |
| 1 | Tool & memory setup (Doctor Schedule API, EHR DB, web search, FAISS, memory) | §2–3 |
| 1 | Prompt engineering & task chaining | §4 |
| 1 | Agent execution flow (sample scenario) | §5 |
| 2 | Model evaluation (QAEvalChain + metrics) | §6 |
| 2 | Streamlit dashboard, memory & logs interface | §7 |

> Runs without an API key (deterministic offline mode). Put `GROQ_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`)
> in `.env` to switch the planner, summariser, RAG answerer and evaluator to a real LLM."""),
    code("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
pd.set_option("display.max_colwidth", 120)

from healthcare_agent.config import get_settings
from healthcare_agent.llm import get_llm
print("LLM:", get_llm().label)
print("Dataset:", sorted(p.name for p in get_settings().raw_dir.iterdir()))"""),

    md("""## 1. Dataset & ingestion
The dataset has `records.xlsx` (one row **per encounter** – Rebeca Nagle appears three times) and two PDF layouts:
simple *History & Physical* notes, and a 12-page EHR export for Rebeca Nagle containing progress notes, a
past-medical-history page and CCD documents with **lab results + reference ranges** and **vital signs**.
Custom parsers turn both into structured rows (encounters, ICD-10 diagnoses, medications, flagged labs, vitals)."""),
    code("""raw = pd.read_excel(get_settings().raw_dir / "records.xlsx")
print(raw.shape, "-> unique patients:", raw.drop_duplicates(["Name", "Phone_number"]).shape[0])
raw[["Name", "Age", "Gender", "Address", "Summary"]]"""),
    code("""from healthcare_agent.ingest import parse_pdf
doc = parse_pdf(get_settings().raw_dir / "sample_patient.pdf")
print("layout:", doc.layout, "| patient:", doc.patient)
for e in doc.encounters:
    print(f"  encounter {e.encounter_id} {e.visit_date}: {e.chief_complaint} -> {[d['description'] + ' ' + d['icd10'] for d in e.diagnoses]}")
labs = pd.DataFrame(doc.labs)
print(f"{len(labs)} lab results parsed; out of range:")
labs[labs.flag.notna()][["taken_on", "test", "value", "unit", "ref_range", "flag"]]"""),
    md("""### Build the EHR database and vector indexes
`build()` loads the spreadsheet, parses every PDF, links documents to patients, adds the synthetic
**doctor directory + rolling slot calendar** (the "Doctor Schedule API" backend – not part of the dataset) and one
synthetic demo patient needed by the reference scenario (**Suresh Negi, 70, CKD stage 3b – father of Rahul Negi**;
flagged `source=synthetic_demo`), then embeds everything into FAISS."""),
    code("""from healthcare_agent.build import build
stats = build(reset=True)
stats"""),
    code("""from healthcare_agent import db, records
pd.DataFrame(records.list_patients())[["patient_id", "name", "age", "gender", "source", "allergies"]]"""),
    code("""pd.DataFrame(db.query('''SELECT d.doctor_id, d.name, d.specialty, d.location, d.work_start || '-' || d.work_end AS hours,
       SUM(s.status='available') AS open_slots, SUM(s.status='blocked') AS held_slots
FROM doctors d JOIN slots s USING(doctor_id) GROUP BY d.doctor_id'''))"""),

    md("""## 2. Tools (the agent's APIs)
Every tool is a plain Python function wrapped by `traced_tool`, which logs arguments, result, success and latency
to SQLite (the source of the tool-success metrics). Business failures (no slot, not authorised) return
`{"ok": False, "error": ...}` so the executor can react.

* **Doctor Schedule API** – `find_doctors`, `check_availability`, `book_appointment` (atomic, idempotent),
  `cancel_appointment`, `reschedule_appointment` (book-then-release), `list_appointments`
* **EHR / Patient DB** – `identify_patient` (relation / fuzzy name / self), `get_patient_history`,
  `search_patient_notes`, staff-only `register_patient`, `update_patient_record`, `add_clinical_note`
* **Medical search** – `medical_info_search`: MedlinePlus Web Service + WHO (site-restricted search & fact-sheet
  extraction) → FAISS → grounded, cited LLM answer

Access control: caregivers see only themselves and linked dependents; only attendants/doctors can edit records."""),
    code("""from healthcare_agent.tools import catalog_text
print(catalog_text())"""),
    code("""from healthcare_agent.context import Actor, current_actor
from healthcare_agent.tools.patient import identify_patient, get_patient_history
current_actor.set(Actor("P001", "Rahul Negi", "patient"))
print(identify_patient(relation="father"))
print(identify_patient(name="Ramesh Kulkarni"))   # not his dependent -> privacy guard"""),
    code("""from healthcare_agent.tools.scheduling import check_availability
res = check_availability(specialty="Nephrology", preferred_date="next week", time_of_day="morning", limit=4)
pd.DataFrame(res["slots"])[["doctor_name", "start_ts", "location"]]"""),

    md("""## 3. Memory
| Memory | Where | Used for |
|---|---|---|
| Short-term (conversation) | LangGraph state persisted per `thread_id` by the checkpointer | follow-ups: *"also book a dietitian **for him**"* |
| Long-term (patient) | SQLite `memories` + FAISS `patient_memory` (profiles, encounter chunks, learned facts) | context recall before planning, history summaries |
| Knowledge | FAISS `medical_knowledge` (MedlinePlus / WHO chunks, cached) | RAG answers |

Embeddings: `BAAI/bge-small-en-v1.5` (normalised → cosine similarity via `IndexFlatIP`)."""),
    code("""from healthcare_agent import memory
for h in memory.recall("falling kidney function and high potassium", k=3, min_score=0):
    print(round(h["score"], 3), h["metadata"]["patient_name"], "|", h["text"][:110])"""),

    md("""## 4. Planning, goal decomposition & prompt chaining
The graph: `safety → context → planner → executor (one tool per visit, loops) → synthesizer → memory`.

* **Planner** – structured-output LLM call returning a typed `Plan` (sub-goals, tool, args, dependencies).
  The system prompt injects the tool catalogue, today's date, the acting user, their dependents, the active patient
  and **recalled memories** (memory lookup in the prompt). Offline, a keyword planner produces the same schema.
* **Plan validation/repair** (applies to any plan) – drops unknown/duplicate tools, removes staff-only steps for
  patients, inserts a missing `identify_patient` step, wires dependencies.
* **Chain**: planner → tools → history-summary prompt / medical-RAG prompt → synthesis prompt → memory-extraction prompt."""),
    code("""from healthcare_agent import prompts
print(prompts.PLANNER_SYSTEM[:1800], "...")"""),
    code("""from healthcare_agent.agent.planner import heuristic_plan, llm_plan, validate_plan
SAMPLE = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
          "Also, can you summarize latest treatment methods?")
actor = Actor("P001", "Rahul Negi", "patient")
ctx = {"dependents": [{k: d[k] for k in ("relation", "patient_id", "name", "age")} for d in records.get_dependents("P001")],
       "active_patient": None, "memories": [], "conversation": []}
plan, mode = llm_plan(SAMPLE, actor, ctx) if get_llm().live else (heuristic_plan(SAMPLE, actor, ctx), "offline")
plan, repairs = validate_plan(plan, actor, ctx)
print("planner mode:", mode, "| repairs:", repairs)
pd.DataFrame([{"step": s.step, "sub-goal": s.goal, "tool": s.tool,
               "args": s.args.model_dump(exclude_none=True), "depends_on": s.depends_on} for s in plan.steps])"""),

    md("""## 5. Agent execution – the reference scenario
Expected workflow from the problem statement: **(1)** identify patient & context → **(2)** retrieve father's
history → **(3)** query doctor calendar & book → **(4)** search & summarise treatments via RAG."""),
    code("""from healthcare_agent.agent import HealthcareAgent
agent = HealthcareAgent()
result = agent.run(SAMPLE, actor, thread_id="notebook-demo")
print(f"success={result['success']}  steps={result['steps_ok']}/{result['steps_total']}  latency={result['latency_ms']} ms")
pd.DataFrame([{"step": s["step"], "tool": s["tool"], "status": s["status"], "ms": s["ms"]} for s in result["steps"]])"""),
    code("""from IPython.display import Markdown
Markdown(result["answer"])"""),
    code("""# Graph path with per-node timings (the same trace the Streamlit "Memory & logs" page shows)
pd.DataFrame([{"node": t["node"], "ms": t["ms"], "detail": json.dumps(t["detail"], default=str)[:150]} for t in result["trace"]])"""),
    md("### Multi-turn follow-up (short-term memory) and long-term memory written"),
    code("""follow = agent.run("Also book a dietitian for him next week in the morning", actor, thread_id="notebook-demo")
print(follow["repairs"])
print(follow["steps"][-1]["result"]["appointment"])
pd.DataFrame(memory.list_memories(result["active_patient"]["patient_id"]))[["created_at", "kind", "content"]]"""),
    md("### Guard-rails"),
    code("""for who, q in [(Actor("P001", "Rahul Negi", "patient"), "My father has chest pain and is sweating and short of breath"),
              (Actor("P004", "Anjali Mehra", "patient"), "Update my address to 14 MG Road, Pune"),
              (Actor("P004", "Anjali Mehra", "patient"), "Show me the medical history of Ramesh Kulkarni")]:
    r = agent.run(q, who, thread_id="guard-" + who.user_id + q[:5])
    print(f"> {q}\\n  safety={r['safety']['level']} repairs={r['repairs']}\\n  {r['answer'][:230]!r}\\n")"""),

    md("""## 6. Evaluation (LLMOps)
`run_evaluation()` scores each module against golden sets and stores case-level rows in `eval_results`:

* **planner** – tool precision/recall/F1, exact match, specialty accuracy, patient-identification accuracy (15 queries)
* **summary** – generated summaries vs the `Summary` column of records.xlsx: semantic similarity, ROUGE-L recall,
  token F1, and **QAEvalChain** CORRECT/INCORRECT when an LLM is configured
* **medical_qa** – RAG answers vs references: similarity, key-fact coverage, groundedness (share of answer
  sentences supported by retrieved passages), citation presence, **QAEvalChain**
* **retrieval** – FAISS hit@1/@3 and MRR for patient memory; hit@3 for the knowledge base
* **booking** – success vs expected outcome, constraint satisfaction, idempotency, race safety (bookings rolled back)
* **tools** – live per-tool success rate & latency from the logs"""),
    code("""from healthcare_agent.evaluation.runner import run_evaluation
report = run_evaluation(verbose=False)
rows = [{"module": m, "metric": k, "mean": v} for m, mets in report["summary"].items() for k, v in mets.items()]
pd.DataFrame(rows)"""),
    code("""import plotly.graph_objects as go
df = pd.DataFrame(rows); df = df[df.module != "tools"]
fig = go.Figure(go.Bar(y=df.module + " · " + df.metric, x=df["mean"], orientation="h", marker_color="#2a78d6",
                       text=df["mean"].round(2), textposition="outside"))
fig.update_layout(title=f"Evaluation – {report['llm']}", height=620, xaxis_range=[0, 1.15],
                  yaxis_autorange="reversed", margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white")
fig"""),
    code("""cases = pd.DataFrame(report["rows"])
cases[cases.module == "medical_qa"].pivot_table(index="case_id", columns="metric", values="score").round(2)"""),

    md("""## 7. Streamlit dashboard
```bash
streamlit run app.py
```
| Page | Requirement covered |
|---|---|
| 💬 Chat with the assistant | live plan + execution trace per turn, example prompts per role |
| 🧪 Scenario tester | interactive elements to test user scenarios (11 curated, incl. guard-rails) |
| 🧑 Patient view | record, alerts, lab trends, appointments, memory; staff: update fields, add notes, upload PDFs, register |
| 🩺 Doctor view | calendar utilisation, upcoming patients with clinical briefing |
| 📅 Appointment tracker | real-time (auto-refresh) tracking, filters, cancellation |
| 📚 Medical information | trusted-source RAG search, latest retrieved information, knowledge base |
| 📊 Evaluation & metrics | tool success & latency, booking success, LLM call modes, evaluation suite runs |
| 🧠 Memory & logs | agent memory traces, planning breakdowns, tool / LLM logs with success-failure filters |"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parent.parent / "notebooks" / "Agentic_Healthcare_Assistant.ipynb"
nbf.write(nb, out)
print("wrote", out)
