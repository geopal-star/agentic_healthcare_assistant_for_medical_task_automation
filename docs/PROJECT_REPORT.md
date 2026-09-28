# Project Report: Agentic Healthcare Assistant for Medical Task Automation

*Applied Generative AI Specialisation: Capstone*

## 1. Problem and objective

Patient-care administration is spread across siloed tools: scheduling, EHRs and medical reference sites. The goal was
a virtual medical assistant that coordinates these tasks from a single natural-language request. It needed to
**book appointments** against doctor availability, let attendants **add or update structured and unstructured
records**, **summarise histories** with alerts, and **fetch current disease information** from trusted sources.
It had to be built with an agent framework, RAG and memory, and to come with evaluation, monitoring and a
Streamlit UI.

## 2. Solution overview

The system is a **plan-and-execute agent** built on LangGraph:

`safety → context → planner → executor ⟲ → synthesizer → memory`

| Node | Responsibility |
|---|---|
| safety | Deterministic red-flag screen (heart attack, stroke, breathing, self-harm…). An emergency notice is always shown first, and the information search is deferred. |
| context | Resolves the acting user and their linked dependents. It loads the *active patient* from the thread's short-term memory and recalls long-term memories from FAISS. |
| planner | An LLM with structured output produces a typed `Plan`: ordered sub-goals, each with one tool, its arguments and step dependencies. |
| executor | Runs one tool per graph step. It injects the resolved patient id and patient context, and skips steps whose dependencies failed. |
| synthesizer | Writes a grounded reply from the tool results only. It keeps citations and states plainly what did not happen. |
| memory | Stores an interaction summary and the durable facts extracted from the request against the patient. |

It is backed by **13 tools** in three groups:
- A *Doctor Schedule API* over SQLite slots.
- An *EHR* built from the dataset.
- A *medical search* that combines the MedlinePlus Web Service with WHO fact sheets, in a RAG pipeline.

A multi-page Streamlit dashboard covers patient, doctor, appointment, medical-information, evaluation and memory/log
views, plus a scenario tester.

## 3. Key design decisions

1. **Plan-and-execute rather than a free ReAct loop.** Healthcare administration requests are compositional
   (*identify → history → book → research*) and have side effects: a booking must not be repeated. An explicit
   plan makes the decomposition visible, which the UI and grading both need. It is also testable (planner accuracy
   is a measured metric) and bounded, with one tool call per step and no runaway loops.
2. **A validation/repair layer between planner and executor.** LLM plans are treated as proposals.
   `validate_plan` enforces the invariants whatever produced the plan:
   - unknown or duplicate tools are dropped;
   - staff-only tools are removed for patients, with a clarification message;
   - a missing `identify_patient` step is inserted;
   - dependencies are re-wired.

   Safety and privacy therefore do not depend on the prompt being obeyed.
3. **Deterministic where correctness matters.** Several things are rules, not generations, because a hallucination
   there would be harmful: clinical alerts (abnormal labs from reference ranges, allergies, BP, polypharmacy,
   overdue follow-ups), access control, the emergency screen and the booking logic. The LLM is used for language
   tasks: planning, summarising, extracting structure from free-text notes, grounded answering and memory extraction.
4. **Booking correctness.**
   - Slots are claimed with a conditional `UPDATE … WHERE status='available'`, so exactly one concurrent claim wins.
   - Booking is idempotent: an existing upcoming appointment in the same specialty is returned, not duplicated.
   - Rescheduling books the new slot *before* releasing the old one.
   - An impossible request returns the nearest alternatives instead of silently booking something else.
5. **Graceful degradation.** Every LLM call site supplies an offline fallback (heuristic planner, template
   synthesis, extractive MMR summary, regex note extraction), and each call is logged as `live`, `fallback` or
   `offline`. The system runs, is demonstrable and is fully testable without an API key. A provider outage
   degrades quality, not availability. The medical corpus is also cached (43 MedlinePlus/WHO documents), so RAG
   still works when the network is down.
6. **Provider-agnostic LLM layer.** The app needs only `structured()` and `text()`. Groq is the default,
   matching the course environment; OpenAI, Anthropic, OpenRouter and Ollama are also supported. Structured output
   uses tool calling, with a JSON-prompting retry for models that reject tool schemas.

## 4. Data

| Input | What was done |
|---|---|
| `records.xlsx` (7 rows) | Rows are per encounter, so they were de-duplicated to 5 patients. The `Summary` column was kept as ground truth for summary evaluation. |
| 3 × *History & Physical* PDFs | Parsed into demographics, SOAP sections, ICD-10 diagnoses, medications (from plan text) and vitals. |
| `sample_patient.pdf` (12 pages) | Pages grouped by encounter footer, giving 2 progress notes, past medical history, and CCD sections with **45 lab results** (reference ranges → H/L flags), medications and vitals. |

**Synthetic additions, clearly tagged `synthetic_demo`:**
- 15 doctors across 13 specialties with working hours.
- A rolling 21-day, 30-minute slot calendar, with ~30% of slots held by other channels for realism.
- One patient required by the reference scenario: **Suresh Negi**, a 70-year-old with CKD stage 3b, type 2
  diabetes and hypertension, linked as the father of the dataset's Rahul Negi.

## 5. Prompt engineering and task chaining

Each sub-task has its own structured prompt in `prompts.py`, with a role, inputs, rules and an output contract:
- **Planner**: tool catalogue, 7 planning rules, and injected context (date, user, dependents, active patient,
  recalled memories, recent turns).
- **History summary**: fixed sections, "use only the record", keep ICD codes and dates.
- **Medical RAG**: numbered excerpts, inline `[n]` citations, a personalisation note, and "say what is missing
  rather than guess".
- **Synthesis**: address every part in order, report failures honestly, keep citations.
- **Note extraction** (typed schema) and **memory extraction** (durable facts only).

The chain is: plan → tools → summary/RAG prompts → synthesis → memory extraction. Patient context produced
by the history step flows into the RAG prompt.

## 6. Memory

| Type | Mechanism | Example |
|---|---|---|
| Short-term | LangGraph checkpointer keyed by chat thread; holds messages and the active patient | "Also book a dietitian **for him**" resolves to Suresh Negi |
| Long-term | SQLite `memories` and the FAISS `patient_memory` index (profile, encounter chunks, facts, interactions) | Recalled before planning and in history summaries |
| Knowledge | FAISS `medical_knowledge` (chunked MedlinePlus/WHO docs, grows with live searches) | RAG context |

Every recall and write is logged to `memory_events` and shown on the *Memory & logs* page.

## 7. Evaluation

**Method.** Golden sets live in `evaluation/cases.py`:
- 15 planner queries across patient, caregiver and attendant roles;
- 5 reference summaries (3 taken verbatim from `records.xlsx`);
- 8 medical QA pairs, with references paraphrased from MedlinePlus/WHO and key facts listed;
- 12 retrieval queries;
- 8 booking scenarios, plus idempotency and race checks.

When an LLM is configured, **LangChain's `QAEvalChain`** grades summaries and QA answers as CORRECT or INCORRECT.
Reference-based metrics always run, so offline and online results stay comparable: embedding similarity,
ROUGE-L recall, token F1, key-fact coverage, **context recall** and **groundedness** (the share of answer sentences
supported by a retrieved passage at cosine ≥ 0.6). Booking evaluations are rolled back afterwards.

**Results.** These are for the offline heuristic mode, run `eval-20260927-222036`, reproducible with
`python -m healthcare_agent.evaluation.runner`.

| Module | Metric | Score |
|---|---|---|
| Planner | tool F1 / exact plan match | 1.00 / 1.00 |
| Planner | specialty accuracy / patient identification | 1.00 / 1.00 |
| History summary | semantic similarity / ROUGE-L recall / token F1 | 0.865 / 0.751 / 0.361 |
| Medical QA (RAG) | answered / cited | 1.00 / 1.00 |
| Medical QA (RAG) | **context recall** (key facts present in retrieved passages) | **1.00** |
| Medical QA (RAG) | key-fact coverage in answer / groundedness / similarity | 0.667 / 0.875 / 0.813 |
| Retrieval | patient hit@1 / MRR; knowledge hit@3 | 1.00 / 1.00; 1.00 |
| Booking | outcome as expected / constraints met / idempotent / race-safe / alternatives offered | 1.00 each |
| Operations | tool success rate (live usage incl. guard-rail tests) | 0.92 |

**Live LLM results** (run `eval-20260928-075859`, Llama 3.3 70B via OpenRouter, same golden sets):

| Metric | Offline | LLM |
|---|---|---|
| Planner tool F1 / exact plan match | 1.00 / 1.00 | 0.96 / 0.80 |
| Summary QAEvalChain correct | n/a | 0.80 |
| Medical QA QAEvalChain correct | n/a | 1.00 |
| Medical QA key-fact coverage / groundedness | 0.67 / 0.88 | 1.00 / 0.99 |
| Booking (all reliability checks) | 1.00 | 1.00 |

The LLM closes the generation gap: context recall was already 1.00. The offline summary lexical scores are inflated
because the offline template echoes the recorded `Summary` field, which is also the reference. The LLM planner's
three mismatches are judgment calls: it skipped history for two specialist-only bookings and added a listing step
before a reschedule. The full discussion is in `docs/Capstone_Writeup.pdf`.

**What the numbers say.**
- **The planner score is optimistic.** The heuristic planner and the golden set were written together, so 1.00
  reflects coverage of known phrasings, not generalisation. Running the same suite with an API key measures the
  LLM planner on identical cases, and *Scenario tester* plus free chat probe unseen phrasings.
- **Retrieval is not the bottleneck; generation is.** Context recall is 1.00, meaning every key fact was in
  the retrieved passages, but the offline extractive answer surfaced only 67% of them. Replacing pure top-similarity
  selection with **MMR** (λ = 0.35) raised key-fact coverage from 0.56 to 0.69 at unchanged groundedness. An LLM
  generator, which synthesises across passages, is expected to close most of the remaining gap; QAEvalChain
  measures that in live mode.
- **Token F1 on summaries is low (0.36) while semantic similarity is high (0.87).** The generated summaries carry
  the same facts in a different, more structured form, and lexical overlap penalises that.
- **The 0.60 agent-run success rate includes deliberate refusals.** Guard-rail runs (emergency, privacy, role)
  count as failures under the strict "every step succeeded" definition, and are the intended behaviour.

## 8. LLMOps and monitoring

`observability.py` records:
- every tool call (arguments, result, success, error, latency);
- every LLM call (purpose, live/fallback/offline, latency, error);
- every memory operation;
- every agent run (plan, trace, answer, success, latency).

The *Evaluation & metrics* page aggregates these into:
- tool success rate and p50/p95 latency;
- per-module success;
- booking success;
- the LLM live/fallback mix.

The same page runs the offline suite and drills down to case level. *Memory & logs* replays any run's planning
breakdown, graph path, memory operations and final answer.

## 9. Testing

`pytest` runs 27 tests against an isolated temporary database, with no network and no LLM. They cover:
- the parsers and range flags;
- fuzzy patient lookup;
- privacy and role guards;
- atomic, idempotent booking, rescheduling and races;
- cited RAG;
- memory;
- the full reference scenario and a multi-turn follow-up;
- the emergency path;
- plan repair;
- the live-LLM code path, via a scripted fake chat model (JSON parsing and fallback);
- the evaluation harness.

## 10. Limitations and future work

- **Synthetic scheduling data.** In production, a real calendar/EHR would sit behind the same tool interfaces
  (FHIR `Appointment`, `Slot`, `Patient` resources).
- **Authentication is simulated** by an acting-user selector. It needs real identity, consent records and audit
  logging (e.g. HIPAA/DPDP).
- **WHO coverage depends on web search.** A curated guideline corpus (KDIGO, ADA, NICE) would improve "latest
  treatment" answers, and freshness metadata should be shown to the user.
- **Evaluation scale.** The golden sets are small. Next steps: expand them with LLM-generated paraphrases,
  add an LLM-judge faithfulness score, and track metrics across model versions in CI.
- **Human-in-the-loop.** Explicit confirmation should be required before bookings initiated by caregivers, and
  clinician sign-off before extracted notes are committed.

*Educational prototype on sample and synthetic data. It does not diagnose or provide medical advice.*
