# Evaluation report `eval-20260928-075859`

LLM: **openrouter:meta-llama/llama-3.3-70b-instruct**

## planner

| metric | mean |
|---|---|
| tool_precision | 0.978 |
| tool_recall | 0.956 |
| tool_f1 | 0.960 |
| exact_match | 0.800 |
| specialty_accuracy | 1.000 |
| patient_id_accuracy | 1.000 |

## summary

| metric | mean |
|---|---|
| semantic_similarity | 0.817 |
| rouge_l_recall | 0.528 |
| token_f1 | 0.286 |
| qa_eval_correct | 0.800 |

## medical_qa

| metric | mean |
|---|---|
| answered | 1.000 |
| semantic_similarity | 0.845 |
| keyword_coverage | 1.000 |
| context_recall | 1.000 |
| groundedness | 0.986 |
| cited_sources | 1.000 |
| qa_eval_correct | 1.000 |

## retrieval

| metric | mean |
|---|---|
| patient_hit@1 | 1.000 |
| patient_hit@3 | 1.000 |
| patient_mrr | 1.000 |
| knowledge_hit@3 | 1.000 |

## booking

| metric | mean |
|---|---|
| outcome_as_expected | 1.000 |
| constraint_satisfaction | 1.000 |
| idempotent | 1.000 |
| offers_alternatives | 1.000 |
| race_safe | 1.000 |

## tools

| metric | mean |
|---|---|
| success_rate | 0.985 |
| run_success_rate | 0.900 |
| live_call_share | 0.486 |
