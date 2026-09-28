"""End-to-end agent behaviour (offline planner) plus the live-LLM code path
exercised with a scripted fake chat model."""
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from healthcare_agent import records
from healthcare_agent.agent.planner import Plan, heuristic_plan, validate_plan
from healthcare_agent.agent.safety import screen
from healthcare_agent.config import Settings
from healthcare_agent.context import Actor
from healthcare_agent.llm import LLM

SAMPLE = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
          "Also, can you summarize latest treatment methods?")


def rahul():
    return Actor(records.find_patients("Rahul Negi")[0]["patient_id"], "Rahul Negi", "patient")


def test_capstone_reference_scenario(agent):
    res = agent.run(SAMPLE, rahul(), thread_id="t-sample")
    assert [s["tool"] for s in res["steps"]] == ["identify_patient", "get_patient_history", "book_appointment",
                                                 "medical_info_search"]
    assert res["success"], res["steps"]
    booked = res["steps"][2]["result"]["appointment"]
    assert booked["specialty"] == "Nephrology"
    assert res["active_patient"]["name"] == "Suresh Negi"
    assert "Nephrology" in res["answer"] and "Sources" in res["answer"]
    assert [t["node"] for t in res["trace"]][:3] == ["safety", "context", "planner"]


def test_follow_up_uses_short_term_memory(agent):
    res = agent.run("Also book a dietitian for him next week", rahul(), thread_id="t-sample")
    assert res["success"]
    assert res["steps"][-1]["result"]["appointment"]["specialty"] == "Nutrition & Dietetics"
    assert any("active patient" in r for r in res["repairs"])


def test_long_term_memory_written(agent):
    from healthcare_agent import memory
    suresh = records.find_patients("Suresh Negi")[0]["patient_id"]
    kinds = {m["kind"] for m in memory.list_memories(suresh)}
    assert {"interaction", "fact"} <= kinds


def test_emergency_is_flagged_first(agent):
    res = agent.run("My father has chest pain and is sweating and short of breath", rahul(), thread_id="t-emg")
    assert res["safety"]["level"] == "emergency"
    assert res["answer"].lstrip("> ").startswith("**This may be an emergency")


def test_screen_levels():
    assert screen("I want to book a checkup")["level"] == "none"
    assert screen("severe headache since morning")["level"] == "urgent"


def test_role_guard_repairs_plan():
    anjali = Actor(records.find_patients("Anjali Mehra")[0]["patient_id"], "Anjali Mehra", "patient")
    plan, repairs = validate_plan(heuristic_plan("Update my address to 1 MG Road", anjali, {}), anjali, {})
    assert all(s.tool != "update_patient_record" for s in plan.steps)
    assert plan.needs_clarification and any("requires attendant" in r for r in repairs)


def test_validate_inserts_identify_step():
    plan = Plan(intent_summary="book", patient_relation="father",
                steps=[{"step": 1, "goal": "book", "tool": "book_appointment", "args": {"specialty": "Nephrology"}}])
    plan, repairs = validate_plan(plan, rahul(), {})
    assert [s.tool for s in plan.steps] == ["identify_patient", "book_appointment"]
    assert plan.steps[1].depends_on == [1] and "inserted identify_patient step" in repairs


def test_live_llm_path_with_fake_model():
    """The structured() call falls back to JSON prompting when the model lacks
    tool-calling, then validates the JSON into the Plan schema."""
    scripted = json.dumps({"intent_summary": "book nephrology", "patient_relation": "father",
                           "steps": [{"step": 1, "goal": "identify", "tool": "identify_patient", "args": {"relation": "father"}},
                                     {"step": 2, "goal": "book", "tool": "book_appointment",
                                      "args": {"specialty": "Nephrology"}, "depends_on": [1]}]})
    llm = LLM(Settings(llm_provider="fake-test", llm_model="fake"))
    llm._model = FakeListChatModel(responses=[f"```json\n{scripted}\n```"])
    plan, mode = llm.structured("planner", "sys", "user", Plan, fallback=lambda: None)
    assert mode == "live" and [s.tool for s in plan.steps] == ["identify_patient", "book_appointment"]

    llm._model = FakeListChatModel(responses=["not json at all"])
    fallback_plan = Plan(intent_summary="fb", steps=[])
    plan, mode = llm.structured("planner", "sys", "user", Plan, fallback=lambda: fallback_plan)
    assert mode == "fallback" and plan is fallback_plan
