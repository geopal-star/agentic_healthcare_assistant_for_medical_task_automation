from healthcare_agent.evaluation.metrics import groundedness, keyword_coverage, rouge_l_recall, set_prf, token_f1
from healthcare_agent.evaluation.runner import run_evaluation


def test_metric_functions():
    assert token_f1("diabetes and high blood pressure", "high blood pressure and diabetes") == 1.0
    assert rouge_l_recall("dialysis or kidney transplant", "kidney transplant") == 1.0
    assert keyword_coverage("Rest and fluids help", ["rest", "fluid", "antibiotic|no cure"]) == 2 / 3
    assert set_prf({"a", "b"}, {"a"}) == (0.5, 1.0, 2 / 3)
    ctx = ["Diabetes and high blood pressure are the most common causes of chronic kidney disease."]
    assert groundedness("The most common causes of chronic kidney disease are diabetes and high blood pressure.", ctx) == 1.0


def test_evaluation_suite_runs_and_cleans_up():
    from healthcare_agent import db
    before = db.query_one("SELECT COUNT(*) AS n FROM appointments WHERE status='booked'")["n"]
    rep = run_evaluation(["planner", "retrieval", "booking"], verbose=False)
    s = rep["summary"]
    assert s["planner"]["tool_f1"] >= 0.9
    assert s["retrieval"]["patient_hit@3"] >= 0.8
    assert s["booking"]["outcome_as_expected"] == 1.0 and s["booking"]["race_safe"] == 1.0
    after = db.query_one("SELECT COUNT(*) AS n FROM appointments WHERE status='booked'")["n"]
    assert after == before                                  # evaluation bookings were rolled back
