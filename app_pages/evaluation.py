"""Evaluation metrics for model responses and tool success."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthcare_agent.evaluation.runner import ALL_MODULES, run_evaluation
from healthcare_agent.tools import TOOLS
from ui.common import SERIES, STATUS, db, kpi_row, sidebar, style_fig

actor = sidebar()
st.title("📊 Evaluation & metrics")

tab_live, tab_eval = st.tabs(["Live operations", "Offline evaluation suite"])

# ---------------------------------------------------------------------------
with tab_live:
    logs = pd.DataFrame(db.query("SELECT * FROM tool_logs WHERE run_id IS NULL OR run_id NOT LIKE 'eval-%'"))
    runs = pd.DataFrame(db.query("SELECT * FROM agent_runs ORDER BY ts"))
    llm_calls = pd.DataFrame(db.query("SELECT * FROM llm_calls"))
    bookings = logs[logs["tool"] == "book_appointment"] if not logs.empty else logs
    kpi_row([
        ("Agent runs", len(runs), None),
        ("Run success rate", f"{runs['success'].mean():.0%}" if not runs.empty else "-", "all plan steps succeeded"),
        ("Tool calls", len(logs), None),
        ("Tool success rate", f"{logs['success'].mean():.0%}" if not logs.empty else "-", None),
        ("Booking success", f"{bookings['success'].mean():.0%}" if not bookings.empty else "-", "book_appointment calls"),
        ("Median run latency", f"{runs['latency_ms'].median() / 1000:.1f}s" if not runs.empty else "-", None),
    ])
    if logs.empty:
        st.info("No tool usage yet - chat with the assistant or run the scenario tester.")
    else:
        agg = logs.groupby("tool").agg(calls=("id", "count"), success=("success", "mean"),
                                       p50=("latency_ms", "median"), p95=("latency_ms", lambda s: s.quantile(0.95))).reset_index()
        agg["module"] = agg["tool"].map(lambda t: TOOLS[t].module if t in TOOLS else "?")
        agg = agg.sort_values("success")
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(y=agg["tool"], x=agg["success"] * 100, orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4),
                               text=[f"{v:.0%} (n={n})" for v, n in zip(agg["success"], agg["calls"])], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.0f}% success<extra></extra>"))
        fig.update_layout(title="Tool success rate (%)", xaxis_range=[0, 135])
        c1.plotly_chart(style_fig(fig, 60 + 34 * len(agg)), width="stretch")
        fig = go.Figure()
        for i, (col, name) in enumerate([("p50", "p50"), ("p95", "p95")]):
            fig.add_bar(y=agg["tool"], x=agg[col], orientation="h", name=f"{name} latency", marker=dict(color=SERIES[i], cornerradius=4),
                        hovertemplate="%{y}: %{x:.0f} ms<extra>" + name + "</extra>")
        fig.update_layout(title="Tool latency (ms)", barmode="group")
        c2.plotly_chart(style_fig(fig, 60 + 34 * len(agg), legend=True), width="stretch")

        mod = logs.assign(module=logs["tool"].map(lambda t: TOOLS[t].module if t in TOOLS else "?")) \
                  .groupby("module")["success"].agg(["mean", "count"]).reset_index()
        st.markdown("**Success rate per module**")
        st.dataframe(mod.rename(columns={"mean": "success_rate", "count": "calls"}), hide_index=True,
                     column_config={"success_rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        failures = logs[logs["success"] == 0][["ts", "tool", "error", "args"]].sort_values("ts", ascending=False)
        with st.expander(f"Failures ({len(failures)})"):
            st.dataframe(failures, hide_index=True, width="stretch")
    if not llm_calls.empty:
        mix = llm_calls.groupby(["purpose", "mode"]).size().unstack(fill_value=0)
        st.markdown("**LLM calls by purpose and mode** (live = model answered, fallback = model failed → heuristic, offline = no model configured)")
        st.dataframe(mix, width="stretch")

# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown("Runs the golden-set evaluation: planner accuracy, summary quality (vs `records.xlsx` summaries), RAG answer "
                "quality (similarity, key-fact coverage, groundedness, **QAEvalChain** when an LLM is configured), FAISS "
                "retrieval hit-rates and booking reliability (success, constraint satisfaction, idempotency, race safety).")
    mods = st.multiselect("Modules", ALL_MODULES, default=ALL_MODULES)
    if st.button("▶ Run evaluation", type="primary"):
        with st.spinner("Evaluating... (medical QA does live retrieval, allow a minute)"):
            rep = run_evaluation(mods, verbose=False)
        st.success(f"Finished {rep['eval_run_id']} with {rep['llm']}")

    ids = [r["eval_run_id"] for r in db.query("SELECT DISTINCT eval_run_id FROM eval_results ORDER BY eval_run_id DESC")]
    if not ids:
        st.info("No evaluation has been run yet.")
        st.stop()
    run_id = st.selectbox("Evaluation run", ids)
    res = pd.DataFrame(db.query("SELECT * FROM eval_results WHERE eval_run_id=?", (run_id,)))
    summary = res.groupby(["module", "metric"])["score"].mean().reset_index()
    summary = summary[summary["module"] != "tools"]
    fig = go.Figure(go.Bar(y=summary["module"] + " · " + summary["metric"], x=summary["score"], orientation="h",
                           marker=dict(color=SERIES[0], cornerradius=4), text=summary["score"].map(lambda v: f"{v:.2f}"),
                           textposition="outside", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.8, line=dict(color=STATUS["good"], dash="dot", width=1), annotation_text="0.8 target",
                  annotation_position="top")
    fig.update_layout(title="Mean score per metric (0-1)", xaxis_range=[0, 1.15], yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_fig(fig, 80 + 26 * len(summary)), width="stretch")

    for module in res["module"].unique():
        with st.expander(f"Case-level results - {module}"):
            m = res[res["module"] == module]
            wide = m.pivot_table(index="case_id", columns="metric", values="score").reset_index()
            details = m.dropna(subset=["details"]).groupby("case_id")["details"].first().map(
                lambda d: json.dumps(json.loads(d))[:220])
            st.dataframe(wide.merge(details.rename("details"), on="case_id", how="left"), hide_index=True, width="stretch")
