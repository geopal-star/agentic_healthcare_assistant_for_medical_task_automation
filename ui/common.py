"""Shared Streamlit helpers: cached resources, the acting-user selector,
formatting and a consistent Plotly chart style."""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from healthcare_agent import db, records  # noqa: E402
from healthcare_agent.build import ensure_built  # noqa: E402
from healthcare_agent.context import Actor, current_actor  # noqa: E402
from healthcare_agent.llm import get_llm  # noqa: E402
from healthcare_agent.seed import STAFF  # noqa: E402

# Validated categorical order (dataviz reference palette) and reserved status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MUTED, GRID = "#898781", "rgba(137,135,129,0.25)"


@st.cache_resource(show_spinner="Preparing EHR database, vector indexes and embedding model...")
def bootstrap() -> bool:
    ensure_built()
    from healthcare_agent.vectorstore import get_embedder
    get_embedder()                           # warm the embedding model once per server
    return True


@st.cache_resource
def get_agent():
    from healthcare_agent.agent import HealthcareAgent
    return HealthcareAgent()


def user_options() -> dict[str, Actor]:
    opts: dict[str, Actor] = {}
    for p in records.list_patients():
        deps = records.get_dependents(p["patient_id"])
        tag = f" - caregiver of {', '.join(d['name'] for d in deps)}" if deps else ""
        opts[f"{p['name']} (patient){tag}"] = Actor(p["patient_id"], p["name"], "patient")
    for s in STAFF:
        opts[f"{s['name']} ({s['role']})"] = Actor(s["user_id"], s["name"], s["role"])
    return opts


def sidebar() -> Actor:
    """Acting-user selector + system status. Returns the actor for this session."""
    bootstrap()
    opts = user_options()
    labels = list(opts)
    default = next((i for i, lbl in enumerate(labels) if lbl.startswith("Rahul Negi")), 0)
    with st.sidebar:
        st.markdown("### Signed in as")
        choice = st.selectbox("Acting user", labels, index=st.session_state.get("actor_idx", default),
                              label_visibility="collapsed")
        st.session_state["actor_idx"] = labels.index(choice)
        actor = opts[choice]
        st.caption(f"Role: **{actor.role}** - " + ("can edit records" if actor.can_edit_records else "read / book only"))
        st.divider()
        llm = get_llm()
        if llm.live:
            st.success(f"LLM: {llm.label}", icon="✅")
        else:
            st.warning("LLM: offline heuristic mode. Add an API key to `.env` for full quality.", icon="⚠️")
        if st.button("New conversation", width="stretch"):
            st.session_state["thread_id"] = uuid.uuid4().hex[:8]
            st.session_state["chat"] = []
            st.rerun()
    if st.session_state.get("actor_user") != actor.user_id:      # switching user starts a fresh thread
        st.session_state["actor_user"] = actor.user_id
        st.session_state["thread_id"] = uuid.uuid4().hex[:8]
        st.session_state["chat"] = []
    current_actor.set(actor)
    return actor


def when(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).strftime("%a %d %b, %I:%M %p")
    except ValueError:
        return ts


def style_fig(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=78 if legend else 40, b=8), showlegend=legend,
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hoverlabel=dict(font_size=13), bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def kpi_row(items: list[tuple[str, str | int | float, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)


def status_badge(ok: bool | None, skipped: bool = False) -> str:
    if skipped:
        return "⏭️ skipped"
    return "✅ success" if ok else "❌ failed"


__all__ = ["db", "records", "Actor", "sidebar", "get_agent", "when", "style_fig", "kpi_row", "SERIES", "STATUS",
           "status_badge", "bootstrap"]
