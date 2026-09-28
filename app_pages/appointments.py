"""Real-time appointment tracking (auto-refreshing fragment)."""
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthcare_agent.tools.scheduling import cancel_appointment
from ui.common import SERIES, db, kpi_row, sidebar, style_fig, when

actor = sidebar()
st.title("📅 Appointment tracker")

live = st.toggle("Live refresh (every 5 s)", value=True)
specs = [r["specialty"] for r in db.query("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")]
f1, f2 = st.columns([2, 1])
spec_filter = f1.multiselect("Specialty", specs)
status_filter = f2.multiselect("Status", ["booked", "cancelled", "completed"], default=["booked"])


@st.fragment(run_every=5 if live else None)
def tracker():
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    rows = pd.DataFrame(db.query("""SELECT a.appointment_id, a.start_ts, a.status, a.reason, a.booked_by, a.created_at,
                                           a.patient_id, p.name AS patient, d.name AS doctor, d.specialty, d.location
                                    FROM appointments a JOIN patients p ON p.patient_id=a.patient_id
                                    JOIN doctors d ON d.doctor_id=a.doctor_id ORDER BY a.start_ts"""))
    if rows.empty:
        st.info("No appointments yet. Book one from the chat page.")
        return
    if actor.role == "patient":            # patients/caregivers only see their own family's appointments
        allowed = {actor.user_id} | {d["patient_id"] for d in db.query("SELECT patient_id FROM relationships WHERE user_id=?", (actor.user_id,))}
        rows = rows[rows["patient_id"].isin(allowed)]
    upcoming = rows[(rows["status"] == "booked") & (rows["start_ts"] >= now)]
    today = date.today().isoformat()
    kpi_row([
        ("Upcoming", len(upcoming), None),
        ("Today", int((upcoming["start_ts"].str[:10] == today).sum()), None),
        ("Booked in last 24 h", int((rows["created_at"] >= (datetime.now() - timedelta(days=1)).isoformat(sep=" ")).sum()), None),
        ("Cancelled", int((rows["status"] == "cancelled").sum()), None),
    ])
    view = rows
    if spec_filter:
        view = view[view["specialty"].isin(spec_filter)]
    if status_filter:
        view = view[view["status"].isin(status_filter)]

    if not upcoming.empty:
        by_day = upcoming.assign(day=upcoming["start_ts"].str[:10]).groupby("day").size()
        fig = go.Figure(go.Bar(x=by_day.index, y=by_day.values, marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{x}: %{y} appointments<extra></extra>"))
        fig.update_layout(title="Upcoming appointments per day")
        st.plotly_chart(style_fig(fig, 240), width="stretch")

    view = view.assign(when=view["start_ts"].map(when))
    st.dataframe(view[["appointment_id", "when", "status", "patient", "doctor", "specialty", "location", "reason", "booked_by"]],
                 hide_index=True, width="stretch")
    st.caption(f"Last refreshed {datetime.now():%H:%M:%S}")


tracker()

with st.expander("Cancel an appointment"):
    appt_id = st.text_input("Appointment ID", placeholder="APT-XXXXXXXX")
    if st.button("Cancel appointment") and appt_id:
        res = cancel_appointment(appointment_id=appt_id.strip())
        if res.get("ok"):
            st.success(f"Cancelled {res['cancelled']['appointment_id']}")
        else:
            st.error(res.get("error"))
