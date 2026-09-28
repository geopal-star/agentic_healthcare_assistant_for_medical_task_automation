"""Doctor view: weekly calendar utilisation, upcoming patients with briefings."""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import SERIES, db, records, sidebar, style_fig, when

actor = sidebar()
st.title("🩺 Doctor view")

doctors = db.query("SELECT * FROM doctors ORDER BY specialty, name")
labels = {f"{d['name']} - {d['specialty']}": d for d in doctors}
default = next((i for i, d in enumerate(doctors) if d["doctor_id"] == actor.user_id), 0)
doc = labels[st.selectbox("Doctor", list(labels), index=default)]
st.caption(f"📍 {doc['location']} · {doc['experience_years']} yrs · languages: {doc['languages']} · "
           f"hours {doc['work_start']}-{doc['work_end']}")

days = st.segmented_control("Horizon", [7, 14, 21], default=7, format_func=lambda d: f"{d} days") or 7
start, end = date.today(), date.today() + timedelta(days=days)
slots = pd.DataFrame(db.query("""SELECT s.*, a.appointment_id, a.status AS appt_status FROM slots s
                                 LEFT JOIN appointments a ON a.slot_id=s.slot_id AND a.status='booked'
                                 WHERE s.doctor_id=? AND date(s.start_ts) BETWEEN ? AND ?""",
                              (doc["doctor_id"], start.isoformat(), end.isoformat())))
if not slots.empty:
    slots["day"] = slots["start_ts"].str[:10]
    slots["kind"] = slots.apply(lambda r: "Booked via assistant" if pd.notna(r["appointment_id"]) else
                                ("Held (other channels)" if r["status"] == "blocked" else "Open"), axis=1)
    pivot = slots.pivot_table(index="day", columns="kind", values="slot_id", aggfunc="count", fill_value=0)
    fig = go.Figure()
    colors = {"Booked via assistant": SERIES[0], "Held (other channels)": SERIES[1], "Open": "#c3c2b7"}   # remainder in neutral
    for kind, color in colors.items():
        if kind in pivot:
            fig.add_bar(x=pivot.index, y=pivot[kind], name=kind, marker=dict(color=color),
                        hovertemplate="%{x}<br>" + kind + ": %{y} slots<extra></extra>")
    fig.update_layout(barmode="stack", title="Slot utilisation per working day")
    st.plotly_chart(style_fig(fig, 300, legend=True), width="stretch")
    c1, c2, c3 = st.columns(3)
    total = len(slots)
    c1.metric("Slots in horizon", total)
    c2.metric("Booked via assistant", int((slots["kind"] == "Booked via assistant").sum()))
    c3.metric("Utilisation", f"{(slots['kind'] != 'Open').mean():.0%}")
else:
    st.info("The doctor has no working days in this horizon.")

st.subheader("Upcoming patients")
appts = db.query("""SELECT a.*, p.name AS patient_name, p.age, p.gender FROM appointments a
                    JOIN patients p ON p.patient_id=a.patient_id
                    WHERE a.doctor_id=? AND a.status='booked' ORDER BY a.start_ts""", (doc["doctor_id"],))
if not appts:
    st.info("No upcoming appointments booked through the assistant.")
for a in appts:
    with st.expander(f"{when(a['start_ts'])} - {a['patient_name']} ({a['age']}y {a['gender'] or ''}) - {a['reason'] or ''}"):
        rec = records.get_full_record(a["patient_id"])
        if actor.role in ("doctor", "attendant"):
            dx = [d["description"] for d in rec["diagnoses"] if d["status"] == "active"]
            meds = [f"{m['name']} {m['dosage'] or ''}" for m in rec["medications"] if m["status"] == "active"]
            st.markdown(f"**Active problems:** {', '.join(dx) or 'none'}  \n**Medications:** {', '.join(meds) or 'none'}")
            for al in records.compute_alerts(rec)[:6]:
                st.markdown(f"- {al['text']}")
            st.caption(f"Appointment {a['appointment_id']} · booked by {a['booked_by']} at {a['created_at']}")
        else:
            st.caption("Clinical briefing is visible to doctors and attendants only.")
