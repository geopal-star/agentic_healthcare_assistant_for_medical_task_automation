"""Patient view: record, alerts, labs, appointments; staff can add/update records."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthcare_agent import memory
from healthcare_agent.ingest import parse_pdf
from healthcare_agent.tools.patient import add_clinical_note, register_patient, summarize_history, update_patient_record
from ui.common import SERIES, records, sidebar, style_fig, when

actor = sidebar()
st.title("🧑 Patient view")

if actor.can_edit_records:
    visible = records.list_patients()
else:
    visible = records.get_dependents(actor.user_id) + [p for p in [records.get_patient(actor.user_id)] if p]
if not visible:
    st.info("No patient records are accessible for this user.")
    st.stop()

labels = {f"{p['name']} ({p['patient_id']})": p["patient_id"] for p in visible}
pid = labels[st.selectbox("Patient", list(labels))]
rec = records.get_full_record(pid)
p = rec["patient"]
alerts = records.compute_alerts(rec)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age / sex", f"{p.get('age') or '?'} / {p.get('gender') or '?'}")
c2.metric("Active diagnoses", sum(d["status"] == "active" for d in rec["diagnoses"]))
c3.metric("Active medications", sum(m["status"] == "active" for m in rec["medications"]))
c4.metric("Alerts", len(alerts))
st.caption(f"📞 {p.get('phone') or '-'} · ✉️ {p.get('email') or '-'} · 🏠 {p.get('address') or '-'} · source: {p.get('source')}"
           + (f" · caregivers: {', '.join(c['name'] + ' (patient is their ' + c['relation'] + ')' for c in rec['caregivers'])}" if rec["caregivers"] else ""))

ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
if alerts:
    with st.container(border=True):
        st.markdown("**Clinical alerts**")
        for a in alerts:
            st.markdown(f"{ICON[a['severity']]} **{a['severity'].upper()}** · {a['type']} - {a['text']}")

tabs = st.tabs(["AI summary", "Encounters", "Diagnoses & meds", "Labs & vitals", "Appointments", "Memory"]
               + (["✏️ Manage record"] if actor.can_edit_records else []))

with tabs[0]:
    focus = st.text_input("Optional focus (e.g. kidney function)", key=f"focus-{pid}")
    if st.button("Generate summary", type="primary"):
        with st.spinner("Summarising history..."):
            st.session_state[f"summary-{pid}"] = summarize_history(pid, focus or None)
    res = st.session_state.get(f"summary-{pid}")
    if res:
        st.markdown(res["summary"])
        st.caption(f"Generated in {res['mode']} mode · memories used: {len(res['memories_used'])}")
    elif p.get("summary"):
        st.markdown(f"**Recorded summary:** {p['summary']}")

with tabs[1]:
    if not rec["encounters"]:
        st.info("No encounters recorded.")
    for e in rec["encounters"]:
        with st.expander(f"{e['visit_date']} - {e['chief_complaint'] or 'Visit'} ({e['location'] or 'n/a'})"):
            for label, key in [("Subjective", "subjective"), ("Objective", "objective"), ("Assessment", "assessment"), ("Plan", "plan")]:
                if e[key]:
                    st.markdown(f"**{label}:** {e[key]}")
            st.caption(f"Provider: {e['provider'] or 'n/a'} · encounter {e['encounter_id']} · source {e['source']}")

with tabs[2]:
    a, b = st.columns(2)
    a.markdown("**Diagnoses**")
    a.dataframe(pd.DataFrame(rec["diagnoses"])[["description", "icd10", "status", "noted_on"]] if rec["diagnoses"] else pd.DataFrame(),
                hide_index=True, width="stretch")
    b.markdown("**Medications**")
    b.dataframe(pd.DataFrame(rec["medications"])[["name", "dosage", "status", "noted_on"]] if rec["medications"] else pd.DataFrame(),
                hide_index=True, width="stretch")

with tabs[3]:
    if rec["labs"]:
        labs = pd.DataFrame(rec["labs"])
        flagged = labs[labs["flag"].notna()]
        counts = labs["test"].value_counts()
        tests = sorted(labs["test"].unique(), key=lambda t: (t not in set(flagged["test"]), -counts[t], t))
        test = st.selectbox("Lab trend", tests, help="Flagged tests with the most readings are listed first")
        series = labs[labs["test"] == test].sort_values("taken_on")
        fig = go.Figure(go.Scatter(x=series["taken_on"], y=series["value"], mode="lines+markers",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=9),
                                   hovertemplate="%{x}<br>%{y} " + str(series["unit"].iloc[0] or "") + "<extra></extra>"))
        fig.update_layout(title=f"{test} ({series['unit'].iloc[0] or ''}) - reference {series['ref_range'].iloc[0] or 'n/a'}")
        st.plotly_chart(style_fig(fig, 280), width="stretch")
        st.dataframe(labs[["taken_on", "test", "value", "unit", "ref_range", "flag"]], hide_index=True, width="stretch")
    else:
        st.info("No lab results.")
    if rec["vitals"]:
        st.markdown("**Vitals**")
        st.dataframe(pd.DataFrame(rec["vitals"])[["taken_on", "name", "value", "unit"]], hide_index=True, width="stretch")

with tabs[4]:
    if rec["appointments"]:
        df = pd.DataFrame(rec["appointments"])
        df["when"] = df["start_ts"].map(when)
        st.dataframe(df[["appointment_id", "when", "doctor_name", "specialty", "reason", "status"]], hide_index=True, width="stretch")
    else:
        st.info("No appointments yet - ask the assistant to book one.")

with tabs[5]:
    mems = memory.list_memories(pid)
    if mems:
        st.dataframe(pd.DataFrame(mems)[["created_at", "kind", "content"]], hide_index=True, width="stretch")
    else:
        st.info("No long-term memories yet. They are written after conversations about this patient.")

if actor.can_edit_records:
    with tabs[6]:
        st.markdown("#### Update structured fields")
        with st.form("update"):
            cols = st.columns(3)
            vals = {f: cols[i % 3].text_input(f.capitalize(), value=str(p.get(f) or ""))
                    for i, f in enumerate(["phone", "email", "address", "allergies", "dob", "gender"])}
            if st.form_submit_button("Save changes"):
                changed = [{"field": f, "value": v} for f, v in vals.items() if v != str(p.get(f) or "")]
                if changed:
                    res = update_patient_record(patient_id=pid, updates=changed)
                    if res.get("ok"):
                        st.toast(f"Updated: {', '.join(res['updated'])}")
                        st.rerun()
                    st.error(res.get("error"))
                else:
                    st.info("Nothing changed.")

        st.markdown("#### Add unstructured clinical note")
        with st.form("note"):
            note = st.text_area("Note", placeholder="e.g. 2026-09-27 BP 150/95, started Amlodipine 5mg OD. Diagnosis: Hypertension (I10). Allergic to penicillin.")
            if st.form_submit_button("Add note") and note.strip():
                res = add_clinical_note(patient_id=pid, note_text=note)
                if res.get("ok"):
                    st.success(f"Added encounter {res['encounter_id']} ({res['extraction_mode']} extraction)")
                    st.json(res["extracted"])
                else:
                    st.error(res.get("error"))

        st.markdown("#### Upload a clinical PDF")
        up = st.file_uploader("History & Physical note / EHR export", type=["pdf"])
        if up and st.button("Parse & attach to this patient"):
            doc = parse_pdf(up.getvalue())
            if doc.layout == "unstructured":
                res = add_clinical_note(patient_id=pid, note_text=doc.raw_text[:6000])
                if res.get("ok"):
                    st.success(f"Unstructured PDF stored as note {res['encounter_id']}")
                else:
                    st.error(res.get("error"))
            else:
                res = records.ingest_document(doc, patient_id=pid, source=up.name)
                memory.index_patient(pid)
                st.success(f"Parsed as {doc.layout}: {len(res['encounters'])} encounters, {res['labs']} labs")

        st.markdown("#### Register a new patient")
        with st.form("register"):
            cols = st.columns(3)
            name = cols[0].text_input("Full name*")
            age = cols[1].number_input("Age", 0, 120, 40)
            gender = cols[2].selectbox("Gender", ["Female", "Male", "Other"])
            phone = cols[0].text_input("Phone")
            address = cols[1].text_input("Address")
            allergies = cols[2].text_input("Allergies")
            if st.form_submit_button("Register") and name:
                res = register_patient(name=name, age=int(age), gender=gender, phone=phone or None,
                                       address=address or None, allergies=allergies or None)
                if res.get("ok"):
                    st.success(f"Registered {res['patient_id']}")
                else:
                    st.error(res.get("error"))
