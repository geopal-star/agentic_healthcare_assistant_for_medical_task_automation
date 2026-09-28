"""Agentic Healthcare Assistant - Streamlit dashboard.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🩺", layout="wide")

pages = {
    "Assistant": [
        st.Page("app_pages/assistant.py", title="Chat with the assistant", icon="💬", default=True),
        st.Page("app_pages/scenarios.py", title="Scenario tester", icon="🧪"),
    ],
    "Views": [
        st.Page("app_pages/patient_view.py", title="Patient view", icon="🧑"),
        st.Page("app_pages/doctor_view.py", title="Doctor view", icon="🩺"),
        st.Page("app_pages/appointments.py", title="Appointment tracker", icon="📅"),
        st.Page("app_pages/medical_info.py", title="Medical information", icon="📚"),
    ],
    "LLMOps": [
        st.Page("app_pages/evaluation.py", title="Evaluation & metrics", icon="📊"),
        st.Page("app_pages/memory_logs.py", title="Memory & logs", icon="🧠"),
    ],
}
st.navigation(pages).run()
