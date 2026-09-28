"""Chat page: the agent plans, executes tools and answers, with a live trace."""
import json

import streamlit as st

from ui.common import get_agent, sidebar, status_badge

actor = sidebar()
st.title("💬 Agentic Healthcare Assistant")
st.caption(f"Signed in as **{actor.name}** ({actor.role}). Ask for bookings, medical history, record updates or disease information - "
           "multi-part requests are decomposed into sub-goals automatically.")

NODE_LABELS = {"safety": "🛡️ Safety screen", "context": "🧠 Loading context & memory", "planner": "🗺️ Planning sub-goals",
               "executor": "🔧 Executing step", "synthesizer": "✍️ Composing answer", "memory": "💾 Updating long-term memory"}

EXAMPLES = {
    "patient": ["My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
                "Also, can you summarize latest treatment methods?",
                "Also book a dietitian for him next week in the morning",
                "Show his upcoming appointments",
                "What are the symptoms of type 2 diabetes?"],
    "attendant": ["Summarize the medical history of Rebeca Nagle",
                  "Update the phone number of David Thompson to +91-99999-00000",
                  "Add a note for Ramesh Kulkarni: BP 150/95 today, started Amlodipine 5mg OD. Diagnosis: Uncontrolled hypertension (I10)",
                  "Book a cardiologist for Ramesh Kulkarni tomorrow afternoon"],
    "doctor": ["Summarize the medical history of Suresh Negi with focus on kidney function",
               "What is the latest treatment for diabetic kidney disease?"],
}


def render_details(res: dict) -> None:
    plan = res.get("plan") or {}
    with st.expander(f"🗺️ Plan ({res.get('plan_mode')}) - {len(plan.get('steps', []))} sub-goals", expanded=False):
        st.markdown(f"**Intent:** {plan.get('intent_summary', '')}")
        for s in plan.get("steps", []):
            args = {k: v for k, v in s["args"].items() if v}
            dep = f" - depends on {s['depends_on']}" if s["depends_on"] else ""
            st.markdown(f"{s['step']}. **{s['goal']}** → `{s['tool']}` {('`' + json.dumps(args) + '`') if args else ''}{dep}")
        if res.get("repairs"):
            st.caption("Plan repairs: " + "; ".join(res["repairs"]))
        if plan.get("needs_clarification"):
            st.info(plan["needs_clarification"])
    with st.expander(f"🔧 Execution trace - {res['steps_ok']}/{res['steps_total']} steps succeeded, {res['latency_ms']} ms"):
        for s in res.get("steps", []):
            st.markdown(f"**{s['step']}. {s['tool']}** - {status_badge((s.get('result') or {}).get('ok'), s['status'] == 'skipped')}"
                        f" · {s.get('ms', 0)} ms" + (f" · _{s['reason']}_" if s.get("reason") else ""))
            if s.get("result"):
                st.json({k: v for k, v in s["result"].items() if k not in ("contexts",)}, expanded=False)
        st.caption("Graph path: " + " → ".join(f"{t['node']} ({t['ms']}ms)" for t in res.get("trace", [])))


for msg in st.session_state.get("chat", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("result"):
            render_details(msg["result"])

examples = EXAMPLES.get(actor.role, EXAMPLES["patient"])
cols = st.columns(len(examples))
clicked = None
for col, ex in zip(cols, examples):
    if col.button(ex[:70] + ("…" if len(ex) > 70 else ""), help=ex, width="stretch"):
        clicked = ex

prompt = st.chat_input("e.g. Book a nephrologist for my father and summarise the latest CKD treatments") or clicked
if prompt:
    st.session_state.setdefault("chat", []).append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        result = None
        with st.status("Thinking...", expanded=True) as status:
            for ev in get_agent().stream(prompt, actor, st.session_state["thread_id"]):
                if ev["type"] == "node":
                    node, upd = ev["node"], ev["update"]
                    detail = ""
                    if node == "planner":
                        detail = ", ".join(f"{s['step']}.{s['tool']}" for s in upd.get("plan", {}).get("steps", []))
                    elif node == "executor" and upd.get("step_results"):
                        last = upd["step_results"][-1]
                        detail = f"{last['tool']} - {status_badge((last.get('result') or {}).get('ok'), last['status'] == 'skipped')}"
                    elif node == "safety" and upd.get("safety", {}).get("level") not in (None, "none"):
                        detail = f"⚠️ {upd['safety']['label']}"
                    st.write(f"{NODE_LABELS.get(node, node)} {('- ' + detail) if detail else ''}")
                else:
                    result = ev["result"]
            status.update(label=f"Done in {result['latency_ms'] / 1000:.1f}s - {result['steps_ok']}/{result['steps_total']} steps succeeded",
                          state="complete" if result["success"] or not result["steps_total"] else "error", expanded=False)
        st.markdown(result["answer"])
        render_details(result)
    st.session_state["chat"].append({"role": "assistant", "content": result["answer"], "result": result})
