"""Medical information: trusted-source RAG search and the latest retrieved answers."""
import json

import pandas as pd
import streamlit as st

from healthcare_agent.tools.medical_search import medical_info_search
from ui.common import db, sidebar

actor = sidebar()
st.title("📚 Medical information")
st.caption("Answers are generated only from MedlinePlus (NIH/NLM) and WHO content retrieved into the FAISS knowledge base, with citations.")

docs = db.query("SELECT source, COUNT(*) AS n, MAX(fetched_at) AS latest FROM medical_docs GROUP BY source")
cols = st.columns(len(docs) + 1 if docs else 1)
cols[0].metric("Documents in knowledge base", sum(d["n"] for d in docs))
for c, d in zip(cols[1:], docs):
    c.metric(d["source"], d["n"], help=f"latest fetch {d['latest']}")

with st.form("search"):
    q = st.text_input("Ask about a disease, treatment, symptom or prevention",
                      placeholder="e.g. latest treatment for chronic kidney disease")
    ctx = st.text_input("Optional patient context", placeholder="e.g. 70-year-old with CKD stage 3b and diabetes")
    go_ = st.form_submit_button("Search trusted sources", type="primary")
if go_ and q:
    with st.spinner("Fetching MedlinePlus / WHO, indexing and summarising..."):
        res = medical_info_search(query=q, patient_context=ctx)
    if res.get("ok"):
        st.markdown(res["answer"])
        st.markdown("**Sources:** " + " · ".join(f"[{s['n']}] [{s['title']}]({s['url']})" for s in res["sources"]))
        f = res["fetch"]
        st.caption(f"Live fetch: {'yes' if f.get('live') else 'no (cached corpus)'} · {f.get('fetched', 0)} new docs · mode {res['mode']}"
                   + (f" · fetch warnings: {'; '.join(f['errors'])}" if f.get("errors") else ""))
        with st.expander("Retrieved passages (RAG context)"):
            st.dataframe(pd.DataFrame(res["retrieved"]), hide_index=True, width="stretch")
    else:
        st.error(res.get("error"))

st.subheader("Latest retrieved medical information")
rows = db.query("""SELECT m.ts, m.question, m.answer, m.sources, m.live_fetch, m.mode, p.name AS patient
                   FROM medical_answers m LEFT JOIN patients p ON p.patient_id=m.patient_id ORDER BY m.id DESC LIMIT 15""")
if not rows:
    st.info("No searches yet.")
for r in rows:
    with st.expander(f"{r['ts']} - {r['question']}" + (f" (for {r['patient']})" if r["patient"] else "")):
        st.markdown(r["answer"])
        for s in json.loads(r["sources"] or "[]"):
            st.markdown(f"{s['n']}. [{s['title']}]({s['url']}) - {s['source']}")
        st.caption(f"live fetch: {bool(r['live_fetch'])} · generation mode: {r['mode']}")

with st.expander("Browse knowledge base documents"):
    st.dataframe(pd.DataFrame(db.query("SELECT source, title, url, fetched_at, length(content) AS chars FROM medical_docs ORDER BY fetched_at DESC")),
                 hide_index=True, width="stretch")
