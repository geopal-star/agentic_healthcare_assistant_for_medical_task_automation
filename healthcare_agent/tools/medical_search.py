"""Trusted medical information search + RAG.

Pipeline:  query -> fetch (MedlinePlus Web Service API, WHO via site-restricted web search
+ fact-sheet scrape) -> cache in SQLite -> chunk & embed into the ``medical_knowledge`` FAISS
index -> retrieve top-k chunks -> LLM answer grounded on numbered excerpts with citations.

Network failures degrade gracefully to the already-indexed corpus (seeded at build time).
"""
from __future__ import annotations

import hashlib
import html
import re
from typing import Any

import requests
import xmltodict

from .. import db
from ..config import get_settings
from ..llm import get_llm
from ..observability import current_run_id, traced_tool
from ..prompts import MEDICAL_RAG_SYSTEM, MEDICAL_RAG_USER
from ..vectorstore import chunk_text, embed, get_store

STORE = "medical_knowledge"
MEDLINE_URL = "https://wsearch.nlm.nih.gov/ws/query"
UA = {"User-Agent": "Mozilla/5.0 (HealthcareAssistant capstone; educational use)"}


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_medlineplus(term: str, n: int = 3) -> list[dict[str, Any]]:
    r = requests.get(MEDLINE_URL, params={"db": "healthTopics", "term": term, "retmax": n},
                     headers=UA, timeout=get_settings().search_timeout)
    r.raise_for_status()
    data = xmltodict.parse(r.text)
    docs = data.get("nlmSearchResult", {}).get("list", {}).get("document", []) or []
    docs = docs if isinstance(docs, list) else [docs]
    out = []
    for d in docs:
        contents = d.get("content", [])
        contents = contents if isinstance(contents, list) else [contents]
        fields: dict[str, str] = {}
        for c in contents:
            if isinstance(c, dict):
                fields.setdefault(c.get("@name"), c.get("#text", ""))
        body = _strip_html(fields.get("FullSummary", "")) or _strip_html(fields.get("snippet", ""))
        if body:
            out.append({"source": "MedlinePlus (NIH/NLM)", "title": _strip_html(fields.get("title", term)),
                        "url": d.get("@url"), "content": body})
    return out


def _scrape_who_page(url: str) -> str:
    from bs4 import BeautifulSoup
    r = requests.get(url, headers=UA, timeout=get_settings().search_timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.select_one("article") or soup.select_one("div.sf-detail-body-wrapper") or soup.body
    parts = [el.get_text(" ", strip=True) for el in main.find_all(["h2", "h3", "p", "li"])] if main else []
    text = " ".join(p for p in parts if len(p) > 25)
    return text[:8000]


_NON_ENGLISH = re.compile(r"who\.int/(?:fr|es|ar|zh|ru|pt)/")


def fetch_who(term: str, n: int = 3) -> list[dict[str, Any]]:
    from ddgs import DDGS
    results = DDGS().text(f"{term} site:who.int", max_results=n + 2) or []
    out = []
    for res in results:
        url = res.get("href", "")
        if "who.int" not in url or ".pdf" in url.lower() or _NON_ENGLISH.search(url):
            continue
        content = res.get("body", "")
        if "/fact-sheets/" in url or "/health-topics/" in url:
            for _ in range(2):             # one retry: WHO occasionally throttles
                try:
                    content = _scrape_who_page(url) or content
                    break
                except Exception:
                    continue
        content = _strip_html(content)
        if len(content) < 150:
            continue
        out.append({"source": "World Health Organization", "title": res.get("title", term), "url": url,
                    "content": content})
        if len(out) >= n:
            break
    return out


def store_documents(docs: list[dict[str, Any]], query: str = "") -> int:
    """Cache documents in SQLite and index their chunks. Returns #new chunks."""
    texts, metas = [], []
    for d in docs:
        if not d.get("url") or not d.get("content"):
            continue
        did = _doc_id(d["url"])
        db.insert("medical_docs", {"doc_id": did, "source": d["source"], "title": d["title"], "url": d["url"],
                                   "content": d["content"], "query": query,
                                   "fetched_at": d.get("fetched_at") or db.now_iso()}, or_replace=True)
        for chunk in chunk_text(d["content"]):
            texts.append(f"{d['title']}: {chunk}")
            metas.append({"doc_id": did, "source": d["source"], "title": d["title"], "url": d["url"]})
    return get_store(STORE).add(texts, metas)


def fetch_and_index(query: str) -> dict[str, Any]:
    """Live fetch from both trusted sources; errors are collected, not raised."""
    if get_settings().search_offline:
        return {"live": False, "errors": ["offline mode"], "fetched": 0}
    docs, errors = [], []
    for name, fn in (("MedlinePlus", fetch_medlineplus), ("WHO", fetch_who)):
        try:
            docs += fn(query)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:200])
    store_documents(docs, query)
    return {"live": bool(docs), "errors": errors, "fetched": len(docs)}


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def retrieve(question: str, k: int = 6, per_doc: int = 2, min_score: float = 0.45) -> list[dict[str, Any]]:
    hits = get_store(STORE).search(question, k=k * 3, min_score=min_score)
    out, counts = [], {}
    for h in hits:
        url = h["metadata"]["url"]
        if counts.get(url, 0) >= per_doc:
            continue
        counts[url] = counts.get(url, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out


def _extractive_answer(question: str, hits: list[dict], sources: list[dict], n_points: int = 7,
                       diversity: float = 0.35) -> str:
    """Offline fallback: Maximal Marginal Relevance over source sentences -
    relevant to the question but not redundant with each other - each cited."""
    url_to_n = {s["url"]: s["n"] for s in sources}
    sentences, seen = [], set()
    for h in hits:
        body = h["text"].split(": ", 1)[-1]
        for s in re.split(r"(?<=[.!?])\s+", body):
            s = s.strip()
            if 40 <= len(s) <= 400 and s[0].isupper() and not s.endswith("?") and s[:60].lower() not in seen:
                seen.add(s[:60].lower())                     # skip fragments, headings, duplicates
                sentences.append((s, url_to_n[h["metadata"]["url"]]))
    if not sentences:
        return "No relevant information was found in the trusted sources."
    vecs = embed([s for s, _ in sentences])
    relevance = vecs @ embed([question])[0]
    chosen: list[int] = []
    while len(chosen) < min(n_points, len(sentences)):
        redundancy = (vecs @ vecs[chosen].T).max(axis=1) if chosen else 0
        score = (1 - diversity) * relevance - diversity * redundancy
        if chosen:
            score[chosen] = -1e9
        chosen.append(int(score.argmax()))
    picked = [f"- {sentences[i][0]} [{sentences[i][1]}]" for i in sorted(chosen, key=lambda i: -relevance[i])]
    return ("Key points from trusted sources (extractive summary):\n" + "\n".join(picked) +
            "\n\nDiscuss any treatment changes with the treating doctor.")


def answer_question(question: str, patient_context: str = "", k: int = 6) -> dict[str, Any]:
    fetch = fetch_and_index(question)
    hits = retrieve(question, k=k)
    if not hits:
        return {"ok": False, "error": "No trusted-source content found for this question.",
                "fetch": fetch, "sources": []}
    sources, url_n = [], {}
    for h in hits:
        md = h["metadata"]
        if md["url"] not in url_n:
            url_n[md["url"]] = len(url_n) + 1
            sources.append({"n": url_n[md["url"]], "title": md["title"], "url": md["url"], "source": md["source"]})
    excerpts = "\n\n".join(f"[{url_n[h['metadata']['url']]}] ({h['metadata']['source']} - {h['metadata']['title']})\n"
                           f"{h['text'][:1200]}" for h in hits)
    answer, mode = get_llm().text(
        "medical_rag", MEDICAL_RAG_SYSTEM,
        MEDICAL_RAG_USER.format(question=question, patient_context=patient_context or "none", sources=excerpts),
        fallback=lambda: _extractive_answer(question, hits, sources))
    return {"ok": True, "answer": answer, "sources": sources, "mode": mode, "fetch": fetch,
            "contexts": [h["text"] for h in hits],
            "retrieved": [{"text": h["text"][:300], "score": round(h["score"], 3), "url": h["metadata"]["url"]}
                          for h in hits]}


@traced_tool("medical_info_search")
def medical_info_search(query: str, patient_context: str = "", patient_id: str | None = None) -> dict[str, Any]:
    """Search MedlinePlus + WHO and return a cited, grounded summary."""
    if not query:
        return {"ok": False, "error": "empty query"}
    res = answer_question(query, patient_context)
    if res.get("ok"):
        db.insert("medical_answers", {
            "ts": db.now_iso(), "run_id": current_run_id.get(), "patient_id": patient_id, "question": query,
            "answer": res["answer"], "sources": db.to_json(res["sources"]),
            "live_fetch": int(res["fetch"].get("live", False)), "mode": res["mode"]})
    return res


def load_seed(path) -> int:
    """Load the pre-fetched knowledge seed (offline corpus) into cache + index."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return 0
    docs = json.loads(p.read_text(encoding="utf-8"))
    return store_documents(docs, query="seed")
