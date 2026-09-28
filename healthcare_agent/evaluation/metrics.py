"""Evaluation metrics. LLM-graded correctness uses LangChain's QAEvalChain
when a model is configured; the reference-based metrics below always run, so
results are comparable across offline and online modes."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from ..vectorstore import embed

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to in on for with is are was were be been by as at it this that from have has had "
            "you your can may will their they he she his her its not no do does".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def semantic_similarity(a: str, b: str) -> float:
    va, vb = embed([a, b])
    return float(np.dot(va, vb))


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(r))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l_recall(pred: str, ref: str) -> float:
    """LCS-based recall of reference tokens found (in order) in the prediction."""
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    dp = [0] * (len(p) + 1)
    for rt in r:
        prev = 0
        for j, pt in enumerate(p, 1):
            cur = dp[j]
            dp[j] = prev + 1 if rt == pt else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def keyword_coverage(pred: str, keywords: Iterable[str]) -> float:
    """Fraction of key concepts present. Each keyword may be 'a|b' alternatives (regex)."""
    kws = list(keywords)
    if not kws:
        return 1.0
    text = (pred or "").lower()
    return sum(bool(re.search(k, text)) for k in kws) / len(kws)


def groundedness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Share of answer sentences whose best cosine similarity to any retrieved
    context passage clears a threshold - a cheap hallucination proxy."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer or "")
             if len(s.strip()) > 30 and not s.strip().startswith(("_", "#", "**Sources", "Discuss any"))]
    ctx = [c for c in contexts if c]
    if not sents or not ctx:
        return 0.0
    sims = embed(sents) @ embed(ctx).T
    return float((sims.max(axis=1) >= threshold).mean())


def set_prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def qa_eval_chain_grades(llm, examples: list[dict], predictions: list[dict]) -> list[str]:
    """Grade with LangChain's QAEvalChain -> 'CORRECT' / 'INCORRECT' per example."""
    from langchain_classic.evaluation.qa import QAEvalChain
    chain = QAEvalChain.from_llm(llm)
    graded = chain.evaluate(examples, predictions, question_key="query", answer_key="answer", prediction_key="result")
    out = []
    for g in graded:
        text = str(g.get("results", g.get("text", ""))).upper()
        out.append("INCORRECT" if "INCORRECT" in text else "CORRECT" if "CORRECT" in text else "UNKNOWN")
    return out
