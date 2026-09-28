"""A small persistent FAISS vector store with metadata filtering.

Two collections are used:
  * ``patient_memory``    - patient summaries, encounter notes, long-term memories
  * ``medical_knowledge`` - chunks of MedlinePlus / WHO documents (RAG corpus)

Embeddings are L2-normalised so inner product == cosine similarity.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from .config import get_settings

_embedder = None
_embed_lock = threading.Lock()


def get_embedder():
    global _embedder
    with _embed_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware character chunker."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to break on a sentence boundary
            cut = text.rfind(". ", start + size // 2, end)
            end = cut + 1 if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class VectorStore:
    def __init__(self, name: str, directory: Path | None = None):
        self.name = name
        self.dir = (directory or get_settings().vector_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.index: faiss.Index | None = None
        self.docs: list[dict[str, Any]] = []   # {"id", "text", "metadata"}
        self._ids: set[str] = set()
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        idx, meta = self.dir / "index.faiss", self.dir / "docs.json"
        if idx.exists() and meta.exists():
            self.index = faiss.read_index(str(idx))
            self.docs = json.loads(meta.read_text(encoding="utf-8"))
            self._ids = {d["id"] for d in self.docs}

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.dir / "index.faiss"))
        (self.dir / "docs.json").write_text(json.dumps(self.docs, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self.index, self.docs, self._ids = None, [], set()
            for f in self.dir.glob("*"):
                f.unlink()

    # -- write -------------------------------------------------------------
    @staticmethod
    def make_id(text: str, metadata: dict) -> str:
        key = text + json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Add texts (deduplicated by content+metadata). Returns #added."""
        metadatas = metadatas or [{} for _ in texts]
        new = [(t, m, self.make_id(t, m)) for t, m in zip(texts, metadatas) if t and t.strip()]
        new = [x for x in new if x[2] not in self._ids]
        if not new:
            return 0
        vecs = embed([t for t, _, _ in new])
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            for t, m, i in new:
                self.docs.append({"id": i, "text": t, "metadata": m})
                self._ids.add(i)
            self.save()
        return len(new)

    # -- read --------------------------------------------------------------
    def search(self, query: str, k: int = 4,
               where: dict[str, Any] | Callable[[dict], bool] | None = None,
               min_score: float = 0.0) -> list[dict[str, Any]]:
        if self.index is None or not self.docs:
            return []
        if isinstance(where, dict):
            cond = where
            where = lambda md: all(md.get(key) == val for key, val in cond.items())  # noqa: E731
        qv = embed([query])
        # Over-fetch when filtering, since FAISS flat index has no native filter.
        fetch = len(self.docs) if where else min(k, len(self.docs))
        scores, idxs = self.index.search(qv, fetch)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            d = self.docs[i]
            if where and not where(d["metadata"]):
                continue
            if score < min_score:
                continue
            out.append({**d, "score": float(score)})
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self.docs)


_stores: dict[str, VectorStore] = {}


def get_store(name: str) -> VectorStore:
    if name not in _stores:
        _stores[name] = VectorStore(name)
    return _stores[name]


def reset_stores() -> None:
    _stores.clear()
