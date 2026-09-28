"""Pre-fetch a small trusted-source corpus (MedlinePlus + WHO) into
data/knowledge_seed.json so the RAG pipeline works offline / in air-gapped
demos. Live searches at runtime add to this corpus.

    python scripts/build_knowledge_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from healthcare_agent import db  # noqa: E402
from healthcare_agent.config import get_settings  # noqa: E402
from healthcare_agent.tools.medical_search import fetch_medlineplus, fetch_who  # noqa: E402

TOPICS = [
    "chronic kidney disease", "chronic kidney disease treatment", "kidney failure dialysis", "diabetic kidney disease",
    "high blood pressure", "type 2 diabetes", "upper respiratory infection", "common cold", "costochondritis chest pain",
    "migraine", "polycystic ovary syndrome", "heart disease", "asthma", "influenza", "healthy diet",
]


def main() -> None:
    docs: dict[str, dict] = {}
    for topic in TOPICS:
        for name, fn in (("MedlinePlus", fetch_medlineplus), ("WHO", fetch_who)):
            try:
                got = fn(topic, n=2)
            except Exception as exc:  # keep going; partial corpus is fine
                print(f"  ! {name} '{topic}': {exc}")
                continue
            for d in got:
                docs.setdefault(d["url"], {**d, "fetched_at": db.now_iso(), "seed_topic": topic})
            print(f"{name:12s} {topic:40s} +{len(got)}")
    out = get_settings().knowledge_seed
    out.write_text(json.dumps(list(docs.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(docs)} documents -> {out}")


if __name__ == "__main__":
    main()
