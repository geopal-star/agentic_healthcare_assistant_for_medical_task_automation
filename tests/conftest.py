"""Tests run against an isolated database / vector store in a temp dir,
with no network and no LLM (deterministic offline mode)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hc")
    os.environ.update({
        "DB_PATH": str(tmp / "test.db"), "VECTOR_DIR": str(tmp / "vs"), "REPORTS_DIR": str(tmp / "reports"),
        "MEDICAL_SEARCH_OFFLINE": "1",
        "LLM_PROVIDER": "offline",
    })
    from healthcare_agent import config, llm, vectorstore
    config.reset_settings()
    llm.reset_llm()
    vectorstore.reset_stores()
    from healthcare_agent.build import build
    build(reset=True, verbose=False)
    yield tmp


@pytest.fixture
def as_actor():
    from healthcare_agent.context import Actor, current_actor
    tokens = []

    def _set(user_id: str, name: str, role: str = "patient"):
        tokens.append(current_actor.set(Actor(user_id, name, role)))
        return Actor(user_id, name, role)
    yield _set
    for t in reversed(tokens):
        current_actor.reset(t)


@pytest.fixture(scope="session")
def agent():
    from healthcare_agent.agent import HealthcareAgent
    return HealthcareAgent()
