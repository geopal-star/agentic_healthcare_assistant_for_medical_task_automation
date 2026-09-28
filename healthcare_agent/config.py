"""Central configuration. Every setting can be overridden through environment
variables (or a `.env` file in the project root)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Default model per provider. Used when LLM_MODEL is not set.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "ollama": "llama3.1",
}

_KEY_FOR_PROVIDER = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _detect_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        return explicit
    for provider, key in _KEY_FOR_PROVIDER.items():
        if _env(key):
            return provider
    return "offline"


@dataclass
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    raw_dir: Path = field(default_factory=lambda: Path(_env("RAW_DATA_DIR", str(ROOT / "data" / "raw"))))
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "data" / "healthcare.db"))))
    vector_dir: Path = field(default_factory=lambda: Path(_env("VECTOR_DIR", str(ROOT / "data" / "vectorstore"))))
    knowledge_seed: Path = field(default_factory=lambda: ROOT / "data" / "knowledge_seed.json")
    reports_dir: Path = field(default_factory=lambda: Path(_env("REPORTS_DIR", str(ROOT / "reports"))))

    llm_provider: str = field(default_factory=_detect_provider)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_timeout: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT", "60")))

    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # When true the medical search never touches the network (tests, air-gapped demos).
    search_offline: bool = field(default_factory=lambda: _env("MEDICAL_SEARCH_OFFLINE", "0") == "1")
    search_timeout: int = field(default_factory=lambda: int(_env("SEARCH_TIMEOUT", "15")))

    slot_horizon_days: int = 21
    slot_minutes: int = 30

    def __post_init__(self) -> None:
        if not self.llm_model and self.llm_provider in DEFAULT_MODELS:
            self.llm_model = DEFAULT_MODELS[self.llm_provider]
        for p in (self.data_dir, self.vector_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "offline"

    @property
    def llm_label(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}" if self.llm_enabled else "offline (heuristic)"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Re-read the environment (used by tests that point DB_PATH elsewhere)."""
    global _settings
    _settings = None
