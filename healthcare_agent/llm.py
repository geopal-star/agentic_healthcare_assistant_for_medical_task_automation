"""Provider-agnostic LLM access.

The agent only needs two capabilities from a model: *structured output* (the
planner, record extraction) and *free text* (summaries, the final answer).
Every call site passes an offline fallback, so the whole system keeps working
- with lower quality - when no API key is configured or the provider errors.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .config import Settings, get_settings
from .observability import log_llm

T = TypeVar("T", bound=BaseModel)

# Client read-timeouts are not enough: some gateways (e.g. OpenRouter) keep a slow
# connection alive with keep-alive bytes, so a request can hang for many minutes.
# Every model call therefore runs under a hard wall-clock deadline.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


class LLMDeadlineExceeded(TimeoutError):
    pass


def _with_deadline(fn: Callable[[], Any], seconds: float) -> Any:
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout:
        future.cancel()
        raise LLMDeadlineExceeded(f"no response within {seconds:.0f}s") from None


def build_chat_model(settings: Settings):
    """Instantiate a LangChain chat model for the configured provider."""
    p = settings.llm_provider
    common = dict(temperature=settings.llm_temperature, timeout=settings.llm_timeout, max_retries=1)
    if p == "openrouter":
        from langchain_openai import ChatOpenAI
        # Route to the fastest upstream provider; default routing was ~5x slower and prone to stalls.
        return ChatOpenAI(model=settings.llm_model, base_url="https://openrouter.ai/api/v1",
                          api_key=os.environ["OPENROUTER_API_KEY"],
                          extra_body={"provider": {"sort": "throughput"}}, **common)
    from langchain.chat_models import init_chat_model
    return init_chat_model(settings.llm_model, model_provider=p, **common)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Anthropic-style content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class LLM:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self.last_error: str | None = None

    @property
    def live(self) -> bool:
        return self.settings.llm_enabled

    @property
    def label(self) -> str:
        return self.settings.llm_label

    def chat_model(self):
        if not self.live:
            return None
        if self._model is None:
            self._model = build_chat_model(self.settings)
        return self._model

    # -- structured -----------------------------------------------------
    def structured(self, purpose: str, system: str, user: str, schema: type[T],
                   fallback: Callable[[], T]) -> tuple[T, str]:
        """Returns (result, mode) where mode is live / offline / fallback."""
        if not self.live:
            t0 = time.perf_counter()
            out = fallback()
            log_llm(purpose, "offline", True, int((time.perf_counter() - t0) * 1000))
            return out, "offline"

        t0 = time.perf_counter()
        messages = [("system", system), ("human", user)]
        try:
            try:
                out = _with_deadline(lambda: self.chat_model().with_structured_output(schema).invoke(messages),
                                     self.settings.llm_timeout)
                if out is None:
                    raise ValueError("model returned no structured output")
            except LLMDeadlineExceeded:
                raise                      # the provider is stalled; don't spend a second deadline on it
            except Exception:
                # Second chance: plain JSON prompting (some models dislike tool-calling).
                schema_json = json.dumps(schema.model_json_schema())
                json_messages = messages + [(
                    "human", f"Respond ONLY with a JSON object matching this JSON schema:\n{schema_json}")]
                raw = _with_deadline(lambda: self.chat_model().invoke(json_messages), self.settings.llm_timeout)
                out = schema.model_validate_json(_extract_json(_content_to_text(raw.content)))
            log_llm(purpose, "live", True, int((time.perf_counter() - t0) * 1000))
            return out, "live"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log_llm(purpose, "fallback", False, int((time.perf_counter() - t0) * 1000), self.last_error)
            return fallback(), "fallback"

    # -- text ------------------------------------------------------------
    def text(self, purpose: str, system: str, user: str, fallback: Callable[[], str]) -> tuple[str, str]:
        if not self.live:
            out = fallback()
            log_llm(purpose, "offline", True, 0)
            return out, "offline"
        t0 = time.perf_counter()
        try:
            resp = _with_deadline(lambda: self.chat_model().invoke([("system", system), ("human", user)]),
                                  self.settings.llm_timeout)
            out = _content_to_text(resp.content).strip()
            if not out:
                raise ValueError("empty completion")
            log_llm(purpose, "live", True, int((time.perf_counter() - t0) * 1000))
            return out, "live"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log_llm(purpose, "fallback", False, int((time.perf_counter() - t0) * 1000), self.last_error)
            return fallback(), "fallback"


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None
