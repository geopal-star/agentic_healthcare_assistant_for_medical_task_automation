"""Run-scoped tracing: every tool call, LLM call and memory operation is written
to SQLite so the Streamlit "Memory & Logs" page can replay what the agent did."""
from __future__ import annotations

import contextvars
import functools
import time
import uuid
from typing import Any, Callable

from . import db

current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run_id", default=None)


def new_run_id() -> str:
    return "run-" + uuid.uuid4().hex[:10]


def _short(obj: Any, limit: int = 1500) -> str:
    s = db.to_json(obj)
    return s if len(s) <= limit else s[:limit] + "...<truncated>"


def log_tool(tool: str, args: dict, result: Any, success: bool, error: str | None, latency_ms: int) -> None:
    db.insert("tool_logs", {
        "ts": db.now_iso(), "run_id": current_run_id.get(), "tool": tool,
        "args": _short(args), "result": _short(result), "success": int(success),
        "error": error, "latency_ms": latency_ms,
    })


def log_llm(purpose: str, mode: str, success: bool, latency_ms: int, error: str | None = None) -> None:
    db.insert("llm_calls", {
        "ts": db.now_iso(), "run_id": current_run_id.get(), "purpose": purpose,
        "mode": mode, "success": int(success), "latency_ms": latency_ms, "error": error,
    })


def log_memory(op: str, subject_id: str | None, detail: Any) -> None:
    db.insert("memory_events", {
        "ts": db.now_iso(), "run_id": current_run_id.get(), "op": op,
        "subject_id": subject_id, "detail": _short(detail, 3000),
    })


def traced_tool(name: str) -> Callable:
    """Decorator: time the call, log args/result/success. A tool signals a
    *business* failure (e.g. no slots) by returning {"ok": False, ...}."""
    def deco(fn: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> dict:
            t0 = time.perf_counter()
            try:
                result = fn(**kwargs)
                ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
                err = None if ok else str(result.get("error", "tool reported failure"))
            except Exception as exc:  # tool crashed: surface it as a failed step
                result, ok, err = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, False, str(exc)
            log_tool(name, kwargs, result, ok, err, int((time.perf_counter() - t0) * 1000))
            return result
        wrapper.tool_name = name  # type: ignore[attr-defined]
        return wrapper
    return deco
