"""Structured logging (spec §51).

Every important event is logged as one JSON line with the fields::

    timestamp, level, component, event, details

Secrets (passwords, tokens, API keys) are redacted before formatting, so a
misconfigured ``details`` payload can never leak a credential into the logs.

Usage::

    from app.core.logging import get_logger

    log = get_logger("brokers.mt5")
    log.info("connected to terminal", event="MT5_CONNECTED", login=123456)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

_LOGGER_NAMESPACE = "aurumx"

#: Keys whose *values* must never appear in logs.
_SECRET_KEY_PATTERN = re.compile(r"(password|passwd|secret|token|api_?key|credential)", re.I)
_REDACTED = "***REDACTED***"
_MAX_REDACT_DEPTH = 4


def redact(details: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``details`` with secret-looking values masked."""

    def _walk(value: Any, depth: int) -> Any:
        if depth > _MAX_REDACT_DEPTH:
            return _REDACTED
        if isinstance(value, dict):
            return {
                key: _REDACTED if _SECRET_KEY_PATTERN.search(str(key)) else _walk(v, depth + 1)
                for key, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_walk(v, depth + 1) for v in value]
        return value

    if not details:
        return {}
    return _walk(details, 0)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        details = getattr(record, "details", None)
        if details:
            payload["details"] = details
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable single-line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
        base = f"{ts} {record.levelname:<7} {record.name} | {record.getMessage()}"
        event = getattr(record, "event", None)
        if event:
            base += f" | event={event}"
        details = getattr(record, "details", None)
        if details:
            base += f" | {json.dumps(details, default=str, ensure_ascii=False)}"
        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


class AurumXLogger:
    """Logger wrapper adding ``event`` and structured ``details`` to records."""

    def __init__(self, component: str) -> None:
        self._logger = logging.getLogger(f"{_LOGGER_NAMESPACE}.{component}")

    @property
    def name(self) -> str:
        return self._logger.name

    def _emit(
        self,
        level: int,
        message: str,
        event: str | None,
        details: dict[str, Any],
        exc_info: bool,
    ) -> None:
        self._logger.log(
            level,
            message,
            exc_info=exc_info,
            extra={"event": event or "", "details": redact(details)},
        )

    def debug(self, message: str, *, event: str | None = None, **details: Any) -> None:
        self._emit(logging.DEBUG, message, event, details, exc_info=False)

    def info(self, message: str, *, event: str | None = None, **details: Any) -> None:
        self._emit(logging.INFO, message, event, details, exc_info=False)

    def warning(self, message: str, *, event: str | None = None, **details: Any) -> None:
        self._emit(logging.WARNING, message, event, details, exc_info=False)

    def error(self, message: str, *, event: str | None = None, **details: Any) -> None:
        self._emit(logging.ERROR, message, event, details, exc_info=True)

    def event(self, name: str, *, level: str = "info", message: str | None = None, **details: Any) -> None:
        """Log a named event, e.g. ``log.event("SYMBOL_DISCOVERED", symbol=...)``."""
        levelno = logging.getLevelName(level.upper())
        if not isinstance(levelno, int):
            levelno = logging.INFO
        self._emit(levelno, message or name, name, details, exc_info=False)


def get_logger(component: str) -> AurumXLogger:
    """Return a structured logger for ``component`` (e.g. ``"market.data_service"``)."""
    return AurumXLogger(component)


def configure_logging(
    *,
    level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
    stream: Any | None = None,
    propagate: bool = False,
) -> None:
    """Configure the ``aurumx`` logger namespace.  Idempotent (handlers reset).

    ``propagate=True`` is used by tests so ``caplog`` can capture records.
    """

    root = logging.getLogger(_LOGGER_NAMESPACE)
    root.handlers.clear()
    root.setLevel(level.upper())
    root.propagate = propagate

    formatter: logging.Formatter = JsonFormatter() if log_format == "json" else TextFormatter()

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        from pathlib import Path

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())  # files are always JSON (machine-readable)
        root.addHandler(file_handler)


def reset_logging() -> None:
    """Remove all handlers (test helper)."""
    root = logging.getLogger(_LOGGER_NAMESPACE)
    root.handlers.clear()
    root.propagate = False
