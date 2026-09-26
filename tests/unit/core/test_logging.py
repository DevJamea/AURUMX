"""Structured logging tests (spec §51) and secret redaction."""

from __future__ import annotations

import json
import logging

from app.core.logging import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    get_logger,
    redact,
    reset_logging,
)


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def attach_capture() -> tuple[CaptureHandler, logging.Logger]:
    handler = CaptureHandler()
    root = logging.getLogger("aurumx")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return handler, root


class TestJsonLogging:
    def setup_method(self):
        configure_logging(level="DEBUG", log_format="json", propagate=True)

    def teardown_method(self):
        reset_logging()

    def test_event_and_details_are_structured_fields(self):
        handler, _ = attach_capture()
        log = get_logger("test.component")
        log.info("connected", event="MT5_CONNECTED", login=123, server="demo")

        assert len(handler.records) == 1
        record = handler.records[0]
        assert record.event == "MT5_CONNECTED"
        assert record.details == {"login": 123, "server": "demo"}

        line = JsonFormatter().format(record)
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["component"] == "aurumx.test.component"
        assert payload["event"] == "MT5_CONNECTED"
        assert payload["message"] == "connected"
        assert payload["details"] == {"login": 123, "server": "demo"}
        assert "timestamp" in payload

    def test_secrets_are_redacted(self):
        handler, _ = attach_capture()
        log = get_logger("test.secrets")
        log.info(
            "connecting",
            event="MT5_CONNECTING",
            password="hunter2",
            api_key="abcdef",
            nested={"token": "xyz", "safe": 1},
            items=[{"secret": "s"}],
        )
        details = handler.records[0].details
        assert details["password"] == "***REDACTED***"
        assert details["api_key"] == "***REDACTED***"
        assert details["nested"]["token"] == "***REDACTED***"
        assert details["nested"]["safe"] == 1
        assert details["items"][0]["secret"] == "***REDACTED***"

    def test_named_event_helper(self):
        handler, _ = attach_capture()
        get_logger("test.events").event("SYMBOL_DISCOVERED", symbol="XAUUSD")
        record = handler.records[0]
        assert record.event == "SYMBOL_DISCOVERED"
        assert record.details == {"symbol": "XAUUSD"}


class TestTextLogging:
    def test_text_format_contains_event(self):
        record = logging.LogRecord(
            name="aurumx.x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.event = "TEST_EVENT"
        record.details = {"a": 1}
        line = TextFormatter().format(record)
        assert "event=TEST_EVENT" in line
        assert "hello" in line


class TestRedact:
    def test_redact_handles_none_and_plain_values(self):
        assert redact(None) == {}
        assert redact({"plain": "value"}) == {"plain": "value"}
        assert redact({"PASSWORD": "x"})["PASSWORD"] == "***REDACTED***"
