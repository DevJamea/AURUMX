"""In-process event bus for observability and loose coupling.

Components publish domain events (``SYMBOL_DISCOVERED``, ``MT5_DISCONNECTED``, …)
instead of calling each other.  Handlers are isolated: an exception in one
handler is logged and never prevents other handlers (or the publisher) from
running.

This is intentionally a small synchronous bus.  When the API layer (Phase 7)
needs websocket fan-out it subscribes to the same bus.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger

log = get_logger("core.events")

EventHandler = Callable[["Event"], None]


class Event:
    """A domain event.  ``type`` uses SCREAMING_SNAKE codes (spec §51)."""

    __slots__ = ("type", "component", "timestamp", "payload")

    def __init__(self, type: str, component: str, payload: dict[str, Any] | None = None) -> None:
        self.type = type
        self.component = component
        self.timestamp = datetime.now(UTC)
        self.payload = payload or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Event({self.type!r}, component={self.component!r}, payload={self.payload!r})"


class EventBus:
    """Thread-safe synchronous pub/sub."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_type: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard: list[EventHandler] = []
        self._history: list[Event] = []
        self._history_limit = 500

    def subscribe(self, handler: EventHandler, *, event_type: str | None = None) -> Callable[[], None]:
        """Subscribe ``handler`` to one event type (``None`` = all events).

        Returns an unsubscribe callable.
        """
        with self._lock:
            if event_type is None:
                self._wildcard.append(handler)
            else:
                self._by_type[event_type].append(handler)

        def _unsubscribe() -> None:
            with self._lock:
                if event_type is None:
                    self._wildcard = [h for h in self._wildcard if h is not handler]
                else:
                    bucket = self._by_type.get(event_type, [])
                    self._by_type[event_type] = [h for h in bucket if h is not handler]

        return _unsubscribe

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to all matching handlers, isolating failures."""
        with self._lock:
            handlers = list(self._by_type.get(event.type, ())) + list(self._wildcard)
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]

        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - isolation is the point of the bus
                log.error(
                    "event handler failed",
                    event="EVENT_HANDLER_ERROR",
                    handler=getattr(handler, "__name__", repr(handler)),
                    event_type=event.type,
                )

    def emit(self, type: str, component: str, **payload: Any) -> None:
        """Convenience: build and publish an event in one call."""
        self.publish(Event(type=type, component=component, payload=payload))

    def history(self, event_type: str | None = None) -> list[Event]:
        with self._lock:
            events = list(self._history)
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        return events

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Process-wide default bus."""
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = EventBus()
        return _bus


def set_event_bus(bus: EventBus | None) -> None:
    """Replace the default bus (used by tests / fresh application contexts)."""
    global _bus
    with _bus_lock:
        _bus = bus
