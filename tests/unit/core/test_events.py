"""Event bus tests: delivery, isolation, thread-safety basics."""

from __future__ import annotations

import threading

from app.core.events import Event, EventBus, get_event_bus, set_event_bus


class TestEventBus:
    def setup_method(self):
        self.bus = EventBus()

    def test_typed_subscription_receives_event(self):
        received: list[Event] = []
        self.bus.subscribe(received.append, event_type="SYMBOL_DISCOVERED")

        self.bus.emit("SYMBOL_DISCOVERED", "test", symbol="XAUUSD")

        assert len(received) == 1
        assert received[0].type == "SYMBOL_DISCOVERED"
        assert received[0].component == "test"
        assert received[0].payload == {"symbol": "XAUUSD"}
        assert received[0].timestamp is not None

    def test_wildcard_subscription_receives_everything(self):
        received: list[Event] = []
        self.bus.subscribe(received.append)

        self.bus.emit("A", "c")
        self.bus.emit("B", "c")

        assert [e.type for e in received] == ["A", "B"]

    def test_unsubscribe_stops_delivery(self):
        received: list[Event] = []
        unsubscribe = self.bus.subscribe(received.append, event_type="X")
        unsubscribe()
        self.bus.emit("X", "c")
        assert received == []

    def test_handler_exception_does_not_break_other_handlers(self):
        received: list[Event] = []

        def broken(_: Event) -> None:
            raise RuntimeError("boom")

        self.bus.subscribe(broken, event_type="X")
        self.bus.subscribe(received.append, event_type="X")

        self.bus.emit("X", "c")  # must not raise

        assert len(received) == 1

    def test_history_and_clear(self):
        self.bus.emit("X", "c", a=1)
        self.bus.emit("Y", "c")
        assert [e.type for e in self.bus.history()] == ["X", "Y"]
        assert len(self.bus.history(event_type="X")) == 1
        self.bus.clear()
        assert self.bus.history() == []

    def test_thread_safety_smoke(self):
        received: list[Event] = []
        self.bus.subscribe(received.append)

        def publish_many() -> None:
            for i in range(200):
                self.bus.emit("T", "thread", i=i)

        threads = [threading.Thread(target=publish_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 800


class TestDefaultBus:
    def test_default_bus_is_shared(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_set_event_bus_replaces_default(self):
        original = get_event_bus()
        try:
            fresh = EventBus()
            set_event_bus(fresh)
            assert get_event_bus() is fresh
        finally:
            set_event_bus(original)
