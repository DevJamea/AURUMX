"""Market data service — the read-only entry point for market data (Phase 1).

Ties together: broker connection, gold symbol discovery, tick validation and
multi-timeframe candle fetching, producing ``MarketSnapshot`` objects.  This is
the object the trading worker (Phase 4+) will poll; agents (Phase 2) receive
snapshots and never see a broker.

Broker clock correction: MT5 servers typically run UTC+2/+3.  The service
estimates the server-clock offset from the freshest valid tick (bounded to
±6h) and passes it to the validators so freshness is judged in the server's
frame of reference.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.brokers.interface import BrokerInterface
from app.core.config import AppConfig
from app.core.enums import TimeFrame
from app.core.events import get_event_bus
from app.core.logging import get_logger
from app.core.models import SymbolSpec, ValidationReport
from app.market.candles import CandleValidator, fetch_timeframes
from app.market.market_state import MarketSnapshot, infer_session_state
from app.market.symbol_discovery import SymbolDiscovery
from app.market.tick import TickValidator

log = get_logger("market.data_service")

#: Broker servers are typically UTC+2/+3; anything beyond this bound is
#: treated as a data anomaly, not a clock offset (seconds).
_MAX_CLOCK_OFFSET_SECONDS = 6 * 3600

#: How far ahead of (offset-corrected) now a tick may be stamped before it is
#: considered future-dated (matches the tick validator tolerance).
_FUTURE_TICK_TOLERANCE = 60.0


class MarketDataService:
    """Read-only market data facade over any ``BrokerInterface``."""

    def __init__(
        self,
        broker: BrokerInterface,
        *,
        symbol: str = "AUTO",
        timeframes: list[TimeFrame] | None = None,
        candle_count: int = 400,
        max_tick_age_seconds: float = 60.0,
        candle_freshness_multiplier: float = 3.0,
        tick_validator: TickValidator | None = None,
        candle_validator: CandleValidator | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._broker = broker
        self._discovery = SymbolDiscovery(broker)
        self._symbol_setting = symbol
        self._timeframes: list[TimeFrame] = list(timeframes) if timeframes else [
            TimeFrame.M15, TimeFrame.H1, TimeFrame.H4
        ]
        self._candle_count = candle_count
        self._clock = clock
        self._tick_validator = tick_validator or TickValidator(
            max_age_seconds=max_tick_age_seconds, clock=clock
        )
        self._candle_validator = candle_validator or CandleValidator(
            freshness_multiplier=candle_freshness_multiplier, clock=clock
        )
        self._symbol_spec: SymbolSpec | None = None
        self._clock_offset = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        broker: BrokerInterface,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> MarketDataService:
        return cls(
            broker,
            symbol=config.symbol,
            timeframes=config.timeframes,
            candle_count=config.candle_count,
            max_tick_age_seconds=config.max_tick_age_seconds,
            candle_freshness_multiplier=config.candle_freshness_multiplier,
            clock=clock or (lambda: datetime.now(UTC)),
        )

    # ------------------------------------------------------------------
    @property
    def broker(self) -> BrokerInterface:
        return self._broker

    @property
    def symbol_spec(self) -> SymbolSpec | None:
        return self._symbol_spec

    @property
    def clock_offset_seconds(self) -> float:
        return self._clock_offset

    def connect(self) -> None:
        self._broker.connect()

    def disconnect(self) -> None:
        self._broker.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._broker.is_connected

    # ------------------------------------------------------------------
    def resolve_symbol(self, *, force: bool = False) -> SymbolSpec:
        """Discover (once) and return the verified gold symbol spec."""
        if self._symbol_spec is not None and not force:
            return self._symbol_spec
        result = self._discovery.discover(self._symbol_setting)
        self._symbol_spec = result.chosen
        get_event_bus().emit(
            "SYMBOL_DISCOVERED",
            component="market.data_service",
            symbol=result.chosen.name,
            method=result.method,
        )
        return self._symbol_spec

    # ------------------------------------------------------------------
    def get_snapshot(self) -> MarketSnapshot:
        """Build one coherent, validated view of the market.  Raises broker
        errors (e.g. ``MT5NotConnectedError``) — connectivity problems are for
        the health monitor, not silently embedded in snapshots."""
        spec = self.resolve_symbol()
        now = self._clock()

        raw_tick = self._broker.get_tick(spec.name)
        tick_check = self._tick_validator.validate(
            raw_tick, symbol_spec=spec, now=now, clock_offset_seconds=self._clock_offset
        )

        # ---- server-clock offset adoption (fail-closed) -------------------
        # MT5 servers typically run UTC+2/+3, so a tick can be legitimately
        # "in the future" when judged with a zero/old offset.  The offset is
        # only ever adopted in the future-stamped direction and within ±6h:
        # an offset that would make an *old* tick look fresh can never be
        # adopted, so a stale feed stays stale.  (A server clock moving
        # backwards — DST edge — surfaces as stale data and blocks trading.)
        if raw_tick is not None and tick_check.age_seconds is not None:
            candidate = raw_tick.time.timestamp() - now.timestamp()
            future_stamped = tick_check.age_seconds < -_FUTURE_TICK_TOLERANCE
            plausible = 0 < candidate <= _MAX_CLOCK_OFFSET_SECONDS
            if future_stamped and plausible and candidate > self._clock_offset:
                log.info(
                    "adopted broker server clock offset",
                    event="CLOCK_OFFSET_ADOPTED",
                    offset_seconds=candidate,
                )
                self._clock_offset = candidate
                tick_check = self._tick_validator.validate(
                    raw_tick,
                    symbol_spec=spec,
                    now=now,
                    clock_offset_seconds=self._clock_offset,
                )

        series_checks = fetch_timeframes(
            self._broker,
            spec.name,
            self._timeframes,
            self._candle_count,
            validator=self._candle_validator,
            now=now,
            clock_offset_seconds=self._clock_offset,
        )

        aggregate = ValidationReport()
        aggregate.extend(tick_check.report)
        for check in series_checks.values():
            aggregate.extend(check.report)

        snapshot = MarketSnapshot(
            created_at=now,
            symbol=spec,
            tick=tick_check,
            series=series_checks,
            clock_offset_seconds=self._clock_offset,
            session_state=infer_session_state(tick_check, series_checks),
            report=aggregate,
        )
        if not snapshot.trading_data_ok:
            log.warning(
                "market data not trading-grade",
                event="MARKET_DATA_STALE",
                symbol=spec.name,
                issues=aggregate.summary()[:500],
            )
        return snapshot
