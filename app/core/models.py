"""Broker-agnostic domain models (spec §7, §36).

Design rules:

* Models **carry** data; validation **reports** live in ``app/market`` validators
  and return ``ValidationReport`` objects instead of raising.  Data quality is a
  first-class output the health monitor and risk gate consume.
* All timestamps are timezone-aware UTC datetimes.  Naive datetimes are assumed
  UTC (MT5 returns epoch-based server times; the market-data service corrects
  for the server clock offset before validation).
* Numbers reject NaN/inf: a candle with ``nan`` close is a data error, not a
  value to propagate into agents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.enums import (
    AccountTradeMode,
    Direction,
    OrderType,
    SymbolTradeMode,
    TimeFrame,
    ValidationLevel,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def ensure_utc(value: datetime) -> datetime:
    """Coerce naive datetimes to UTC (assumed UTC), leave aware ones untouched."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


# ==========================================================================
# Validation reports
# ==========================================================================
class ValidationIssue(BaseModel):
    """One data-quality finding.  ``code`` is a stable SCREAMING_SNAKE string."""

    level: ValidationLevel
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """A collection of issues.  ``ok`` means: no ERROR-level issues."""

    issues: list[ValidationIssue] = Field(default_factory=list)

    def error(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(
            ValidationIssue(level=ValidationLevel.ERROR, code=code, message=message, context=context)
        )

    def warning(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(
            ValidationIssue(level=ValidationLevel.WARNING, code=code, message=message, context=context)
        )

    @property
    def ok(self) -> bool:
        return not any(issue.level is ValidationLevel.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.level is ValidationLevel.WARNING for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level is ValidationLevel.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level is ValidationLevel.WARNING]

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)

    def summary(self) -> str:
        if not self.issues:
            return "OK"
        parts = [f"{i.level.value} {i.code}: {i.message}" for i in self.issues]
        return "; ".join(parts)


# ==========================================================================
# Market data
# ==========================================================================
class _FiniteModel(BaseModel):
    """Base config: reject NaN/inf floats everywhere."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class MarketTick(_FiniteModel):
    """A raw quote.  Sanity (bid>0, ask>=bid, freshness) is judged by the
    ``TickValidator``, not by the model — the model records what the broker sent."""

    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: float | None = None
    source_time_ms: int | None = None

    _ensure_utc = field_validator("time")(lambda cls, v: ensure_utc(v))

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2.0


class TickCheck(_FiniteModel):
    """Result of validating a tick (spec §53 edge cases)."""

    tick: MarketTick | None = None
    valid: bool = False
    fresh: bool = False
    report: ValidationReport = Field(default_factory=ValidationReport)
    spread: float | None = None
    spread_points: float | None = None
    age_seconds: float | None = None


class Candle(_FiniteModel):
    """One OHLC bar.  A forming (in-progress) bar is only ever present when
    ``CandleSeries.include_forming`` is true; analysis uses closed bars only."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0
    real_volume: float = 0.0
    spread_points: int | None = None

    _ensure_utc = field_validator("time")(lambda cls, v: ensure_utc(v))


class CandleSeries(_FiniteModel):
    """Candles for one symbol/timeframe, ordered oldest -> newest."""

    symbol: str
    timeframe: TimeFrame
    candles: list[Candle] = Field(default_factory=list)
    include_forming: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.candles

    @property
    def last(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    @property
    def last_closed(self) -> Candle | None:
        """The most recent *closed* candle (forming bar excluded)."""
        if not self.candles:
            return None
        if self.include_forming:
            return self.candles[-2] if len(self.candles) >= 2 else None
        return self.candles[-1]

    def to_dataframe(self) -> pd.DataFrame:
        """Candles as a pandas DataFrame (columns: time, open, high, low, close,
        tick_volume, real_volume, spread_points)."""
        import pandas as pd

        rows = [
            {
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "tick_volume": c.tick_volume,
                "real_volume": c.real_volume,
                "spread_points": c.spread_points,
            }
            for c in self.candles
        ]
        return pd.DataFrame(rows)


class SeriesCheck(_FiniteModel):
    """Result of validating one timeframe's candle series."""

    timeframe: TimeFrame
    series: CandleSeries | None = None
    valid: bool = False
    fresh: bool = False
    report: ValidationReport = Field(default_factory=ValidationReport)
    age_seconds: float | None = None


# ==========================================================================
# Broker metadata
# ==========================================================================
class SymbolSpec(_FiniteModel):
    """Verified broker metadata for one symbol (spec §7 verification list)."""

    name: str
    description: str = ""
    visible: bool = False
    selected: bool = False
    trade_mode: SymbolTradeMode = SymbolTradeMode.UNKNOWN
    digits: int = 0
    point: float = 0.0
    tick_size: float = 0.0
    tick_value: float = 0.0
    contract_size: float = 0.0
    volume_min: float = 0.0
    volume_max: float = 0.0
    volume_step: float = 0.0
    stops_level_points: int = 0
    freeze_level_points: int = 0
    currency_profit: str = ""
    currency_margin: str = ""

    @property
    def is_gold_name(self) -> bool:
        from app.market.symbol_discovery import is_gold_symbol

        return is_gold_symbol(self.name)

    def validate(self) -> ValidationReport:
        """Broker-metadata verification.  Errors => symbol unusable."""
        report = ValidationReport()
        if self.point <= 0:
            report.error("POINT_NOT_POSITIVE", f"point={self.point}", symbol=self.name)
        if self.digits < 1:
            report.error("DIGITS_INVALID", f"digits={self.digits}", symbol=self.name)
        elif self.digits > 4:
            report.warning("DIGITS_UNUSUAL", f"digits={self.digits} (gold is typically 2-3)", symbol=self.name)
        if self.tick_size <= 0:
            report.error("TICK_SIZE_NOT_POSITIVE", f"trade_tick_size={self.tick_size}", symbol=self.name)
        if self.tick_value <= 0:
            report.error("TICK_VALUE_NOT_POSITIVE", f"tick_value={self.tick_value}", symbol=self.name)
        if self.contract_size <= 0:
            report.error("CONTRACT_SIZE_NOT_POSITIVE", f"contract_size={self.contract_size}", symbol=self.name)
        if self.volume_min <= 0:
            report.error("VOLUME_MIN_NOT_POSITIVE", f"volume_min={self.volume_min}", symbol=self.name)
        if self.volume_max < self.volume_min:
            report.error(
                "VOLUME_RANGE_INVALID",
                f"volume_max={self.volume_max} < volume_min={self.volume_min}",
                symbol=self.name,
            )
        if self.volume_step <= 0:
            report.error("VOLUME_STEP_NOT_POSITIVE", f"volume_step={self.volume_step}", symbol=self.name)
        if self.stops_level_points < 0:
            report.error("STOPS_LEVEL_NEGATIVE", f"stops_level={self.stops_level_points}", symbol=self.name)
        if self.freeze_level_points < 0:
            report.error("FREEZE_LEVEL_NEGATIVE", f"freeze_level={self.freeze_level_points}", symbol=self.name)
        if self.trade_mode is SymbolTradeMode.DISABLED:
            report.error("TRADING_DISABLED", "symbol trade mode is DISABLED", symbol=self.name)
        elif self.trade_mode in (SymbolTradeMode.UNKNOWN,):
            report.error("TRADE_MODE_UNKNOWN", "symbol trade mode could not be determined", symbol=self.name)
        if not self.visible:
            report.error("NOT_VISIBLE", "symbol is not visible in Market Watch", symbol=self.name)
        if self.currency_profit and self.currency_profit != "USD":
            report.warning(
                "NON_USD_PROFIT_CURRENCY",
                f"profit currency is {self.currency_profit}, risk math assumes USD quote",
                symbol=self.name,
            )
        return report

    def normalize_volume(self, volume: float) -> float:
        """Clamp to [volume_min, volume_max] and round down to volume_step.

        Result is rounded to 8 decimals to avoid float artifacts (e.g.
        0.01 + 9*0.01 == 0.09999999999999999).
        """
        if self.volume_step <= 0:
            return volume
        steps = int((volume - self.volume_min) / self.volume_step + 1e-9)
        normalized = self.volume_min + steps * self.volume_step
        return round(max(self.volume_min, min(normalized, self.volume_max)), 8)


class AccountSnapshot(_FiniteModel):
    """Read-only account state.  Never contains credentials."""

    login: int
    trade_mode: AccountTradeMode = AccountTradeMode.UNKNOWN
    server: str = ""
    currency: str = "USD"
    leverage: int = 100
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    margin_free: float = 0.0
    margin_level: float | None = None
    name: str = ""

    @property
    def is_demo(self) -> bool:
        return self.trade_mode is AccountTradeMode.DEMO

    @property
    def is_real(self) -> bool:
        return self.trade_mode is AccountTradeMode.REAL


class Position(_FiniteModel):
    """An open position as reported by the broker."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    price_open: float
    price_current: float | None = None
    price_sl: float | None = None
    price_tp: float | None = None
    profit: float = 0.0
    swap: float = 0.0
    time: datetime
    comment: str = ""
    magic: int = 0

    _ensure_utc = field_validator("time")(lambda cls, v: ensure_utc(v))

    @property
    def has_sl(self) -> bool:
        return self.price_sl is not None and self.price_sl > 0


class PendingOrder(_FiniteModel):
    """A pending order as reported by the broker."""

    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price_open: float
    price_sl: float | None = None
    price_tp: float | None = None
    time_setup: datetime
    time_expiration: datetime | None = None
    comment: str = ""
    magic: int = 0

    _ensure_utc_setup = field_validator("time_setup")(lambda cls, v: ensure_utc(v))
    _ensure_utc_exp = field_validator("time_expiration")(lambda cls, v: ensure_utc(v) if v else v)


# ==========================================================================
# Execution contracts (used from Phase 5; defined now to fix the interface)
# ==========================================================================
class MarketOrderRequest(_FiniteModel):
    """Request for an immediate-fill order.  The risk gate must approve a
    request before any broker is allowed to see it (enforced in Phase 5)."""

    symbol: str
    direction: Direction
    volume: float = Field(gt=0)
    sl: float | None = None
    tp: float | None = None
    deviation_points: int = 20
    comment: str = ""
    magic: int = 0


class PendingOrderRequest(_FiniteModel):
    """Request for a pending order (spec §28)."""

    symbol: str
    order_type: OrderType
    volume: float = Field(gt=0)
    price_open: float = Field(gt=0)
    sl: float | None = None
    tp: float | None = None
    expiration: datetime | None = None
    comment: str = ""
    magic: int = 0


class OrderResult(_FiniteModel):
    """Normalized broker response.  ``accepted`` is true only when the broker
    confirmed execution (e.g. ``TRADE_RETCODE_DONE``); a sent request is never
    assumed successful without verification (spec §30)."""

    accepted: bool = False
    retcode: int | None = None
    retcode_description: str = ""
    ticket: int | None = None
    deal_ticket: int | None = None
    price: float | None = None
    volume: float | None = None
    message: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AccountSnapshot",
    "Candle",
    "CandleSeries",
    "MarketOrderRequest",
    "MarketTick",
    "OrderResult",
    "PendingOrder",
    "PendingOrderRequest",
    "Position",
    "SeriesCheck",
    "SymbolSpec",
    "TickCheck",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationReport",
    "ensure_utc",
]
