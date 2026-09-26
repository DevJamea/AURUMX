"""Domain enums.

These enums are broker-agnostic: mapping from MT5 integer constants happens only
inside ``app/brokers/mt5.py`` so that the rest of the system never depends on
MetaQuotes constant values.
"""

from __future__ import annotations

from enum import StrEnum


class TimeFrame(StrEnum):
    """Analysis timeframes.  ``minutes`` is the broker-agnostic bar duration."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def minutes(self) -> int:
        return _TF_MINUTES[self]

    @property
    def seconds(self) -> int:
        return self.minutes * 60


_TF_MINUTES = {
    TimeFrame.M1: 1,
    TimeFrame.M5: 5,
    TimeFrame.M15: 15,
    TimeFrame.M30: 30,
    TimeFrame.H1: 60,
    TimeFrame.H4: 240,
    TimeFrame.D1: 1440,
}


class Direction(StrEnum):
    """Market direction an agent or trade refers to."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class DecisionAction(StrEnum):
    """Final decision actions produced by the decision engine."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingMode(StrEnum):
    """Operating mode of the whole system (spec §5, §54).

    READ_ONLY observes, DRY_RUN runs the full decision pipeline but sends no
    orders, PAPER fills against a local simulator, MT5_DEMO/MT5_REAL send real
    orders to a demo/real account.  Only the execution phase (Phase 5) can put
    the system into the last two modes, and real requires an explicit
    confirmation phrase plus runtime account verification.
    """

    READ_ONLY = "READ_ONLY"
    DRY_RUN = "DRY_RUN"
    PAPER = "PAPER"
    MT5_DEMO = "MT5_DEMO"
    MT5_REAL = "MT5_REAL"


class SystemState(StrEnum):
    """Explicit state machine (spec §50) — no random booleans."""

    STARTING = "STARTING"
    CONNECTED = "CONNECTED"
    ANALYZING = "ANALYZING"
    SIGNAL_READY = "SIGNAL_READY"
    RISK_CHECK = "RISK_CHECK"
    EXECUTING = "EXECUTING"
    MANAGING = "MANAGING"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"


class MarketRegime(StrEnum):
    """Market regime detected by the regime engine (spec §18)."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"


class VolatilityLevel(StrEnum):
    """Volatility classification (spec §14)."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class StructureBias(StrEnum):
    """Market-structure bias from swing analysis (HH/HL vs LH/LL)."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class SessionState(StrEnum):
    """Inferred market session state for the watched symbol."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class AccountTradeMode(StrEnum):
    """Account trading mode as reported by the broker."""

    DEMO = "DEMO"
    CONTEST = "CONTEST"
    REAL = "REAL"
    UNKNOWN = "UNKNOWN"


class SymbolTradeMode(StrEnum):
    """Per-symbol trading permission as reported by the broker."""

    DISABLED = "DISABLED"
    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    CLOSE_ONLY = "CLOSE_ONLY"
    FULL = "FULL"
    UNKNOWN = "UNKNOWN"


class OrderType(StrEnum):
    """Supported order kinds (spec §28)."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"
    BUY_STOP_LIMIT = "BUY_STOP_LIMIT"
    SELL_STOP_LIMIT = "SELL_STOP_LIMIT"
    UNKNOWN = "UNKNOWN"


class ValidationLevel(StrEnum):
    """Severity of a data-validation issue.  Errors block trading; warnings don't."""

    ERROR = "ERROR"
    WARNING = "WARNING"


class AgentDirection(StrEnum):
    """Directional lean produced by an analysis agent.

    ``signal_strength`` (0..1) is a *deterministic evidence score*, never a
    probability — it is not calibrated against outcome frequencies.
    """

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class DataQuality(StrEnum):
    """Data-quality status attached to every agent result."""

    OK = "OK"
    DEGRADED = "DEGRADED"          # usable, but some evidence is missing/stale
    INSUFFICIENT = "INSUFFICIENT"  # not enough candles for the agent's logic
    INVALID = "INVALID"            # data unusable (failed validation, etc.)
    NO_DATA = "NO_DATA"            # no source configured (e.g. macro provider)


class SwingKind(StrEnum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingLabel(StrEnum):
    """Structural label of a swing relative to its predecessor."""

    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"
    NONE = "NONE"  # first swing of its kind — no predecessor to compare


class StructureEventType(StrEnum):
    BOS_UP = "BOS_UP"          # break of structure, bullish continuation
    BOS_DOWN = "BOS_DOWN"      # break of structure, bearish continuation
    CHOCH_UP = "CHOCH_UP"      # change of character to bullish
    CHOCH_DOWN = "CHOCH_DOWN"  # change of character to bearish


class VolatilityState(StrEnum):
    EXPANDING = "EXPANDING"
    CONTRACTING = "CONTRACTING"
    STABLE = "STABLE"


class TimeframeRole(StrEnum):
    """Explicit role of a timeframe in the multi-timeframe context."""

    MACRO = "MACRO"        # e.g. H4: primary/macro trend context
    STRUCTURE = "STRUCTURE"  # e.g. H1: market structure & directional context
    ENTRY = "ENTRY"        # e.g. M15: entry timing / local setups
