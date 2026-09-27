"""Decision-engine configuration — the single home of every threshold.

Nothing in the engine hardcodes a magic number: all tunables live here, are
bounded, documented, and carry the same warning: **these are initial
engineering parameters, not statistically optimal values.**  They must be
validated against historical data (walk-forward, Phase 6) before any live
use.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import TimeFrame


class TimeframeWeights(BaseModel):
    """Contribution of each timeframe to alignment (need not sum to 1 —
    scores are renormalized over the timeframes actually present)."""

    model_config = ConfigDict(extra="forbid")

    h4: float = Field(default=0.30, gt=0, le=1.0)
    h1: float = Field(default=0.45, gt=0, le=1.0)
    m15: float = Field(default=0.25, gt=0, le=1.0)

    @property
    def total(self) -> float:
        """Sum of all weights (alignment renormalizes over present TFs)."""
        return self.h4 + self.h1 + self.m15

    def weight_of(self, timeframe: TimeFrame) -> float:
        return {
            TimeFrame.D1: self.h4,
            TimeFrame.H4: self.h4,
            TimeFrame.H1: self.h1,
            TimeFrame.M30: self.h1,
            TimeFrame.M15: self.m15,
            TimeFrame.M5: self.m15,
            TimeFrame.M1: self.m15,
        }.get(timeframe, 0.0)


class DecisionEngineConfig(BaseModel):
    """All decision-engine thresholds.

    Initial engineering parameters — REQUIRE HISTORICAL VALIDATION (spec
    §61; see docs/DECISIONS.md).  Defaults are deliberately conservative:
    the engine would rather HOLD than trade a marginal setup.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- edge thresholds ------------------------------------------------
    #: minimum relevance-weighted side score for a BUY (from synthesis)
    buy_threshold: float = Field(default=1.00, gt=0, le=10.0)
    #: minimum relevance-weighted side score for a SELL
    sell_threshold: float = Field(default=1.00, gt=0, le=10.0)
    #: minimum net (side_score − opposing_score) required
    net_threshold: float = Field(default=0.80, gt=0, le=10.0)
    #: the strongest single supporting agent must reach this strength
    minimum_signal_strength: float = Field(default=0.45, gt=0, le=1.0)
    #: weighted multi-timeframe agreement required for a proposal
    minimum_timeframe_alignment: float = Field(default=0.80, gt=0, le=1.0)
    #: maximum opposing share of weighted evidence (conflict tolerance)
    max_conflict: float = Field(default=0.35, ge=0, le=1.0)
    #: minimum reward:risk for a proposal to be actionable
    minimum_rr: float = Field(default=1.50, gt=0, le=20.0)

    # ---- timeframe alignment --------------------------------------------
    timeframe_weights: TimeframeWeights = Field(default_factory=TimeframeWeights)
    #: |EMA20-EMA50| in ATR beyond which a timeframe read counts as "strong"
    strong_alignment_atr: float = Field(default=0.50, gt=0, le=10.0)

    # ---- stop loss / take profit ----------------------------------------
    #: SL hierarchy: 1) structure invalidation level, 2) ATR distance,
    #: 3) broker minimum distance (fallback of last resort, documented)
    atr_stop_multiple: float = Field(default=2.00, gt=0, le=20.0)
    #: TP method: "rr" (risk/reward target) or "structure" (opposing level)
    tp_method: str = Field(default="rr", pattern="^(rr|structure)$")
    #: reward multiple of risk distance used for TP when tp_method == "rr"
    target_rr: float = Field(default=2.00, gt=0, le=20.0)

    # ---- risk limits (checked before any proposal is built) --------------
    #: max acceptable quoted spread in points; None disables the gate
    max_spread_points: float | None = Field(default=50.0, gt=0)
    #: fraction of equity risked per trade (conservative default)
    risk_per_trade_pct: float = Field(default=0.50, gt=0, le=10.0)
    max_open_positions: int = Field(default=1, ge=1, le=20)
    max_pending_orders: int = Field(default=2, ge=0, le=50)
    #: consecutive losses tolerated before the engine stands down
    max_consecutive_losses: int = Field(default=3, ge=1, le=50)
    #: daily loss limit as a fraction of equity (used when the supplied
    #: RiskState does not carry an explicit monetary limit)
    daily_loss_limit_pct: float = Field(default=2.00, gt=0, le=50.0)

    # ---- proposal lifecycle ----------------------------------------------
    #: proposal validity window (minutes) — a proposal never lives forever
    proposal_ttl_minutes: int = Field(default=15, ge=1, le=1440)

    @classmethod
    def from_app_config(cls, config) -> DecisionEngineConfig:
        """Build from the runtime AppConfig (risk section reuses Phase-1
        settings so operators configure one place)."""
        return cls(
            max_spread_points=config.max_spread_points,
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_open_positions=config.max_open_positions,
            max_pending_orders=config.max_pending_orders,
            max_consecutive_losses=config.max_consecutive_losses
            if hasattr(config, "max_consecutive_losses")
            else 3,
            daily_loss_limit_pct=config.max_daily_loss_pct,
            minimum_rr=config.min_reward_risk,
        )
