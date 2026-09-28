"""TradeProposal — a pure data object (Phase-3 §9/§10/§17/§18).

A proposal **is not an order** and never touches MT5.  It carries everything
a (future) execution layer or a human needs to evaluate the trade: geometry,
sizing, provenance, invalidation conditions and an expiry.

Entry prices: the live path uses the *validated tick* — ask for BUY, bid for
SELL (never a historical candle close).  Backtesting uses a separate
historical entry adapter (``EntryPriceProvider``); the distinction is part of
the contract, see docs/DECISIONS.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AgentDirection, MarketRegime, TimeFrame
from app.core.models import SymbolSpec
from app.decision.config import DecisionEngineConfig
from app.decision.levels import LevelPlan
from app.risk.sizing import RiskSizing


class EntryPriceProvider(Protocol):
    """Where the entry price comes from.

    ``TickEntryProvider`` (live/paper): the validated snapshot tick.
    Phase 6 adds a historical bar-based provider for backtests — decision
    logic never assumes which one is in use.
    """

    def entry_price(self, direction: AgentDirection, snapshot) -> float | None: ...


class TickEntryProvider:
    """Live/paper entry: BUY at the validated ask, SELL at the validated bid."""

    def entry_price(self, direction: AgentDirection, snapshot) -> float | None:
        if direction is AgentDirection.BUY:
            return snapshot.ask
        if direction is AgentDirection.SELL:
            return snapshot.bid
        return None


class TradeProposal(BaseModel):
    """Everything about one proposed trade — and nothing that executes it."""

    model_config = ConfigDict(allow_inf_nan=False)

    decision_id: str
    fingerprint: str
    setup_type: str
    symbol: str
    direction: AgentDirection
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_reward: float = Field(ge=0)
    risk_distance: float = Field(ge=0)
    reward_distance: float = Field(ge=0)
    suggested_volume: float = Field(ge=0)
    max_allowed_volume: float = Field(ge=0)
    sizing: RiskSizing
    sl_source: str
    tp_source: str
    timeframe: TimeFrame
    regime: MarketRegime
    reasons: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime

    @classmethod
    def build(
        cls,
        *,
        decision_id: str,
        fingerprint: str,
        setup_type: str,
        direction: AgentDirection,
        entry: float,
        levels: LevelPlan,
        sizing: RiskSizing,
        symbol: SymbolSpec,
        timeframe: TimeFrame,
        regime: MarketRegime,
        reasons: list[str],
        invalidation_conditions: list[str],
        created_at: datetime,
        config: DecisionEngineConfig,
    ) -> TradeProposal:
        return cls(
            decision_id=decision_id,
            fingerprint=fingerprint,
            setup_type=setup_type,
            symbol=symbol.name,
            direction=direction,
            entry_price=round(entry, symbol.digits),
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            risk_reward=round(levels.reward_risk, 4),
            risk_distance=round(levels.risk_distance, 10),
            reward_distance=round(levels.reward_distance, 10),
            suggested_volume=sizing.suggested_volume,
            max_allowed_volume=symbol.volume_max,
            sizing=sizing,
            sl_source=levels.sl_source,
            tp_source=levels.tp_source,
            timeframe=timeframe,
            regime=regime,
            reasons=list(reasons),
            invalidation_conditions=list(invalidation_conditions),
            created_at=created_at,
            expires_at=created_at + timedelta(minutes=config.proposal_ttl_minutes),
        )

    # ------------------------------------------------------------------
    def status(self, at: datetime) -> str:
        """VALID / EXPIRED — a pure function of the given time, no globals."""
        return "VALID" if at <= self.expires_at else "EXPIRED"

    def is_expired(self, at: datetime) -> bool:
        return self.status(at) == "EXPIRED"

    def geometry_valid(self, symbol: SymbolSpec) -> bool:
        """BUY: SL < entry < TP; SELL mirrored; broker distances respected."""
        from app.decision.levels import validate_levels

        return not validate_levels(
            direction=self.direction,
            entry=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            symbol=symbol,
        )

    def summary(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "fingerprint": self.fingerprint,
            "setup_type": self.setup_type,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry": self.entry_price,
            "sl": self.stop_loss,
            "tp": self.take_profit,
            "rr": self.risk_reward,
            "volume": self.suggested_volume,
            "risk": self.sizing.monetary_risk,
            "expires_at": self.expires_at.isoformat(),
        }


#: canonical invalidation conditions attached to every proposal (Phase-3 §17)
DEFAULT_INVALIDATION_CONDITIONS = [
    "structure invalidated: price closes beyond the setup's invalidation level",
    "regime changed away from the setup's regime",
    "spread exceeded the configured maximum",
    "proposal expired (expires_at passed)",
    "data quality degraded below trading grade",
    "risk limits violated (daily loss / position count)",
]


def invalidation_conditions(
    levels: LevelPlan, regime: MarketRegime, config: DecisionEngineConfig
) -> list[str]:
    conditions = list(DEFAULT_INVALIDATION_CONDITIONS)
    conditions.append(f"SL distance invalid (< broker minimum or > {config.atr_stop_multiple:g}x ATR fallback)")
    if levels.sl_structure_level is not None:
        conditions.append(
            f"setup structure level {levels.sl_structure_level:.2f} broken before entry"
        )
    conditions.append(f"regime no longer {regime.value}")
    return conditions
