"""Pure risk-based position sizing (Phase-3 §14).

A deterministic calculator — no broker, no account history, no position
knowledge, and **by construction no martingale / loss-recovery multiplier /
DCA / averaging down**: the function sees only (equity, risk %, entry, SL,
symbol metadata).  Nothing about past losses can change the output because
nothing about past losses is an input.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.core.models import SymbolSpec


class RiskSizing(BaseModel):
    """Full provenance of one sizing calculation."""

    model_config = ConfigDict(allow_inf_nan=False)

    equity: float
    risk_per_trade_pct: float
    risk_amount: float
    risk_distance: float
    #: loss per 1.0 lot if SL is hit (monetary)
    loss_per_lot: float
    raw_volume: float
    suggested_volume: float
    #: volume after step normalization and min/max clamping
    normalized_volume: float
    monetary_risk: float
    percentage_risk: float
    clamped_to_min: bool = False
    clamped_to_max: bool = False
    feasible: bool
    infeasible_reason: str | None = None


def calculate_position_size(
    *,
    equity: float,
    risk_per_trade_pct: float,
    entry: float,
    stop_loss: float,
    symbol: SymbolSpec,
) -> RiskSizing:
    """Risk-based sizing with step normalization.

    ``loss_per_lot = risk_distance / tick_size * tick_value`` — the monetary
    loss of one full lot between entry and SL.  Raw volume is
    ``risk_amount / loss_per_lot``, then floored to the broker's volume step
    (never rounded up: risk must not exceed the budget) and clamped to
    [volume_min, volume_max].
    """
    risk_distance = abs(entry - stop_loss)
    risk_amount = equity * (risk_per_trade_pct / 100.0)

    if risk_distance <= 0:
        return RiskSizing(
            equity=equity, risk_per_trade_pct=risk_per_trade_pct,
            risk_amount=risk_amount, risk_distance=0.0, loss_per_lot=0.0,
            raw_volume=0.0, suggested_volume=0.0, normalized_volume=0.0,
            monetary_risk=0.0, percentage_risk=0.0,
            feasible=False, infeasible_reason="zero_risk_distance",
        )
    if symbol.tick_size <= 0 or symbol.tick_value <= 0 or symbol.volume_step <= 0:
        return RiskSizing(
            equity=equity, risk_per_trade_pct=risk_per_trade_pct,
            risk_amount=risk_amount, risk_distance=risk_distance, loss_per_lot=0.0,
            raw_volume=0.0, suggested_volume=0.0, normalized_volume=0.0,
            monetary_risk=0.0, percentage_risk=0.0,
            feasible=False, infeasible_reason="invalid_symbol_metadata",
        )

    loss_per_lot = (risk_distance / symbol.tick_size) * symbol.tick_value
    raw_volume = risk_amount / loss_per_lot

    # floor to the broker's volume step — never round up (risk cap is hard)
    steps = int(raw_volume / symbol.volume_step)
    normalized = round(steps * symbol.volume_step, 8)

    clamped_to_min = clamped_to_max = False
    if normalized < symbol.volume_min:
        normalized = 0.0  # below minimum is infeasible, not forced
        clamped_to_min = True
    elif normalized > symbol.volume_max:
        normalized = symbol.volume_max
        clamped_to_max = True

    feasible = normalized > 0 and not clamped_to_min
    monetary_risk = (normalized * loss_per_lot) if normalized else 0.0
    percentage_risk = (monetary_risk / equity * 100.0) if equity > 0 else 0.0

    return RiskSizing(
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
        risk_amount=round(risk_amount, 2),
        risk_distance=round(risk_distance, 10),
        loss_per_lot=round(loss_per_lot, 4),
        raw_volume=round(raw_volume, 8),
        suggested_volume=round(normalized, 8),
        normalized_volume=round(normalized, 8),
        monetary_risk=round(monetary_risk, 2),
        percentage_risk=round(percentage_risk, 6),
        clamped_to_min=clamped_to_min,
        clamped_to_max=clamped_to_max,
        feasible=feasible,
        infeasible_reason=None if feasible else "volume_below_minimum",
    )
