"""Deterministic stop-loss / take-profit levels (Phase-3 §11/§12/§13).

SL hierarchy (most reliable first):

1. **structure invalidation** — the last confirmed HL (long) / LH (short)
   from the Phase-2 structure agent; price breaking it disproves the setup;
2. **ATR protective distance** — ``atr_stop_multiple × ATR`` from entry;
3. **broker minimum distance** — ``stops_level_points`` (plus freeze level)
   from the verified Phase-1 symbol metadata; a fallback of last resort,
   never an arbitrary constant.

TP methods (config ``tp_method``):

* ``rr`` (default): ``entry ± target_rr × risk_distance`` — a documented
  risk/reward target;
* ``structure``: the nearest opposing confirmed swing level (resistance for
  longs, support for shorts), floored to at least ``minimum_rr`` by the
  engine's RR gate (the gate HOLDs rather than silently widening a target).

Every level is validated against direction, tick size, digits, stops level
and freeze level — reusing the verified Phase-1 ``SymbolSpec`` metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import AgentDirection
from app.core.models import SymbolSpec
from app.decision.config import DecisionEngineConfig


@dataclass(frozen=True)
class LevelPlan:
    """Computed SL/TP with full provenance of which rule produced them."""

    stop_loss: float
    take_profit: float
    sl_source: str  # structure | atr | broker_minimum
    tp_source: str  # rr_target | structure_level
    sl_structure_level: float | None = None
    tp_structure_level: float | None = None
    risk_distance: float = 0.0
    reward_distance: float = 0.0
    reward_risk: float = 0.0
    notes: tuple[str, ...] = ()


def _round_to_tick(price: float, symbol: SymbolSpec) -> float:
    """Snap to the broker's tick grid (and digits)."""
    if symbol.tick_size <= 0:
        return round(price, symbol.digits)
    ticks = round(price / symbol.tick_size)
    return round(ticks * symbol.tick_size, symbol.digits)


def min_stop_distance(symbol: SymbolSpec) -> float:
    """Minimum allowed distance between entry and SL/TP in price units."""
    points = max(symbol.stops_level_points, symbol.freeze_level_points)
    return points * symbol.point if symbol.point > 0 else 0.0


def validate_levels(
    *, direction: AgentDirection, entry: float, stop_loss: float, take_profit: float,
    symbol: SymbolSpec,
) -> list[str]:
    """Directional + broker-constraint validation.  Returns violation strings
    (empty = valid).  BUY: SL < entry < TP; SELL: TP < entry < SL."""
    issues: list[str] = []
    minimum = min_stop_distance(symbol)
    if direction is AgentDirection.BUY:
        if not stop_loss < entry:
            issues.append("SL must be below entry for BUY")
        if not take_profit > entry:
            issues.append("TP must be above entry for BUY")
    elif direction is AgentDirection.SELL:
        if not stop_loss > entry:
            issues.append("SL must be above entry for SELL")
        if not take_profit < entry:
            issues.append("TP must be below entry for SELL")
    if abs(entry - stop_loss) < minimum:
        issues.append(f"SL distance {abs(entry - stop_loss):.2f} below broker minimum {minimum:.2f}")
    if abs(take_profit - entry) < minimum:
        issues.append(f"TP distance {abs(take_profit - entry):.2f} below broker minimum {minimum:.2f}")
    if symbol.tick_size > 0:
        for label, price in (("SL", stop_loss), ("TP", take_profit)):
            if abs(price / symbol.tick_size - round(price / symbol.tick_size)) > 1e-9:
                issues.append(f"{label} not aligned to tick size {symbol.tick_size}")
    return issues


def compute_levels(
    *,
    direction: AgentDirection,
    entry: float,
    atr: float | None,
    structure_invalidation: float | None,
    opposing_structure_level: float | None,
    symbol: SymbolSpec,
    config: DecisionEngineConfig,
) -> LevelPlan:
    """Deterministic SL/TP plan (see module docstring for the hierarchy).

    Unrounded distances feed the RR calculation; only the final price levels
    are snapped to the tick grid.
    """
    minimum = min_stop_distance(symbol)
    notes: list[str] = []

    # ---- stop loss -------------------------------------------------------
    sl_source = "atr"
    sl_distance = config.atr_stop_multiple * atr if atr else None

    if structure_invalidation is not None:
        structural_distance = abs(entry - structure_invalidation)
        if structural_distance >= minimum and (
            sl_distance is None or structural_distance <= sl_distance
        ):
            # structure level is valid AND tighter-or-equal than the ATR stop
            sl_distance = structural_distance
            sl_source = "structure"
            notes.append(f"SL at structure invalidation {structure_invalidation:.2f}")
        elif structural_distance < minimum:
            notes.append(
                f"structure level {structure_invalidation:.2f} too close to entry "
                f"(< {minimum:.2f}) — ATR stop used"
            )
        else:
            notes.append("structure stop wider than ATR stop — ATR stop used")
    elif sl_distance is not None:
        notes.append(f"SL at {config.atr_stop_multiple:g}x ATR")

    if sl_distance is None or sl_distance < minimum:
        sl_distance = max(sl_distance or 0.0, minimum)
        sl_source = "broker_minimum"
        notes.append("SL at broker minimum distance (no structure level, no ATR)")

    if direction is AgentDirection.BUY:
        stop_loss = _round_to_tick(entry - sl_distance, symbol)
    else:
        stop_loss = _round_to_tick(entry + sl_distance, symbol)

    # ---- take profit -------------------------------------------------------
    if config.tp_method == "structure" and opposing_structure_level is not None:
        target = opposing_structure_level
        # keep the structure target on the profitable side, at least the
        # broker minimum away
        if direction is AgentDirection.BUY:
            target = max(target, entry + minimum)
        else:
            target = min(target, entry - minimum)
        tp_source = "structure_level"
        take_profit = _round_to_tick(target, symbol)
        notes.append(f"TP at opposing structure level {opposing_structure_level:.2f}")
    else:
        tp_source = "rr_target"
        reward = config.target_rr * sl_distance
        if direction is AgentDirection.BUY:
            take_profit = _round_to_tick(entry + reward, symbol)
        else:
            take_profit = _round_to_tick(entry - reward, symbol)
        notes.append(f"TP at {config.target_rr:g}x risk (RR target)")

    # ---- distances / RR (unrounded inputs, one final rounding) ------------
    risk_distance = abs(entry - stop_loss)
    reward_distance = abs(take_profit - entry)
    reward_risk = reward_distance / risk_distance if risk_distance > 0 else 0.0

    return LevelPlan(
        stop_loss=stop_loss,
        take_profit=take_profit,
        sl_source=sl_source,
        tp_source=tp_source,
        sl_structure_level=structure_invalidation,
        tp_structure_level=opposing_structure_level if tp_source == "structure_level" else None,
        risk_distance=risk_distance,
        reward_distance=reward_distance,
        reward_risk=reward_risk,
        notes=tuple(notes),
    )
