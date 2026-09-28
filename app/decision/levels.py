"""Deterministic stop-loss / take-profit levels (Phase-3 §11/§12/§13).

SL hierarchy (most reliable first):

1. **structure invalidation** — the last confirmed HL (long) / LH (short)
   from the Phase-2 structure agent; price breaking it disproves the setup;
2. **ATR protective distance** — ``atr_stop_multiple × ATR`` from entry;
3. **broker minimum distance** — ``stops_level_points`` (plus freeze level)
   from the verified Phase-1 symbol metadata; a fallback of last resort,
   never an arbitrary constant.

TP methods (config ``tp_method``, enum ``TPMethod``):

* ``TP_BY_RR`` (``"rr"``, the default — unchanged by the hardening pass):
  TP = ``entry ± target_rr × risk_distance``.  Deterministic and
  geometry-clean, but the resulting RR is derived from the target ratio
  itself, so the engine's ``minimum_rr`` gate is **self-referential** for
  this method: passing it proves valid geometry and broker distances, NOT a
  market-quality RR opportunity.  This is recorded as a note on every such
  proposal and documented in docs/DECISIONS.md §6.
* ``TP_BY_STRUCTURE`` (``"structure"``): TP = the opposing confirmed swing
  level (resistance for longs, support for shorts), derived independently
  of the risk distance.  The actual RR is computed afterwards, and the
  engine's ``minimum_rr`` gate is then a genuine constraint — it HOLDs
  (``insufficient_rr``) when the structure offers less reward than the
  configured minimum.  An opposing level on the wrong side of entry (or
  closer than the broker minimum) is an INVALID target: it is used
  verbatim, flagged with a note, and rejected by the geometry gate — never
  silently clamped or replaced.

Every level is validated against direction, tick size, digits, stops level
and freeze level — reusing the verified Phase-1 ``SymbolSpec`` metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import AgentDirection
from app.core.models import SymbolSpec
from app.decision.config import DecisionEngineConfig, TPMethod


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
    if config.tp_method is TPMethod.TP_BY_STRUCTURE and opposing_structure_level is not None:
        # the structure target is used VERBATIM (tick-snapped only): it is
        # never clamped to the profitable side or padded to the broker
        # minimum — a level that needs padding is not a level the market
        # offered.  Invalid targets (wrong side / too close) are flagged
        # here and rejected by validate_levels at the geometry gate.
        target = opposing_structure_level
        tp_source = "structure_level"
        take_profit = _round_to_tick(target, symbol)
        if direction is AgentDirection.BUY and target <= entry:
            notes.append(
                f"INVALID structure target {target:.2f} not above entry {entry:.2f} for BUY"
            )
        elif direction is AgentDirection.SELL and target >= entry:
            notes.append(
                f"INVALID structure target {target:.2f} not below entry {entry:.2f} for SELL"
            )
        elif abs(target - entry) < minimum:
            notes.append(
                f"structure target {target:.2f} closer than broker minimum {minimum:.2f}"
            )
        else:
            notes.append(f"TP at opposing structure level {opposing_structure_level:.2f}")
    elif (
        config.tp_method is TPMethod.TP_BY_STRUCTURE and opposing_structure_level is None
    ):
        # no opposing level exists: documented fallback to the RR target
        tp_source = "rr_target"
        reward = config.target_rr * sl_distance
        if direction is AgentDirection.BUY:
            take_profit = _round_to_tick(entry + reward, symbol)
        else:
            take_profit = _round_to_tick(entry - reward, symbol)
        notes.append("no opposing structure level — TP falls back to RR target")
    else:
        tp_source = "rr_target"
        reward = config.target_rr * sl_distance
        if direction is AgentDirection.BUY:
            take_profit = _round_to_tick(entry + reward, symbol)
        else:
            take_profit = _round_to_tick(entry - reward, symbol)
        notes.append(f"TP at {config.target_rr:g}x risk (RR target)")
        notes.append(
            "RR gate is self-referential for TP_BY_RR (TP derived from the "
            "target ratio) — it validates geometry, not market-quality RR"
        )

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
