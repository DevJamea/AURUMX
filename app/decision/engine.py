"""DecisionEngine — deterministic BUY / SELL / HOLD / ABORT (Phase-3 §2).

Pipeline position (spec §2): everything upstream is read-only analysis; the
engine consumes it and *proposes*.  It never executes: there is no MT5, no
orders, no broker anywhere in this module — the only outputs are a
``Decision`` and an optional pure ``TradeProposal``.

Decision semantics (the distinction is the point):

* **BUY / SELL** — valid market, sufficient edge, valid geometry, risk OK.
* **HOLD** — *market* answer: valid data, but no sufficient edge, timeframe
  conflict, excessive disagreement, insufficient RR, duplicate setup …
* **ABORT** — *safety* answer: invalid tick, stale data, unknown session,
  spread too high, risk/limit violations.  Never a trading signal.

Gate order is fixed and documented (fail-fast, first failure decides):

1. data validity (symbol, tick, series)        → ABORT
2. market session (CLOSED → HOLD, UNKNOWN → ABORT)
3. freshness & validation report               → ABORT
4. spread                                       → ABORT
5. risk state (equity, daily loss, streak,
   open positions, pending orders)             → ABORT
6. analysis (closed-candles-only slicing →
   context → regime → agents → synthesis)
7. edge thresholds, timeframe alignment,
   conflict tolerance                          → HOLD
8. duplicate-setup fingerprint                 → HOLD
9. SL/TP levels, RR, position sizing,
   geometry validation                         → HOLD (fail-safe)

No look-ahead: step 6 slices every series to candles **closed by ``now``**,
so future candles physically cannot influence a historical decision.
Determinism: ``now`` is an explicit input (never ``datetime.now()``); the
same snapshot + risk state + config always produce the same decision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

from app.agents import build_market_context, default_agents
from app.agents.base import AgentResult, BaseAgent
from app.agents.registry import AgentRegistry
from app.core.enums import (
    AgentDirection,
    DataQuality,
    DecisionAction,
    MarketRegime,
    SessionState,
    TimeFrame,
    VolatilityLevel,
)
from app.core.models import CandleSeries, SymbolSpec
from app.decision.alignment import compute_alignment
from app.decision.config import DecisionEngineConfig
from app.decision.levels import compute_levels, validate_levels
from app.decision.proposal import (
    EntryPriceProvider,
    TickEntryProvider,
    TradeProposal,
    invalidation_conditions,
)
from app.decision.regime import RegimeAssessment, RegimeDetector
from app.decision.synthesis import SynthesisInput, SynthesisOutput, synthesize
from app.market.market_state import MarketSnapshot
from app.risk.sizing import calculate_position_size
from app.risk.state import RiskState


# ==========================================================================
# results
# ==========================================================================
@dataclass(frozen=True)
class GateResult:
    """One gate's outcome — full provenance for the journal."""

    gate: str
    passed: bool
    action: DecisionAction  # the action the gate enforces on failure
    reason: str | None = None


class Decision(BaseModel):
    """The engine's complete, journal-ready output."""

    model_config = ConfigDict(allow_inf_nan=False)

    decision_id: str
    decision: DecisionAction
    symbol: str
    timestamp: datetime  # decision time (explicit input, not wall clock)
    source_candle_time: datetime | None = None
    regime: MarketRegime | None = None
    volatility: VolatilityLevel | None = None
    session_state: SessionState | None = None
    spread_points: float | None = None
    conflict_score: float = 0.0
    alignment_score: float | None = None
    supporting_agents: list[str] = []
    opposing_agents: list[str] = []
    reasons: list[str] = []
    warnings: list[str] = []
    rejection_reasons: list[str] = []
    gates: list[dict] = []
    data_quality: DataQuality = DataQuality.OK
    fingerprint: str | None = None
    setup_type: str | None = None
    proposal: TradeProposal | None = None
    snapshot_created_at: datetime | None = None
    #: full evidence (JSON-safe) so a journal record is self-contained
    agent_results: list[dict] = []
    synthesis_payload: dict | None = None

    @property
    def is_actionable(self) -> bool:
        return self.decision in (DecisionAction.BUY, DecisionAction.SELL) and self.proposal is not None


# ==========================================================================
# engine
# ==========================================================================
class DecisionEngine:
    """Deterministic, configurable decision evaluation.

    The engine owns the full analysis pipeline from the snapshot down (it
    builds the context, detects the regime, runs the agents and synthesizes)
    so the no-look-ahead slicing in step 6 cannot be bypassed.
    """

    def __init__(
        self,
        config: DecisionEngineConfig | None = None,
        *,
        agents: list[BaseAgent] | None = None,
        regime_detector: RegimeDetector | None = None,
        entry_provider: EntryPriceProvider | None = None,
        journal=None,
    ) -> None:
        self.config = config or DecisionEngineConfig()
        # canonical ordering: the roster order can never leak into decisions
        roster = agents if agents is not None else default_agents()
        self._registry = AgentRegistry(agents=sorted(roster, key=lambda a: a.name))
        self._detector = regime_detector or RegimeDetector()
        self._entry_provider = entry_provider or TickEntryProvider()
        self._journal = journal

    # ------------------------------------------------------------------
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        *,
        risk_state: RiskState | None = None,
        now: datetime,
    ) -> Decision:
        """Evaluate one decision cycle.  Pure: same inputs → same decision."""
        state = risk_state or RiskState()
        gates: list[GateResult] = []
        decision = self._run_gates(snapshot, state, now, gates)
        if self._journal is not None:
            from app.decision.journal import DecisionRecord

            self._journal.record(
                DecisionRecord.from_decision(
                    decision,
                    agent_results=decision.agent_results,
                    synthesis=decision.synthesis_payload,
                    config_snapshot=self.config.model_dump(),
                    snapshot_ref=snapshot.summary(),
                )
            )
        return decision

    # ------------------------------------------------------------------
    # gate pipeline
    # ------------------------------------------------------------------
    def _run_gates(
        self,
        snapshot: MarketSnapshot,
        state: RiskState,
        now: datetime,
        gates: list[GateResult],
    ) -> Decision:
        def _finish(action, **kwargs) -> Decision:
            return self._make_decision(snapshot, now, action, gates=gates, **kwargs)

        # ---- gate 1: data validity -------------------------------------
        if snapshot.symbol is None:
            gates.append(GateResult("data_validity", False, DecisionAction.ABORT, "missing_symbol"))
            return _finish(DecisionAction.ABORT, reason="missing_symbol", data_quality=DataQuality.INVALID)
        if snapshot.tick.tick is None or not snapshot.tick.valid:
            gates.append(GateResult("data_validity", False, DecisionAction.ABORT, "invalid_tick"))
            return _finish(DecisionAction.ABORT, reason="invalid_tick", data_quality=DataQuality.INVALID)
        if not snapshot.series:
            gates.append(GateResult("data_validity", False, DecisionAction.ABORT, "no_candle_data"))
            return _finish(DecisionAction.ABORT, reason="no_candle_data", data_quality=DataQuality.INVALID)
        invalid_series = [tf.value for tf, c in snapshot.series.items() if not c.valid]
        if invalid_series:
            code = "invalid_series:" + ",".join(invalid_series)
            gates.append(GateResult("data_validity", False, DecisionAction.ABORT, code))
            return _finish(DecisionAction.ABORT, reason=code, data_quality=DataQuality.INVALID)
        if TimeFrame.H1 not in snapshot.series or snapshot.series[TimeFrame.H1].series is None:
            gates.append(
                GateResult("data_validity", False, DecisionAction.ABORT, "missing_primary_timeframe_h1")
            )
            return _finish(
                DecisionAction.ABORT, reason="missing_primary_timeframe_h1",
                data_quality=DataQuality.INVALID,
            )
        gates.append(GateResult("data_validity", True, DecisionAction.HOLD))

        # ---- gate 2: session --------------------------------------------
        session = snapshot.session_state
        if session is SessionState.CLOSED:
            gates.append(GateResult("session", False, DecisionAction.HOLD, "market_closed"))
            return _finish(DecisionAction.HOLD, reason="market_closed")
        if session is SessionState.UNKNOWN:
            gates.append(GateResult("session", False, DecisionAction.ABORT, "session_unknown"))
            return _finish(DecisionAction.ABORT, reason="session_unknown", data_quality=DataQuality.DEGRADED)
        gates.append(GateResult("session", True, DecisionAction.HOLD))

        # ---- gate 3: freshness & report ---------------------------------
        if not snapshot.tick.fresh:
            gates.append(GateResult("freshness", False, DecisionAction.ABORT, "stale_tick"))
            return _finish(DecisionAction.ABORT, reason="stale_tick", data_quality=DataQuality.DEGRADED)
        stale = [tf.value for tf, c in snapshot.series.items() if not c.fresh]
        if stale:
            code = "stale_candles:" + ",".join(stale)
            gates.append(GateResult("freshness", False, DecisionAction.ABORT, code))
            return _finish(DecisionAction.ABORT, reason=code, data_quality=DataQuality.DEGRADED)
        if not snapshot.report.ok:
            codes = [f"{i.code}" for i in snapshot.report.issues if i.level.value == "ERROR"]
            gates.append(
                GateResult("validation_report", False, DecisionAction.ABORT,
                           "validation_errors:" + ",".join(codes))
            )
            return _finish(DecisionAction.ABORT, reason="validation_errors", data_quality=DataQuality.INVALID)
        gates.append(GateResult("freshness", True, DecisionAction.HOLD))

        # ---- gate 4: spread ----------------------------------------------
        spread = snapshot.tick.spread_points
        max_spread = self.config.max_spread_points
        if max_spread is not None and spread is not None and spread > max_spread:
            gates.append(GateResult("spread", False, DecisionAction.ABORT, "spread_too_high"))
            return _finish(
                DecisionAction.ABORT,
                reason="spread_too_high",
                warnings=[f"spread {spread:.0f} points exceeds limit {max_spread:.0f}"],
            )
        gates.append(GateResult("spread", True, DecisionAction.HOLD))

        # ---- gate 5: risk state -------------------------------------------
        if not state.has_equity:
            gates.append(GateResult("risk_state", False, DecisionAction.ABORT, "missing_equity"))
            return _finish(DecisionAction.ABORT, reason="missing_equity", data_quality=DataQuality.DEGRADED)
        equity = float(state.equity)
        daily_limit = state.daily_loss_limit
        if daily_limit is None:
            daily_limit = equity * self.config.daily_loss_limit_pct / 100.0
        if state.daily_loss >= daily_limit:
            gates.append(GateResult("risk_state", False, DecisionAction.ABORT, "daily_loss_limit"))
            return _finish(
                DecisionAction.ABORT,
                reason="daily_loss_limit",
                warnings=[f"daily loss {state.daily_loss:.2f} ≥ limit {daily_limit:.2f}"],
            )
        if state.consecutive_losses >= self.config.max_consecutive_losses:
            gates.append(GateResult("risk_state", False, DecisionAction.ABORT, "consecutive_losses"))
            return _finish(DecisionAction.ABORT, reason="consecutive_losses")
        if state.open_positions >= self.config.max_open_positions:
            gates.append(GateResult("risk_state", False, DecisionAction.ABORT, "position_limit"))
            return _finish(DecisionAction.ABORT, reason="position_limit")
        if state.pending_orders >= self.config.max_pending_orders:
            gates.append(GateResult("risk_state", False, DecisionAction.ABORT, "pending_order_limit"))
            return _finish(DecisionAction.ABORT, reason="pending_order_limit")
        gates.append(GateResult("risk_state", True, DecisionAction.HOLD))

        # ---- gate 6: analysis (closed candles only) ------------------------
        sliced = self._slice_to_now(snapshot, now)
        context = build_market_context(sliced)
        h1 = context.features(TimeFrame.H1)
        if h1 is None or h1.length < 60:
            gates.append(GateResult("analysis", False, DecisionAction.HOLD, "insufficient_candles"))
            return _finish(
                DecisionAction.HOLD,
                reason="insufficient_candles",
                context=context,
                data_quality=DataQuality.INSUFFICIENT,
            )
        if not context.has_timeframe(TimeFrame.M15) or context.features(TimeFrame.M15) is None:
            gates.append(GateResult("analysis", False, DecisionAction.HOLD, "missing_entry_timeframe"))
            return _finish(
                DecisionAction.HOLD,
                reason="missing_entry_timeframe",
                context=context,
                warnings=["M15 (entry timing) missing — no proposal without entry confirmation context"],
                data_quality=DataQuality.DEGRADED,
            )

        assessment = self._detector.detect(context)
        context = context.with_regime(assessment)
        results = self._registry.run_all(context)
        synthesis = synthesize(
            SynthesisInput(
                symbol=context.symbol,
                timestamp=now,
                regime=assessment,
                agent_results=results,
            )
        )
        gates.append(GateResult("analysis", True, DecisionAction.HOLD))

        # ---- gate 7: edge quality ------------------------------------------
        if synthesis.action is DecisionAction.HOLD:
            gates.append(GateResult("edge", False, DecisionAction.HOLD, "no_edge"))
            return _finish(
                DecisionAction.HOLD,
                reason="no_edge",
                reasons=["synthesis found no actionable edge"]
                + list(synthesis.reasons),
                warnings=list(synthesis.conflicts),
                context=context,
                synthesis=synthesis,
                results=results,
            )

        direction = (
            AgentDirection.BUY if synthesis.action is DecisionAction.BUY else AgentDirection.SELL
        )
        side_score = synthesis.buy_score if direction is AgentDirection.BUY else synthesis.sell_score
        opposing_score = (
            synthesis.sell_score if direction is AgentDirection.BUY else synthesis.buy_score
        )
        net = side_score - opposing_score
        threshold = (
            self.config.buy_threshold if direction is AgentDirection.BUY else self.config.sell_threshold
        )
        if side_score < threshold or net < self.config.net_threshold:
            gates.append(GateResult("edge", False, DecisionAction.HOLD, "below_threshold"))
            return _finish(
                DecisionAction.HOLD,
                reason="below_threshold",
                reasons=[
                    f"side score {side_score:.2f} (threshold {threshold:.2f}), net {net:+.2f} "
                    f"(required {self.config.net_threshold:.2f})"
                ],
                context=context,
                synthesis=synthesis,
            )
        gates.append(GateResult("edge", True, DecisionAction.HOLD))

        supporting = [s for s in synthesis.supporting]
        if supporting and max(s.signal_strength for s in supporting) < self.config.minimum_signal_strength:
            gates.append(GateResult("signal_strength", False, DecisionAction.HOLD, "weak_signal"))
            return _finish(
                DecisionAction.HOLD,
                reason="weak_signal",
                reasons=[
                    f"strongest supporting agent below minimum signal strength "
                    f"{self.config.minimum_signal_strength:.2f}"
                ],
                context=context,
                synthesis=synthesis,
            )
        gates.append(GateResult("signal_strength", True, DecisionAction.HOLD))

        alignment = compute_alignment(context, direction, self.config)
        # purely threshold-driven: with default weights an opposing entry
        # timeframe (M15, weight .25) caps agreement at .75 < .80 -> HOLD;
        # a configured lower threshold explicitly permits the setup.
        if alignment.score < self.config.minimum_timeframe_alignment:
            opposing = ", ".join(tf.value for tf in alignment.opposing) or "none"
            gates.append(GateResult("timeframe_alignment", False, DecisionAction.HOLD, "timeframe_conflict"))
            return _finish(
                DecisionAction.HOLD,
                reason="timeframe_conflict",
                reasons=[
                    f"timeframe alignment {alignment.score:.2f} "
                    f"(required {self.config.minimum_timeframe_alignment:.2f}); "
                    f"opposing timeframes: {opposing}"
                ],
                context=context,
                synthesis=synthesis,
                alignment=alignment,
            )
        gates.append(GateResult("timeframe_alignment", True, DecisionAction.HOLD))

        conflict_score = self._conflict_score(synthesis)
        if conflict_score > self.config.max_conflict:
            gates.append(GateResult("conflict", False, DecisionAction.HOLD, "conflict_exceeds_tolerance"))
            return _finish(
                DecisionAction.HOLD,
                reason="conflict_exceeds_tolerance",
                reasons=[
                    f"conflict score {conflict_score:.2f} exceeds tolerance "
                    f"{self.config.max_conflict:.2f}"
                ],
                warnings=list(synthesis.conflicts),
                context=context,
                synthesis=synthesis,
                alignment=alignment,
                conflict_score=conflict_score,
            )
        gates.append(GateResult("conflict", True, DecisionAction.HOLD))

        # ---- gate 8: duplicate setup fingerprint ----------------------------
        fingerprint, setup_type = self._fingerprint(context, results, direction, assessment)
        if fingerprint in state.active_setup_fingerprints:
            gates.append(GateResult("duplicate_setup", False, DecisionAction.HOLD, "duplicate_setup"))
            return _finish(
                DecisionAction.HOLD,
                reason="duplicate_setup",
                reasons=["identical setup fingerprint already active — no repeated proposal"],
                context=context,
                synthesis=synthesis,
                alignment=alignment,
                conflict_score=conflict_score,
                fingerprint=fingerprint,
                setup_type=setup_type,
            )
        gates.append(GateResult("duplicate_setup", True, DecisionAction.HOLD))

        # ---- gate 9: proposal construction -----------------------------------
        decision = self._build_proposal(
            snapshot, context, results, synthesis, assessment, alignment,
            conflict_score, direction, fingerprint, setup_type, state, now, gates,
        )
        return decision

    # ------------------------------------------------------------------
    def _build_proposal(
        self, snapshot, context, results, synthesis, assessment, alignment,
        conflict_score, direction, fingerprint, setup_type, state, now, gates,
    ) -> Decision:
        symbol: SymbolSpec = snapshot.symbol
        entry = self._entry_provider.entry_price(direction, snapshot)
        if entry is None or entry <= 0:
            gates.append(GateResult("entry_price", False, DecisionAction.ABORT, "invalid_entry_price"))
            return self._make_decision(
                snapshot, now, DecisionAction.ABORT, reason="invalid_entry_price",
                context=context, synthesis=synthesis, alignment=alignment,
                conflict_score=conflict_score, fingerprint=fingerprint,
                setup_type=setup_type, gates=gates, data_quality=DataQuality.INVALID,
                results=results,
            )

        h1 = context.features(TimeFrame.H1)
        structure = next((r for r in results if r.agent == "structure"), None)
        invalidation = structure.features.get("invalidation_level") if structure else None
        last_high = structure.features.get("last_swing_high") if structure else None
        last_low = structure.features.get("last_swing_low") if structure else None
        opposing_level = None
        if self.config.tp_method == "structure":
            if direction is AgentDirection.BUY and last_high and last_high.get("price"):
                opposing_level = float(last_high["price"])
            elif direction is AgentDirection.SELL and last_low and last_low.get("price"):
                opposing_level = float(last_low["price"])

        levels = compute_levels(
            direction=direction,
            entry=entry,
            atr=h1.atr if h1 else None,
            structure_invalidation=invalidation,
            opposing_structure_level=opposing_level,
            symbol=symbol,
            config=self.config,
        )

        if levels.reward_risk < self.config.minimum_rr:
            gates.append(GateResult("reward_risk", False, DecisionAction.HOLD, "insufficient_rr"))
            return self._make_decision(
                snapshot, now, DecisionAction.HOLD, reason="insufficient_rr",
                reasons=[
                    f"reward:risk {levels.reward_risk:.2f} below minimum "
                    f"{self.config.minimum_rr:.2f} ({levels.tp_source})"
                ],
                context=context, synthesis=synthesis, alignment=alignment,
                conflict_score=conflict_score, fingerprint=fingerprint,
                setup_type=setup_type, gates=gates, results=results,
            )
        gates.append(GateResult("reward_risk", True, DecisionAction.HOLD))

        sizing = calculate_position_size(
            equity=float(state.equity),
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            entry=entry,
            stop_loss=levels.stop_loss,
            symbol=symbol,
        )
        if not sizing.feasible:
            gates.append(GateResult("position_size", False, DecisionAction.HOLD, "position_size_below_minimum"))
            return self._make_decision(
                snapshot, now, DecisionAction.HOLD, reason="position_size_below_minimum",
                reasons=["risk budget cannot fund the broker's minimum volume"],
                context=context, synthesis=synthesis, alignment=alignment,
                conflict_score=conflict_score, fingerprint=fingerprint,
                setup_type=setup_type, gates=gates, results=results,
            )
        gates.append(GateResult("position_size", True, DecisionAction.HOLD))

        geometry_issues = validate_levels(
            direction=direction, entry=entry,
            stop_loss=levels.stop_loss, take_profit=levels.take_profit,
            symbol=symbol,
        )
        if geometry_issues:
            gates.append(GateResult("geometry", False, DecisionAction.HOLD, "invalid_geometry"))
            return self._make_decision(
                snapshot, now, DecisionAction.HOLD, reason="invalid_geometry",
                reasons=geometry_issues,
                context=context, synthesis=synthesis, alignment=alignment,
                conflict_score=conflict_score, fingerprint=fingerprint,
                setup_type=setup_type, gates=gates, results=results,
            )
        gates.append(GateResult("geometry", True, DecisionAction.HOLD))

        decision_id = self._decision_id(snapshot, now, fingerprint)
        reasons = [
            f"{direction.value} proposal: {setup_type} setup in {assessment.regime.value} regime",
            f"entry {'ask' if direction is AgentDirection.BUY else 'bid'} {entry:.2f} (validated tick)",
            *levels.notes,
            f"alignment {alignment.score:.2f}, conflict {conflict_score:.2f}, "
            f"side score {(synthesis.buy_score if direction is AgentDirection.BUY else synthesis.sell_score):.2f}",
        ]
        proposal = TradeProposal.build(
            decision_id=decision_id,
            fingerprint=fingerprint,
            setup_type=setup_type,
            direction=direction,
            entry=entry,
            levels=levels,
            sizing=sizing,
            symbol=symbol,
            timeframe=TimeFrame.H1,
            regime=assessment.regime,
            reasons=reasons,
            invalidation_conditions=invalidation_conditions(levels, assessment.regime, self.config),
            created_at=now,
            config=self.config,
        )
        return self._make_decision(
            snapshot, now,
            DecisionAction.BUY if direction is AgentDirection.BUY else DecisionAction.SELL,
            reasons=reasons,
            context=context, synthesis=synthesis, alignment=alignment,
            conflict_score=conflict_score, fingerprint=fingerprint,
            setup_type=setup_type, gates=gates, proposal=proposal,
            results=results,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_decision(self, snapshot, now, action, *, reason=None, reasons=None,
                       warnings=None, context=None, synthesis=None, alignment=None,
                       conflict_score=0.0, fingerprint=None, setup_type=None,
                       gates=None, proposal=None, data_quality=DataQuality.OK,
                       results=None) -> Decision:
        if gates is None:
            gates = []
        rejection = [reason] if reason and action in (DecisionAction.HOLD, DecisionAction.ABORT) else []
        src_time = regime = volatility = None
        supporting: list[str] = []
        opposing: list[str] = []
        if context is not None:
            h1 = context.features(TimeFrame.H1)
            src_time = h1.last_time if h1 else None
            if context.regime is not None:
                regime = context.regime.regime
                volatility = context.regime.volatility
        if synthesis is not None:
            supporting = [s.agent for s in synthesis.supporting]
            opposing = [s.agent for s in synthesis.opposing]
        return Decision(
            decision_id=self._decision_id(snapshot, now, fingerprint or "none"),
            decision=action,
            symbol=snapshot.symbol.name if snapshot.symbol else "",
            timestamp=now,
            source_candle_time=src_time,
            regime=regime,
            volatility=volatility,
            session_state=snapshot.session_state,
            spread_points=snapshot.tick.spread_points,
            conflict_score=round(conflict_score, 6),
            alignment_score=round(alignment.score, 6) if alignment else None,
            supporting_agents=supporting,
            opposing_agents=opposing,
            reasons=reasons or ([reason] if reason else []),
            warnings=warnings or [],
            rejection_reasons=rejection,
            gates=[g if isinstance(g, dict) else {"gate": g.gate, "passed": g.passed,
                                                  "action": g.action.value, "reason": g.reason} for g in gates],
            data_quality=data_quality,
            fingerprint=fingerprint,
            setup_type=setup_type,
            proposal=proposal,
            snapshot_created_at=snapshot.created_at,
            agent_results=[r.model_dump(mode="json") for r in (results or [])],
            synthesis_payload=synthesis.model_dump(mode="json") if synthesis else None,
        )

    @staticmethod
    def _conflict_score(synthesis: SynthesisOutput) -> float:
        """Share of opposing weighted evidence in the total directional mass."""
        total = sum(s.weighted_strength for s in synthesis.supporting + synthesis.opposing)
        if total <= 0:
            return 0.0
        opposing = sum(s.weighted_strength for s in synthesis.opposing)
        return opposing / total

    def _fingerprint(
        self, context, results: list[AgentResult], direction: AgentDirection,
        assessment: RegimeAssessment,
    ) -> tuple[str, str]:
        """Deterministic setup fingerprint (Phase-3 §24).

        Identity = symbol + decision candle + direction + regime + structure
        state + leading setup type.  Two evaluation cycles with the same
        fingerprint are the SAME setup — the anti-overtrading gate suppresses
        the duplicate.  No wall-clock time is involved: a new candle, a new
        regime or a different leading agent changes the fingerprint.
        """
        h1 = context.features(TimeFrame.H1)
        candle = h1.last_time.isoformat() if h1 and h1.last_time else "none"
        structure = next((r for r in results if r.agent == "structure"), None)
        structure_state = "none"
        if structure is not None:
            last_event = structure.features.get("last_event")
            structure_state = (
                f"{structure.features.get('bias', 'UNKNOWN')}|{last_event['kind'] if last_event else 'no-event'}"
            )
        # leading agent = highest regime-weighted supporting strength
        from app.decision.regime import AGENT_RELEVANCE_WEIGHTS, relevance_of

        leading = None
        best = -1.0

        for result in results:
            if result.direction is not direction:
                continue
            relevance = relevance_of(assessment.regime, result.agent)
            weighted = result.signal_strength * AGENT_RELEVANCE_WEIGHTS.get(relevance, 1.0)
            if weighted > best:
                best = weighted
                leading = result.agent
        setup_type = leading or "unknown"
        payload = "|".join([
            context.symbol, candle, direction.value, assessment.regime.value,
            structure_state, setup_type,
        ])
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return digest[:16], setup_type

    @staticmethod
    def _decision_id(snapshot: MarketSnapshot, now: datetime, fingerprint: str) -> str:
        payload = f"{snapshot.symbol.name if snapshot.symbol else ''}|{now.isoformat()}|{fingerprint}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _slice_to_now(self, snapshot: MarketSnapshot, now: datetime) -> MarketSnapshot:
        """Defense-in-depth no-look-ahead: keep only candles CLOSED by ``now``.

        A candle opened at ``t`` closes at ``t + timeframe``; anything closing
        later than ``now`` is the future and must not reach the analysis.
        """
        sliced: dict[TimeFrame, object] = {}
        changed = False
        for timeframe, check in snapshot.series.items():
            series = check.series
            if series is None or not series.candles:
                sliced[timeframe] = check
                continue
            duration = timedelta(minutes=timeframe.minutes)
            kept = [c for c in series.candles if c.time + duration <= now]
            if len(kept) != len(series.candles):
                changed = True
                new_series = CandleSeries(
                    symbol=series.symbol, timeframe=series.timeframe, candles=kept
                )
                sliced[timeframe] = check.model_copy(update={"series": new_series})
            else:
                sliced[timeframe] = check
        if not changed:
            return snapshot
        return snapshot.model_copy(update={"series": sliced})
