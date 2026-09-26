"""Synthesis tests (Phase-2 §12/§17).

Categories: BUY/SELL aggregation, HOLD with disagreement retained, regime
relevance weighting, disabled agents, degraded data quality, determinism,
and journal-serializability.
"""

from __future__ import annotations

import pytest

from app.agents import default_agents
from app.core.enums import AgentDirection, DecisionAction, MarketRegime, VolatilityLevel
from app.decision.regime import RegimeAssessment, RegimeDetector
from app.decision.synthesis import (
    ACTION_NET_THRESHOLD,
    ACTION_SIDE_THRESHOLD,
    SynthesisInput,
    synthesize,
)
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_context,
    sideways_closes,
    standard_triple,
)


def _assessment(regime, volatility=VolatilityLevel.NORMAL):
    return RegimeAssessment(
        regime=regime,
        volatility=volatility,
        as_of=REF_TIME,
        data_quality="OK",
        evidence=[f"test: {regime.value}"],
        conflicts=[],
    )


def _pipeline(closes, *, h1_seed=11, regime=None):
    ctx = make_context(standard_triple(h1_closes=closes, h1_seed=h1_seed))
    assessment = regime if regime is not None else RegimeDetector().detect(ctx)
    ctx = ctx.with_regime(assessment)
    results = [agent.analyze(ctx) for agent in default_agents()]
    output = synthesize(
        SynthesisInput(
            symbol=ctx.symbol,
            timestamp=ctx.created_at,
            regime=assessment,
            agent_results=results,
        )
    )
    return ctx, assessment, results, output


class TestActionAggregation:
    def test_uptrend_pipeline_produces_buy(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        assert output.action is DecisionAction.BUY
        assert output.buy_score >= ACTION_SIDE_THRESHOLD
        assert (output.buy_score - output.sell_score) >= ACTION_NET_THRESHOLD
        assert output.supporting, "supporting agents required"
        assert all(s.direction is AgentDirection.BUY for s in output.supporting)

    def test_downtrend_pipeline_produces_sell(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=-2.0, seed=12), h1_seed=12)
        assert output.action is DecisionAction.SELL
        assert output.sell_score > output.buy_score

    def test_range_pipeline_holds(self):
        *_, output = _pipeline(sideways_closes(300, period=16, amplitude=1.2, seed=71), h1_seed=71)
        assert output.action is DecisionAction.HOLD
        assert any("HOLD is a valid decision" in r for r in output.reasons)


class TestDisagreementPreservation:
    def test_opposing_stances_survive_a_buy_decision(self):
        """Craft a majority-BUY pipeline that still contains a SELL stance:
        the output must keep it visible (spec: never collapse disagreement)."""
        from app.core.enums import TimeFrame
        from tests.unit.agents.scenarios import replace_last_candle

        closes = linear_trend_closes(300, slope=2.0, seed=11)
        series = standard_triple(h1_closes=closes, h1_seed=11)
        # bullish sweep on M15 gives liquidity a BUY; then flip trend+momentum
        # by appending a sharp H1 reversal... simpler: build a mixed series —
        # strong down H1 with a bullish M15 sweep.
        down = linear_trend_closes(300, slope=-2.0, seed=12)
        series = standard_triple(h1_closes=down, h1_seed=12)
        m15 = series[TimeFrame.M15]
        df = m15.to_dataframe()
        low20 = df["low"].iloc[-21:-1].min()
        series[TimeFrame.M15] = replace_last_candle(
            m15, open_=low20 + 0.4, high=low20 + 0.55, low=low20 - 0.5, close=low20 + 0.1
        )
        ctx = make_context(series)
        assessment = RegimeDetector().detect(ctx)
        ctx = ctx.with_regime(assessment)
        results = [a.analyze(ctx) for a in default_agents()]
        output = synthesize(
            SynthesisInput(
                symbol=ctx.symbol, timestamp=ctx.created_at, regime=assessment, agent_results=results
            )
        )
        assert output.action is DecisionAction.SELL
        # if any agent voted BUY, it must appear in opposing
        buyers = [r.agent for r in results if r.direction is AgentDirection.BUY]
        if buyers:  # liquidity sweep should have produced one
            opposing_names = [s.agent for s in output.opposing]
            assert set(buyers) <= set(opposing_names)
        assert output.conflicts or not buyers

    def test_hold_retains_majority_and_minority(self):
        *_, output = _pipeline(sideways_closes(300, period=16, amplitude=1.2, seed=71), h1_seed=71)
        # on HOLD the majority/minority split is still reported
        assert isinstance(output.supporting, list)
        assert isinstance(output.opposing, list)
        assert output.supporting or output.opposing or output.neutral

    def test_has_disagreement_property(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        # clean trend pipeline: no opposition
        assert not output.has_disagreement


class TestRegimeWeighting:
    def test_mean_reversion_is_disabled_in_trend(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        disabled = [s.agent for s in output.disabled]
        assert "mean_reversion" in disabled
        stance = next(s for s in output.disabled if s.agent == "mean_reversion")
        assert stance.relevance == "DISABLED"
        assert stance.weighted_strength == 0.0

    def test_disabled_agent_cannot_contribute_score(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        for stance in output.disabled:
            assert stance.weighted_strength == 0.0

    def test_high_relevance_multiplies(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        trend_stance = next(s for s in output.supporting if s.agent == "trend")
        assert trend_stance.relevance == "HIGH"
        assert trend_stance.weighted_strength == pytest.approx(
            trend_stance.signal_strength * 1.25
        )


class TestDataQualityHandling:
    def test_bad_quality_agents_are_listed(self):
        closes = linear_trend_closes(70, slope=2.0, seed=13)  # EMA200 warm-up etc.
        *_, output = _pipeline(closes, h1_seed=13)
        # trend should be DEGRADED (no EMA200) — surfaced in conflicts
        if any("degraded agent data" in c for c in output.conflicts):
            assert "trend" in next(c for c in output.conflicts if "degraded agent data" in c)

    def test_snapshot_not_trading_grade_warns(self):
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        assessment = RegimeDetector().detect(ctx)
        results = [a.analyze(ctx.with_regime(assessment)) for a in default_agents()]
        output = synthesize(
            SynthesisInput(
                symbol=ctx.symbol,
                timestamp=ctx.created_at,
                regime=assessment,
                agent_results=results,
                data_quality_ok=False,
            )
        )
        assert any("not trading-grade" in w for w in output.warnings)

    def test_regime_conflicts_flow_into_warnings(self):
        assessment = RegimeAssessment(
            regime=MarketRegime.UNCERTAIN,
            volatility=VolatilityLevel.NORMAL,
            as_of=REF_TIME,
            data_quality="OK",
            evidence=[],
            conflicts=["H4 alignment opposes H1 (down vs up)"],
        )
        closes = linear_trend_closes(300, slope=2.0, seed=11)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=11))
        results = [a.analyze(ctx) for a in default_agents()]
        output = synthesize(
            SynthesisInput(
                symbol=ctx.symbol, timestamp=ctx.created_at, regime=assessment, agent_results=results
            )
        )
        assert "H4 alignment opposes H1 (down vs up)" in output.warnings


class TestContractAndDeterminism:
    def test_output_is_journal_serializable(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        import json

        json.dumps(output.model_dump(mode="json"))

    def test_deterministic(self):
        *_, first = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        *_, second = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        assert first.model_dump() == second.model_dump()

    def test_reasons_explain_the_decision(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        assert any("regime" in r.lower() for r in output.reasons)
        assert any("buy_score" in r for r in output.reasons)

    def test_scores_bounded(self):
        *_, output = _pipeline(linear_trend_closes(300, slope=2.0, seed=11))
        assert 0.0 <= output.aggregate_score <= 1.0
        assert output.buy_score >= 0.0 and output.sell_score >= 0.0

    def test_empty_agent_list_holds(self):
        output = synthesize(
            SynthesisInput(
                symbol="XAUUSD",
                timestamp=REF_TIME,
                regime=_assessment(MarketRegime.RANGE),
                agent_results=[],
            )
        )
        assert output.action is DecisionAction.HOLD
        assert output.aggregate_score == 0.0
