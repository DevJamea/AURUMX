"""MeanReversionAgent behavioral tests (Phase-2 §9).

Categories: valid long/short setup, regime gate (trend excludes), local range
gate, edge-without-confirmation (no falling-knife catches), boundary
requirement, insufficient data, missing M15, and the explicit no-martingale /
no-averaging guarantees.
"""

from __future__ import annotations

from app.core.enums import AgentDirection, DataQuality, TimeFrame
from tests.unit.agents.scenarios import (
    linear_trend_closes,
    make_context,
    replace_last_candle,
    sideways_closes,
    standard_triple,
)


def _range_base() -> list[float]:
    return sideways_closes(299, period=16, amplitude=1.2, seed=71)


def _long_setup_triple(*, m15_confirmed: bool):
    triple = standard_triple(h1_closes=_range_base(), h1_seed=71)
    # deep dip below the lower band, closing near its own low
    triple[TimeFrame.H1] = replace_last_candle(
        triple[TimeFrame.H1], open_=2649.5, high=2649.6, low=2645.4, close=2645.8
    )
    if m15_confirmed:
        # bullish hammer: long lower wick (>= 33% of range), close at the top
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2646.0, high=2646.3, low=2645.2, close=2646.2
        )
    else:
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2646.4, high=2646.6, low=2645.9, close=2646.1
        )
    return triple


def _short_setup_triple(*, m15_confirmed: bool):
    triple = standard_triple(h1_closes=_range_base(), h1_seed=71)
    triple[TimeFrame.H1] = replace_last_candle(
        triple[TimeFrame.H1], open_=2650.5, high=2654.6, low=2650.4, close=2654.2
    )
    if m15_confirmed:
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2654.0, high=2654.8, low=2653.8, close=2653.8
        )
    else:
        triple[TimeFrame.M15] = replace_last_candle(
            triple[TimeFrame.M15], open_=2653.6, high=2654.1, low=2653.4, close=2653.9
        )
    return triple


class TestValidSetups:
    def test_confirmed_long_setup_buys(self, agents, range_regime):
        ctx = make_context(_long_setup_triple(m15_confirmed=True), regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.BUY
        assert result.is_actionable
        assert 0.5 <= result.signal_strength <= 1.0
        joined = " | ".join(result.reasons)
        assert "lower statistical edge" in joined
        assert "range low" in joined
        assert "M15" in joined

    def test_confirmed_short_setup_sells(self, agents, range_regime):
        ctx = make_context(_short_setup_triple(m15_confirmed=True), regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.SELL
        assert result.is_actionable

    def test_strength_documents_its_components(self, agents, range_regime):
        ctx = make_context(_long_setup_triple(m15_confirmed=True), regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        for key in ("percent_b", "deviation_atr", "boundary_proximity", "confirmation"):
            assert key in result.features, f"provenance missing: {key}"


class TestRegimeGate:
    def test_strong_uptrend_excludes_mean_reversion(self, agents, trend_up_regime):
        """Even a textbook oversold reading must stay NEUTRAL in a trend."""
        ctx = make_context(_long_setup_triple(m15_confirmed=True), regime=trend_up_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert any("regime gate" in r for r in result.reasons)
        assert result.features["gated"] is True

    def test_strong_downtrend_excludes_mean_reversion(self, agents, trend_down_regime):
        ctx = make_context(_short_setup_triple(m15_confirmed=True), regime=trend_down_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert any("regime gate" in r for r in result.reasons)

    def test_high_volatility_excludes_mean_reversion(self, agents, high_vol_regime):
        ctx = make_context(_long_setup_triple(m15_confirmed=True), regime=high_vol_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert any("regime gate" in r for r in result.reasons)

    def test_local_gate_blocks_directional_markets_without_regime(self, agents):
        closes = linear_trend_closes(300, slope=-2.0, seed=12)
        ctx = make_context(standard_triple(h1_closes=closes, h1_seed=12))
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert any("range not confirmed locally" in r for r in result.reasons)


class TestConfirmationGate:
    def test_edge_without_confirmation_is_neutral_with_warning(self, agents, range_regime):
        ctx = make_context(_long_setup_triple(m15_confirmed=False), regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert any("no M15 reversal confirmation" in r for r in result.reasons)
        assert any("falling-knife" in w for w in result.warnings)

    def test_mid_range_deviation_is_not_an_edge(self, agents, range_regime):
        # plain range data, no dip: percent_b comfortably inside the bands
        ctx = make_context(
            standard_triple(h1_closes=_range_base(), h1_seed=71), regime=range_regime
        )
        result = agents["mean_reversion"].analyze(ctx)
        assert result.direction is AgentDirection.NEUTRAL
        assert any("no statistical edge" in r for r in result.reasons)


class TestNoMartingaleGuarantees:
    def test_features_explicitly_disallow_martingale_and_averaging(self, agents, range_regime):
        ctx = make_context(_long_setup_triple(m15_confirmed=True), regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.features["no_martingale"] is True
        assert result.features["no_averaging"] is True


class TestDataRequirements:
    def test_insufficient_h1(self, agents, range_regime):
        short = linear_trend_closes(45, slope=0.0)
        series = standard_triple(h1_closes=short, h1_seed=16)
        ctx = make_context(series, regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.data_quality is DataQuality.INSUFFICIENT

    def test_missing_m15_degrades(self, agents, range_regime):
        series = standard_triple(h1_closes=_range_base(), h1_seed=71)
        del series[TimeFrame.M15]
        ctx = make_context(series, regime=range_regime)
        result = agents["mean_reversion"].analyze(ctx)
        assert result.data_quality is DataQuality.DEGRADED
        assert any("M15" in r for r in result.reasons)
