"""Adversarial input tests (Phase-2 §18).

Every hostile input must fail SAFE: a well-formed (usually NEUTRAL) result or
a hard validation rejection — never an exception, never a crash, never a
guess dressed up as a signal.

Covered: NaN, infinities, duplicate timestamps, unordered timestamps, missing
candles (gaps), insufficient candles, zero volume, extreme spikes, constant
prices, flat markets, abnormal ATR, invalid/absent timeframes.
"""

from __future__ import annotations

import pydantic
import pytest

from app.agents import default_agents
from app.core.enums import AgentDirection, DataQuality, TimeFrame
from app.core.models import Candle, CandleSeries
from tests.unit.agents.scenarios import (
    REF_TIME,
    linear_trend_closes,
    make_context,
    make_series,
    sideways_closes,
    standard_triple,
)

AGENTS = {a.name: a for a in default_agents()}


def _all_analyze(ctx):
    """Run every agent; none may raise, all results well-formed."""
    results = {}
    for name, agent in AGENTS.items():
        result = agent.analyze(ctx)  # must not raise
        assert 0.0 <= result.signal_strength <= 1.0
        assert isinstance(result.direction, AgentDirection)
        results[name] = result
    return results


class TestNaNAndInfinity:
    def test_nan_candle_is_rejected_at_validation(self):
        with pytest.raises(pydantic.ValidationError):
            Candle(
                time=REF_TIME,
                open=float("nan"),
                high=2651.0,
                low=2649.0,
                close=2650.0,
                tick_volume=100.0,
                real_volume=0.0,
            )

    @pytest.mark.parametrize("field", ["open", "high", "low", "close"])
    def test_infinity_candle_is_rejected(self, field):
        kwargs = dict(
            time=REF_TIME,
            open=2650.0,
            high=2651.0,
            low=2649.0,
            close=2650.0,
            tick_volume=100.0,
            real_volume=0.0,
        )
        kwargs[field] = float("inf")
        with pytest.raises(pydantic.ValidationError):
            Candle(**kwargs)

    def test_results_never_contain_nan_or_inf(self):
        import math

        ctx = make_context(standard_triple(h1_closes=sideways_closes(300, amplitude=1.2, seed=71), h1_seed=71))
        for result in _all_analyze(ctx).values():
            dump = result.model_dump()
            def _check(value):
                if isinstance(value, float):
                    assert math.isfinite(value)
                elif isinstance(value, dict):
                    for v in value.values():
                        _check(v)
                elif isinstance(value, (list, tuple)):
                    for v in value:
                        _check(v)
            _check(dump)


class TestBrokenTimeSeries:
    def _base_series(self):
        return make_series(linear_trend_closes(120, slope=1.0), TimeFrame.H1, seed=21)

    def test_duplicate_timestamps_fail_safe(self):
        series = self._base_series()
        candles = list(series.candles)
        candles[-1] = candles[-1].model_copy(update={"time": candles[-2].time})
        dup = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)
        results = _all_analyze(make_context({TimeFrame.H1: dup, TimeFrame.M15: dup}))
        # fail-safe: no crashes; every result carries a sane data quality
        allowed = (DataQuality.OK, DataQuality.DEGRADED, DataQuality.INSUFFICIENT,
                   DataQuality.INVALID, DataQuality.NO_DATA)
        assert all(r.data_quality in allowed for r in results.values())

    def test_unordered_timestamps_fail_safe(self):
        series = self._base_series()
        candles = list(series.candles)
        candles[10], candles[11] = candles[11], candles[10]
        shuffled = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)
        _all_analyze(make_context({TimeFrame.H1: shuffled, TimeFrame.M15: shuffled}))

    def test_missing_candles_time_gaps_fail_safe(self):
        series = self._base_series()
        candles = [c for i, c in enumerate(series.candles) if i not in (50, 51, 52, 53)]
        gapped = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)
        results = _all_analyze(make_context({TimeFrame.H1: gapped, TimeFrame.M15: gapped}))
        assert all(r.direction is not None for r in results.values())

    def test_series_with_one_candle(self):
        series = make_series([2650.0], TimeFrame.H1, seed=1)
        results = _all_analyze(make_context({TimeFrame.H1: series, TimeFrame.M15: series}))
        for name, result in results.items():
            assert result.direction is AgentDirection.NEUTRAL, name
            assert result.signal_strength == 0.0, name
            assert result.data_quality in (DataQuality.INSUFFICIENT, DataQuality.INVALID, DataQuality.NO_DATA), name

    def test_empty_series(self):
        series = CandleSeries(symbol="XAUUSD", timeframe=TimeFrame.H1, candles=[])
        results = _all_analyze(make_context({TimeFrame.H1: series}))
        assert all(r.direction is AgentDirection.NEUTRAL for r in results.values())

    def test_only_m15_available(self):
        """Agents whose primary timeframe is missing must degrade to a neutral
        unavailable result rather than pretending to analyze M15."""
        series = make_series(linear_trend_closes(300, slope=2.0), TimeFrame.M15, seed=22)
        results = _all_analyze(make_context({TimeFrame.M15: series}))
        for name in ("trend", "momentum", "structure", "volatility", "mean_reversion"):
            assert results[name].direction is AgentDirection.NEUTRAL, name
            assert results[name].data_quality in (DataQuality.INSUFFICIENT, DataQuality.INVALID), name
        # M15-primary liquidity still works
        assert results["liquidity"].data_quality is DataQuality.OK


class TestHostilePrices:
    def test_extreme_spike_does_not_crash_or_explode(self):
        closes = sideways_closes(299, period=16, amplitude=1.2, seed=71)
        series = make_series(closes, TimeFrame.H1, seed=71)
        spike = list(series.candles)
        last = spike[-1]
        spike[-1] = Candle(
            time=last.time,
            open=last.open,
            high=3650.0,
            low=last.low,
            close=3600.0,
            tick_volume=last.tick_volume,
            real_volume=0.0,
        )
        spiked = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=spike)
        m15 = make_series(closes, TimeFrame.M15, seed=72)
        results = _all_analyze(make_context({TimeFrame.H1: spiked, TimeFrame.M15: m15}))
        # volatility flags it rather than hiding it
        assert results["volatility"].features["level"] in ("HIGH", "EXTREME")
        # nothing directional with a huge strength
        for result in results.values():
            assert result.signal_strength <= 1.0

    def test_constant_prices_are_neutral_everywhere(self):
        """Spec case: literally constant candles (zero range, zero volume
        movement) — every agent must fail safe to NEUTRAL."""
        from tests.unit.agents.scenarios import candle_times

        def constant(tf, count):
            times = candle_times(tf, count)
            return CandleSeries(
                symbol="XAUUSD",
                timeframe=tf,
                candles=[
                    Candle(time=t, open=2650.0, high=2650.0, low=2650.0, close=2650.0,
                           tick_volume=500.0, real_volume=0.0)
                    for t in times
                ],
            )

        series = {
            TimeFrame.H1: constant(TimeFrame.H1, 300),
            TimeFrame.M15: constant(TimeFrame.M15, 300),
            TimeFrame.H4: constant(TimeFrame.H4, 300),
        }
        results = _all_analyze(make_context(series))
        for agent_name, result in results.items():
            assert result.direction is AgentDirection.NEUTRAL, agent_name
            assert result.signal_strength == 0.0, agent_name

    def test_flat_closes_with_wick_noise_stay_weak(self):
        """Flat closes + random wicks: liquidity may read micro-sweeps, but
        only at proportionally weak strength (scale-free pattern detection)."""
        series = standard_triple(h1_closes=[2650.0] * 300, h1_seed=7)
        results = _all_analyze(make_context(series))
        for name, result in results.items():
            assert result.signal_strength <= 0.35, f"{name} too strong on flat data"

    def test_zero_volume_liquidity_warns(self):
        series = make_series(sideways_closes(60, period=16, amplitude=0.8, seed=81), TimeFrame.M15, seed=82)
        candles = [c.model_copy(update={"tick_volume": 0.0}) for c in series.candles]
        zero = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)
        result = AGENTS["liquidity"].analyze(make_context({TimeFrame.M15: zero}))
        assert result.direction is AgentDirection.NEUTRAL or result.data_quality is DataQuality.DEGRADED
        assert any("volume" in w.lower() for w in result.warnings)

    def test_abnormal_atr_from_alternating_extremes(self):
        closes = sideways_closes(280, period=16, amplitude=1.2, seed=71) + [
            2656.0, 2644.0, 2656.0, 2644.0, 2656.0, 2644.0, 2656.0, 2644.0,
            2656.0, 2644.0, 2656.0, 2644.0,
        ]
        results = _all_analyze(make_context(standard_triple(h1_closes=closes, h1_seed=71)))
        assert results["volatility"].features["level"] == "EXTREME"
        assert any("extreme" in w.lower() for w in results["volatility"].warnings)

    def test_tiny_prices_and_huge_prices_fail_safe(self):
        for scale in (0.001, 100000.0):
            closes = [round(2650.0 * scale + i * 2.0 * scale, 6) for i in range(120)]
            series = standard_triple(h1_closes=closes, h1_seed=23)
            results = _all_analyze(make_context(series))
            assert all(r.signal_strength <= 1.0 for r in results.values())


class TestFailSafePrinciple:
    def test_hostile_inputs_never_produce_strong_directional_signals(self):
        """Across every hostile dataset, no agent may return a strong signal —
        hostile data must degrade confidence, never manufacture it."""
        hostile = [
            ("flat", [2650.0] * 120),
            ("spike", sideways_closes(119, period=16, amplitude=0.5, seed=92) + [4000.0]),
        ]
        for label, closes in hostile:
            series = standard_triple(h1_closes=closes, h1_seed=24)
            results = _all_analyze(make_context(series))
            for name, result in results.items():
                assert result.signal_strength < 0.7, (
                    f"{label}/{name} manufactured strength from hostile data"
                )

    def test_extreme_spike_is_fully_suppressed_by_bad_tick_guard(self):
        """A +50% single-bar spike is physically impossible for gold: every
        directional agent must return NEUTRAL with the bad-tick warning."""
        closes = sideways_closes(119, period=16, amplitude=0.5, seed=92) + [4000.0]
        series = standard_triple(h1_closes=closes, h1_seed=24)
        results = _all_analyze(make_context(series))
        for name in ("trend", "momentum", "structure", "liquidity", "mean_reversion"):
            assert results[name].direction is AgentDirection.NEUTRAL, name
            assert any("abnormal" in w.lower() for w in results[name].warnings), name
            assert results[name].data_quality is DataQuality.DEGRADED, name
        # volatility still reports the measurement (informative, not directional)
        assert results["volatility"].features["level"] == "EXTREME"

    def test_every_agent_survives_the_full_adversarial_matrix(self):
        """Every agent x every hostile context: no exceptions, ever."""
        contexts = []
        series = make_series(linear_trend_closes(120, slope=1.0), TimeFrame.H1, seed=21)
        candles = list(series.candles)
        candles[-1] = candles[-1].model_copy(update={"time": candles[-2].time})
        dup = CandleSeries(symbol=series.symbol, timeframe=series.timeframe, candles=candles)
        contexts.append(("dup-time", make_context({TimeFrame.H1: dup, TimeFrame.M15: series})))
        contexts.append(("insufficient", make_context(standard_triple(h1_closes=linear_trend_closes(20), h1_seed=25))))
        contexts.append(("empty", make_context({})))
        for _, ctx in contexts:
            for agent in AGENTS.values():
                result = agent.analyze(ctx)  # no exception
                assert result.agent == agent.name
                assert 0.0 <= result.signal_strength <= 1.0
