"""End-to-end agent pipeline: FakeMT5 -> MarketDataService -> context ->
RegimeDetector -> AgentRegistry -> synthesize.

The agents sit downstream of real Phase-1 machinery (broker, data service,
validation) — this proves the whole read-only analysis stack works together,
deterministically, with disagreement preserved and nothing mutated.
"""

from __future__ import annotations

import json

from app.agents import build_market_context, default_agents
from app.agents.registry import AgentRegistry
from app.brokers.mt5 import MT5Broker
from app.core.enums import AgentDirection, DataQuality, DecisionAction, TimeFrame
from app.decision.regime import RegimeDetector
from app.decision.synthesis import SynthesisInput, synthesize
from app.market.data_service import MarketDataService
from tests.conftest import REF_TIME, make_fake_mt5
from tests.fakes.mt5_fake import TF_MINUTES
from tests.unit.agents.scenarios import linear_trend_closes, sideways_closes


def make_service(fake, *, candle_count: int = 300) -> MarketDataService:
    broker = MT5Broker(mt5_module=fake)
    broker.connect()
    return MarketDataService(
        broker,
        symbol="AUTO",
        candle_count=candle_count,
        max_tick_age_seconds=60,
        candle_freshness_multiplier=3.0,
        clock=lambda: REF_TIME,
    )


def inject_scenario(fake, closes_by_tf: dict[TimeFrame, list[float]], *, seed: int = 7):
    """Serve hand-built scenario closes through the fake terminal (oldest ->
    newest, last bar = forming bar, mirroring a live terminal)."""
    now_epoch = int(REF_TIME.timestamp())
    from tests.fakes.mt5_fake import TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_M15
    tf_const = {TimeFrame.M15: TIMEFRAME_M15, TimeFrame.H1: TIMEFRAME_H1, TimeFrame.H4: TIMEFRAME_H4}
    import random

    for tf, closes in closes_by_tf.items():
        step = TF_MINUTES[tf_const[tf]] * 60
        end = now_epoch  # forming bar opens at floor(now)
        n = len(closes)
        times = [end - i * step for i in range(n)][::-1]
        rng = random.Random(seed)
        bars = []
        for i, (t, close) in enumerate(zip(times, closes, strict=True)):
            open_price = closes[i - 1] if i else close
            high = max(open_price, close) + rng.uniform(0.05, 0.3)
            low = min(open_price, close) - rng.uniform(0.05, 0.3)
            bars.append(
                {
                    "time": t,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "tick_volume": rng.randint(80, 900),
                    "spread": 20,
                    "real_volume": 0,
                }
            )
        fake.add_rates("XAUUSD", tf_const[tf], bars)
    fake.set_tick("XAUUSD", epoch=now_epoch - 2, bid=closes_by_tf[TimeFrame.H1][-1],
                  ask=closes_by_tf[TimeFrame.H1][-1] + 0.2)


def run_pipeline(service):
    snapshot = service.get_snapshot()
    context = build_market_context(snapshot)
    assessment = RegimeDetector().detect(context)
    context = context.with_regime(assessment)
    results = AgentRegistry().run_all(context)
    output = synthesize(
        SynthesisInput(
            symbol=context.symbol,
            timestamp=context.created_at,
            regime=assessment,
            agent_results=results,
        )
    )
    return snapshot, context, assessment, results, output


class TestFullPipeline:
    def test_random_walk_market_runs_end_to_end(self, fake_mt5):
        service = make_service(fake_mt5)
        snapshot, context, assessment, results, output = run_pipeline(service)
        assert snapshot.trading_data_ok
        assert len(results) == 7
        assert output.action in (DecisionAction.BUY, DecisionAction.SELL, DecisionAction.HOLD)
        for result in results:
            assert result.data_quality in (
                DataQuality.OK, DataQuality.DEGRADED, DataQuality.INSUFFICIENT,
                DataQuality.INVALID, DataQuality.NO_DATA,
            )
        json.dumps(output.model_dump(mode="json"))

    def test_trending_market_synthesizes_buy(self):
        fake = make_fake_mt5(with_market=False)
        up = linear_trend_closes(301, slope=2.0, seed=11)  # +1 forming bar
        inject_scenario(fake, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up}, seed=11)
        *_, assessment, results, output = run_pipeline(make_service(fake))
        assert assessment.regime.value == "TREND_UP"
        by_name = {r.agent: r for r in results}
        assert by_name["trend"].direction is AgentDirection.BUY
        assert by_name["mean_reversion"].direction is AgentDirection.NEUTRAL  # regime gate
        assert output.action is DecisionAction.BUY

    def test_range_market_holds(self):
        fake = make_fake_mt5(with_market=False)
        rng = sideways_closes(301, period=16, amplitude=1.2, seed=71)
        inject_scenario(fake, {TimeFrame.M15: rng, TimeFrame.H1: rng, TimeFrame.H4: rng}, seed=71)
        *_, assessment, results, output = run_pipeline(make_service(fake))
        assert assessment.regime.value in ("RANGE", "UNCERTAIN", "LOW_VOLATILITY")
        by_name = {r.agent: r for r in results}
        assert by_name["trend"].direction is AgentDirection.NEUTRAL
        assert output.action is DecisionAction.HOLD

    def test_determinism_across_identical_services(self):
        outputs = []
        for _ in range(2):
            fake = make_fake_mt5(with_market=False)
            up = linear_trend_closes(301, slope=2.0, seed=11)
            inject_scenario(fake, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up}, seed=11)
            _, _, _, results, output = run_pipeline(make_service(fake))
            outputs.append((output.model_dump(), [r.model_dump() for r in results]))
        assert outputs[0] == outputs[1]

    def test_pipeline_does_not_mutate_the_snapshot(self):
        fake = make_fake_mt5(with_market=False)
        up = linear_trend_closes(301, slope=2.0, seed=11)
        inject_scenario(fake, {TimeFrame.M15: up, TimeFrame.H1: up, TimeFrame.H4: up}, seed=11)
        snapshot, context, *_ = run_pipeline(make_service(fake))
        before = snapshot.model_dump()
        for agent in default_agents():  # second pass over the same context
            agent.analyze(context)
        assert snapshot.model_dump() == before

    def test_stale_market_still_analyzes_fail_safe(self):
        """Candles older than the freshness budget: the pipeline must still
        produce a complete analysis cycle (agents analyze what exists)."""
        from datetime import timedelta

        stale_now = REF_TIME + timedelta(hours=8)
        fake = make_fake_mt5(now=stale_now, with_market=True)
        # service clock still at REF_TIME -> data looks stale
        service = MarketDataService(
            MT5Broker(mt5_module=fake),
            symbol="AUTO",
            candle_count=300,
            max_tick_age_seconds=60,
            candle_freshness_multiplier=3.0,
            clock=lambda: REF_TIME,
        )
        service.connect()
        snapshot = service.get_snapshot()
        context = build_market_context(snapshot)
        results = AgentRegistry().run_all(context)
        assert len(results) == 7  # fail-safe: every agent still reports

    def test_agent_order_stable_in_pipeline(self, fake_mt5):
        *_, results, _ = run_pipeline(make_service(fake_mt5))
        assert [r.agent for r in results] == [
            "trend", "momentum", "structure", "liquidity",
            "volatility", "mean_reversion", "macro",
        ]


class TestDisagreementEndToEnd:
    def test_mixed_market_preserves_stances(self):
        """Down-trending H1 with a bullish M15 liquidity sweep: whatever the
        action, every non-neutral stance stays visible in the output."""
        from tests.unit.agents.scenarios import make_context, standard_triple

        # build the series directly (integration at the context level)
        down = linear_trend_closes(300, slope=-2.0, seed=12)
        series = standard_triple(h1_closes=down, h1_seed=12)
        context = make_context(series)
        assessment = RegimeDetector().detect(context)
        context = context.with_regime(assessment)
        results = AgentRegistry().run_all(context)
        output = synthesize(
            SynthesisInput(
                symbol=context.symbol,
                timestamp=context.created_at,
                regime=assessment,
                agent_results=results,
            )
        )
        assert output.action is DecisionAction.SELL
        all_stances = {s.agent: s for s in output.supporting + output.opposing + output.neutral + output.disabled}
        # every agent appears exactly once somewhere
        assert len(all_stances) == 7
        stance_map = output.model_dump(mode="json")
        assert stance_map["conflicts"] is not None

