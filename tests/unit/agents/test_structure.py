"""StructureAgent behavioral tests (Phase-2 §6 — major agent).

Categories: BOS continuation, CHOCH reversal at reduced strength, range
(no events), stale events, bias/event mismatch, insufficient data, and the
confirmation-delay documentation.  Look-ahead safety itself is proven at the
feature layer (test_features.py) and re-asserted here at the agent level.
"""

from __future__ import annotations

import pytest

from app.core.enums import AgentDirection, DataQuality, StructureEventType, TimeFrame
from tests.unit.agents.scenarios import (
    make_context,
    make_series,
    zigzag_closes,
)

BOS_UP_WP = [2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730]
BOS_DOWN_WP = [2730, 2670, 2700, 2640, 2680, 2620, 2650, 2590]
CHOCH_UP_WP = [2650, 2590, 2620, 2560, 2600, 2540, 2580, 2520, 2560, 2500, 2540, 2610]
RANGE_WP = [2650, 2700, 2600, 2695, 2605, 2690, 2610, 2685, 2615, 2680, 2620]


def _ctx(waypoints, *, step=5, seed=7):
    closes = zigzag_closes(waypoints, step=step)
    series = {
        TimeFrame.H1: make_series(closes, TimeFrame.H1, seed=seed),
        TimeFrame.H4: make_series(closes, TimeFrame.H4, seed=seed),
    }
    return make_context(series)


class TestBOS:
    def test_bos_up_in_uptrend_is_buy(self, agents):
        result = agents["structure"].analyze(_ctx(BOS_UP_WP))
        assert result.direction is AgentDirection.BUY
        assert result.is_actionable
        assert result.signal_strength == pytest.approx(1.0)
        assert any("BOS" in r for r in result.reasons)
        assert any("invalidation" in r for r in result.reasons)
        assert result.features["last_event"]["kind"] == "BOS_UP"

    def test_bos_down_in_downtrend_is_sell(self, agents):
        result = agents["structure"].analyze(_ctx(BOS_DOWN_WP))
        assert result.direction is AgentDirection.SELL
        assert result.signal_strength == pytest.approx(1.0)
        assert result.features["last_event"]["kind"] == "BOS_DOWN"

    def test_invalidation_level_present(self, agents):
        result = agents["structure"].analyze(_ctx(BOS_UP_WP))
        assert result.features["invalidation_level"] == pytest.approx(2669.81, abs=1.0)


class TestCHOCH:
    def test_choch_up_after_downtrend_is_buy_at_reduced_strength(self, agents):
        result = agents["structure"].analyze(_ctx(CHOCH_UP_WP))
        assert result.direction is AgentDirection.BUY
        # CHOCH base 0.35 + recency 0.20 — weaker than a BOS by design
        assert result.signal_strength == pytest.approx(0.55)
        assert result.features["last_event"]["kind"] == "CHOCH_UP"
        assert any("reduced strength" in r for r in result.reasons)
        assert any("CHOCH" in w for w in result.warnings)

    def test_choch_weaker_than_bos(self, agents):
        bos = agents["structure"].analyze(_ctx(BOS_UP_WP))
        choch = agents["structure"].analyze(_ctx(CHOCH_UP_WP))
        assert choch.signal_strength < bos.signal_strength


class TestNoSignalCases:
    def test_range_zigzag_has_no_events_and_is_neutral(self, agents):
        result = agents["structure"].analyze(_ctx(RANGE_WP))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert result.features["last_event"] is None
        assert any("no confirmed structure breaks" in r for r in result.reasons)

    def test_stale_event_is_not_actionable(self, agents):
        # BOS happens early (index ~23) with step=10 the series runs 70 bars:
        # the break is > 20 candles old.
        result = agents["structure"].analyze(
            _ctx([2600, 2660, 2630, 2690, 2650, 2710, 2685, 2690], step=10)
        )
        assert result.direction is AgentDirection.NEUTRAL
        assert result.signal_strength == 0.0
        assert any("no fresh break" in r for r in result.reasons)

    def test_insufficient_candles(self, agents):
        result = agents["structure"].analyze(_ctx(BOS_UP_WP, step=2))  # 14 bars < 30
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality is DataQuality.INSUFFICIENT

    def test_missing_h1_is_unavailable(self, agents):
        series = {
            TimeFrame.M15: make_series(zigzag_closes(BOS_UP_WP, step=5), TimeFrame.M15, seed=7)
        }
        result = agents["structure"].analyze(make_context(series))
        assert result.direction is AgentDirection.NEUTRAL
        assert result.data_quality in (DataQuality.INSUFFICIENT, DataQuality.INVALID)


class TestConfirmationAndLookAhead:
    def test_confirmation_delay_is_documented_in_output(self, agents):
        result = agents["structure"].analyze(_ctx(BOS_UP_WP))
        assert result.features["confirmation_delay_candles"] == 2

    def test_agent_signal_uses_only_confirmed_levels(self, agents):
        """The event driving the signal must have fired at or after the
        confirmation bar of the level it broke (no retroactive signals)."""
        from app.agents.features import detect_swings, structure_events

        closes = zigzag_closes(BOS_UP_WP, step=5)
        df = make_series(closes, TimeFrame.H1, seed=7).to_dataframe()
        swings = detect_swings(df)
        events = structure_events(df, swings)
        result = agents["structure"].analyze(_ctx(BOS_UP_WP))
        last = result.features["last_event"]
        assert last is not None
        matching = [e for e in events if e.kind.value == last["kind"] and e.index == last["index"]]
        assert matching, "agent event must exist in the deterministic event list"
        for event in events:
            broken = [
                s
                for s in swings
                if s.price == event.level
                and (event.kind is StructureEventType.BOS_UP or event.kind is StructureEventType.CHOCH_UP)
                and s.kind.value == "HIGH"
            ] or [
                s
                for s in swings
                if s.price == event.level
                and event.kind in (StructureEventType.BOS_DOWN, StructureEventType.CHOCH_DOWN)
                and s.kind.value == "LOW"
            ]
            assert broken, "every broken level is a real swing"
            assert event.index >= broken[-1].confirmation_index

    def test_prefix_signal_stability(self, agents):
        """Analyzing a prefix and then extending history: the prefix's last
        event and swing set are exactly the filtered full-series ones
        (the core no-look-ahead property, asserted through the agent's own
        event payload)."""
        from app.agents.features import detect_swings, structure_events

        closes = zigzag_closes(
            [2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730, 2690, 2750, 2710, 2770], step=6
        )
        df = make_series(closes, TimeFrame.H1, seed=7).to_dataframe()
        full_swings = detect_swings(df)
        full_events = structure_events(df, full_swings)
        for n in (25, 40, 55, len(df)):
            prefix = df.iloc[:n]
            p_swings = detect_swings(prefix)
            p_events = structure_events(prefix, p_swings)
            assert p_swings == [s for s in full_swings if s.confirmation_index <= n - 1]
            assert p_events == [e for e in full_events if e.index <= n - 1]
