"""Feature-layer tests: indicators, swings, structure events, look-ahead safety.

These pin the deterministic primitives every agent builds on.  The look-ahead
test is the critical one (Phase-2 §6): adding future candles must never change
a signal that was already derived from a prefix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.features import (
    SWING_RIGHT,
    bollinger,
    compute_features,
    detect_swings,
    directional_index,
    ema,
    label_swings,
    macd,
    rsi,
    structure_bias,
    structure_events,
    true_range,
)
from app.core.enums import StructureEventType, SwingKind, TimeFrame
from tests.unit.agents.scenarios import (
    flat_closes,
    linear_trend_closes,
    make_series,
    sideways_closes,
    zigzag_closes,
)

# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------


def test_ema_matches_hand_computed_value():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = ema(close, 3)
    # span=3 -> alpha=0.5; warm-up leaves the first 2 as NaN
    assert out.iloc[:2].isna().all()
    # pandas seeds the EMA with the first value, then recurses:
    assert out.iloc[2] == pytest.approx(2.25)
    assert out.iloc[3] == pytest.approx(3.125)
    assert out.iloc[6] == pytest.approx(6.015625)


def test_ema_constant_series_is_constant():
    out = ema(pd.Series([2650.0] * 50), 20)
    assert out.iloc[19:].notna().all()
    assert (out.iloc[19:] - 2650.0).abs().max() < 1e-9


def test_rsi_all_gains_is_100_all_losses_is_0():
    up = pd.Series(linear_trend_closes(60, slope=2.0, noise=0.0))
    down = pd.Series(linear_trend_closes(60, slope=-2.0, noise=0.0))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_series_is_50():
    out = rsi(pd.Series(flat_closes(60)), 14)
    assert out.iloc[-1] == pytest.approx(50.0)


def test_macd_signs_and_histogram_semantics():
    close = pd.Series(linear_trend_closes(300, slope=2.0, noise=0.0))
    line, signal, hist = macd(close)
    # steady uptrend: line positive, and (line - signal) is the acceleration
    assert line.iloc[-1] > 0
    # exponential growth accelerates -> histogram positive
    close_exp = pd.Series(2650.0 * 1.002 ** np.arange(300))
    _, _, hist_exp = macd(close_exp)
    assert hist_exp.iloc[-1] > 0


def test_adx_flat_market_is_low_trend_is_high():
    flat = make_series(flat_closes(300), TimeFrame.H1, seed=7)
    flat_df = flat.to_dataframe()
    adx_flat, di_p, di_m = directional_index(
        flat_df["high"], flat_df["low"], flat_df["close"]
    )
    assert adx_flat.iloc[-1] < 20
    # flat closes: both DIs small, neither dominates
    assert di_p.iloc[-1] < 25 and di_m.iloc[-1] < 25

    trend = make_series(linear_trend_closes(300, slope=2.0), TimeFrame.H1, seed=2)
    trend_df = trend.to_dataframe()
    adx_trend, plus_di, minus_di = directional_index(
        trend_df["high"], trend_df["low"], trend_df["close"]
    )
    assert adx_trend.iloc[-1] > 40
    assert plus_di.iloc[-1] > minus_di.iloc[-1]

    downtrend = make_series(linear_trend_closes(300, slope=-2.0), TimeFrame.H1, seed=3)
    down_df = downtrend.to_dataframe()
    _, plus_d, minus_d = directional_index(down_df["high"], down_df["low"], down_df["close"])
    assert minus_d.iloc[-1] > plus_d.iloc[-1]


def test_adx_no_nan_on_constant_series():
    df = make_series(flat_closes(300), TimeFrame.H1, seed=4).to_dataframe()
    adx, plus, minus = directional_index(df["high"], df["low"], df["close"])
    tail = slice(-50, None)
    assert not adx.iloc[tail].isna().any()
    assert not plus.iloc[tail].isna().any()
    assert not minus.iloc[tail].isna().any()
    assert np.isfinite(adx.iloc[-1])


def test_bollinger_zero_width_gives_mid_percent_b():
    close = pd.Series(flat_closes(60))
    lower, mid, upper, width, percent_b = bollinger(close, 20, 2.0)
    assert percent_b.iloc[-1] == pytest.approx(0.5)  # documented convention


def test_true_range_uses_max_of_three():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 10.0])
    close = pd.Series([9.5, 11.0])
    tr = true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(max(12.0 - 10.0, abs(12.0 - 9.5), abs(10.0 - 9.5)))


def test_rolling_extremes_exclude_last():
    series = make_series(sideways_closes(60, seed=5), TimeFrame.H1, seed=5)
    f = compute_features(series)
    low, high = f.rolling_extremes(window=20, exclude_last=1)
    # excludes the final candle: equals min/max over rows [-21:-1]
    assert low == pytest.approx(f.df["low"].iloc[-21:-1].min())
    assert high == pytest.approx(f.df["high"].iloc[-21:-1].max())
    low_all, high_all = f.rolling_extremes(window=20)
    assert low_all == pytest.approx(f.df["low"].iloc[-20:].min())
    assert high_all == pytest.approx(f.df["high"].iloc[-20:].max())


def test_compute_features_short_series_is_safe():
    series = make_series(linear_trend_closes(25), TimeFrame.H1, seed=6)
    f = compute_features(series)
    assert f.length == 25
    # nothing may raise; indicators simply stay None until warmed up
    assert f.ema200 is None
    assert f.rsi is not None  # 25 > 14


# ---------------------------------------------------------------------------
# swings
# ---------------------------------------------------------------------------


def test_swing_confirmation_delay_is_two_candles():
    """A fractal at bar i is only reported from bar i+2 onward (documented)."""
    closes = zigzag_closes([2600, 2660, 2630, 2690, 2650, 2710], step=5)
    series = make_series(closes, TimeFrame.H1, seed=7)
    df = series.to_dataframe()

    full_swings = detect_swings(df)
    assert full_swings, "zigzag must produce swings"
    for swing in full_swings:
        assert swing.confirmation_index == swing.index + SWING_RIGHT

    # prefix cut BEFORE confirmation: swing must not appear
    for swing in full_swings:
        prefix = df.iloc[: swing.confirmation_index]  # ends one bar too early
        prefix_swings = detect_swings(prefix)
        assert swing not in prefix_swings
        # prefix ending exactly at the confirmation bar DOES contain it
        ok_prefix = df.iloc[: swing.confirmation_index + 1]
        assert swing in detect_swings(ok_prefix)


def test_swing_prominence_filter_rejects_wick_jitter():
    """Sine legs: real swings kept; wick-level jitter filtered."""
    series = make_series(sideways_closes(300, amplitude=1.2, seed=71), TimeFrame.H1, seed=71)
    swings = detect_swings(series.to_dataframe())
    # p=16 sine over 300 bars has ~18 real alternations; jitter would add dozens
    assert 10 <= len(swings) <= 45
    highs = [s for s in swings if s.kind is SwingKind.HIGH]
    lows = [s for s in swings if s.kind is SwingKind.LOW]
    assert len(highs) >= 8 and len(lows) >= 8


def test_structure_bias_labels_uptrend_and_downtrend():
    up = zigzag_closes([2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730], step=5)
    swings = label_swings(detect_swings(make_series(up, TimeFrame.H1, seed=7).to_dataframe()))
    assert structure_bias(swings).value == "UPTREND"

    down = zigzag_closes([2730, 2670, 2700, 2640, 2680, 2620, 2650, 2590], step=5)
    swings = label_swings(detect_swings(make_series(down, TimeFrame.H1, seed=7).to_dataframe()))
    assert structure_bias(swings).value == "DOWNTREND"


def test_structure_bias_mixed_is_range():
    # lower highs AND higher lows -> converging -> RANGE
    wp = [2650, 2700, 2600, 2695, 2605, 2690, 2610, 2685, 2615, 2680, 2620]
    closes = zigzag_closes(wp, step=5)
    swings = label_swings(detect_swings(make_series(closes, TimeFrame.H1, seed=7).to_dataframe()))
    assert structure_bias(swings).value == "RANGE"


# ---------------------------------------------------------------------------
# structure events + LOOK-AHEAD SAFETY
# ---------------------------------------------------------------------------


def test_structure_events_bos_and_choch_kinds():
    up = zigzag_closes([2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730], step=5)
    df = make_series(up, TimeFrame.H1, seed=7).to_dataframe()
    events = structure_events(df, detect_swings(df))
    assert events, "uptrend zigzag must break swing highs"
    kinds = {StructureEventType.BOS_UP, StructureEventType.BOS_DOWN,
             StructureEventType.CHOCH_UP, StructureEventType.CHOCH_DOWN}
    assert all(e.kind in kinds for e in events)
    # in a clean uptrend every break is a bullish BOS
    assert all(e.kind is StructureEventType.BOS_UP for e in events)

    # downtrend that reverses at the end: the final break is a bullish CHOCH
    rev = zigzag_closes(
        [2650, 2590, 2620, 2560, 2600, 2540, 2580, 2520, 2560, 2500, 2540, 2610], step=5
    )
    df2 = make_series(rev, TimeFrame.H1, seed=7).to_dataframe()
    events2 = structure_events(df2, detect_swings(df2))
    assert events2[0].kind is StructureEventType.BOS_DOWN
    assert events2[-1].kind is StructureEventType.CHOCH_UP


def test_look_ahead_prefix_consistency():
    """CORE SAFETY PROPERTY: extending history with future candles cannot
    change any swing or structure event derived from an earlier prefix."""
    closes = zigzag_closes(
        [2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730, 2690, 2750, 2710, 2770], step=6
    )
    df = make_series(closes, TimeFrame.H1, seed=7).to_dataframe()
    full_swings = detect_swings(df)
    full_events = structure_events(df, full_swings)

    for n in range(20, len(df) + 1, 7):  # a spread of prefix lengths
        prefix = df.iloc[:n]
        p_swings = detect_swings(prefix)
        p_events = structure_events(prefix, p_swings)

        expected_swings = [s for s in full_swings if s.confirmation_index <= n - 1]
        assert p_swings == expected_swings, f"swing divergence at prefix length {n}"

        expected_events = [e for e in full_events if e.index <= n - 1]
        assert p_events == expected_events, f"event divergence at prefix length {n}"


def test_no_event_before_its_level_is_confirmed():
    """A break can only be recorded after the broken level's confirmation bar."""
    closes = zigzag_closes([2600, 2660, 2630, 2690, 2650, 2710, 2670, 2730], step=5)
    df = make_series(closes, TimeFrame.H1, seed=7).to_dataframe()
    swings = detect_swings(df)
    events = structure_events(df, swings)
    swing_highs = [s for s in swings if s.kind is SwingKind.HIGH]
    for event in events:
        if event.kind in (StructureEventType.BOS_UP, StructureEventType.CHOCH_UP):
            broken = [s for s in swing_highs if s.price == event.level]
            assert broken, "broken level must be a confirmed swing high"
            assert event.index >= broken[-1].confirmation_index, (
                "break recorded before the level was confirmed — retroactive signal"
            )
