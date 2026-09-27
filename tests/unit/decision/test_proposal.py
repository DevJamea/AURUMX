"""TradeProposal tests (Phase-3 §9/§17/§18): pure data, geometry, expiry."""

from __future__ import annotations

from datetime import timedelta

from app.core.enums import AgentDirection, TimeFrame
from app.decision import TickEntryProvider, TradeProposal
from app.decision.proposal import DEFAULT_INVALIDATION_CONDITIONS
from app.market.market_state import MarketSnapshot  # noqa: F401  (type docs)
from app.risk import RiskSizing
from tests.unit.agents.scenarios import REF_TIME, gold_symbol_spec, make_tick


def _sizing() -> RiskSizing:
    return RiskSizing(
        equity=10_000.0, risk_per_trade_pct=0.5, risk_amount=50.0,
        risk_distance=5.0, loss_per_lot=500.0, raw_volume=0.10,
        normalized_volume=0.10, suggested_volume=0.10,
        clamped_to_max=False, monetary_risk=50.0, percentage_risk=0.5,
        feasible=True,
    )


def _proposal(**overrides) -> TradeProposal:
    base = dict(
        symbol="XAUUSD",
        direction=AgentDirection.BUY,
        entry_price=2650.20,
        stop_loss=2645.20,
        take_profit=2660.20,
        risk_distance=5.0,
        reward_distance=10.0,
        risk_reward=2.0,
        suggested_volume=0.10,
        max_allowed_volume=100.0,
        timeframe=TimeFrame.H1,
        regime="TREND_UP",
        decision_id="abc123",
        setup_type="trend",
        fingerprint="fp0123456789abcdef",
        reasons=["trend up", "momentum up"],
        invalidation_conditions=list(DEFAULT_INVALIDATION_CONDITIONS),
        created_at=REF_TIME,
        expires_at=REF_TIME + timedelta(minutes=15),
        sl_source="atr",
        tp_source="rr_target",
        sizing=_sizing(),
    )
    base.update(overrides)
    return TradeProposal(**base)


class TestPureData:
    def test_carries_every_required_field(self):
        p = _proposal()
        for field in (
            "symbol", "direction", "entry_price", "stop_loss", "take_profit",
            "risk_reward", "suggested_volume", "max_allowed_volume", "timeframe",
            "regime", "decision_id", "reasons", "invalidation_conditions",
            "created_at", "expires_at",
        ):
            assert getattr(p, field) is not None

    def test_has_no_side_channels(self):
        """A TradeProposal is a pydantic model — no sockets, no broker handles,
        no order methods anywhere in its source."""
        import inspect

        src = inspect.getsource(type(_proposal()))
        assert "order_send" not in src
        assert "mt5" not in src.lower().replace("mt5_demo", "")
        assert "requests" not in src

    def test_summary_covers_the_trade(self):
        s = _proposal().summary()
        assert s["symbol"] == "XAUUSD"
        assert s["direction"] == "BUY"
        assert s["entry"] == 2650.2
        assert s["decision_id"] == "abc123"
        assert s["setup_type"] == "trend"


SYMBOL = gold_symbol_spec()


class TestGeometry:
    def test_buy_geometry_valid(self):
        assert _proposal().geometry_valid(SYMBOL)

    def test_sl_on_wrong_side_invalidates(self):
        assert not _proposal(stop_loss=2655.0).geometry_valid(SYMBOL)

    def test_tp_on_wrong_side_invalidates(self):
        assert not _proposal(take_profit=2640.0).geometry_valid(SYMBOL)

    def test_sell_geometry(self):
        p = _proposal(
            direction=AgentDirection.SELL, entry_price=2650.0,
            stop_loss=2655.0, take_profit=2640.0,
        )
        assert p.geometry_valid(SYMBOL)
        assert not _proposal(
            direction=AgentDirection.SELL, entry_price=2650.0,
            stop_loss=2645.0, take_profit=2640.0,
        ).geometry_valid(SYMBOL)

    def test_below_broker_minimum_invalidates(self):
        assert not _proposal(
            entry_price=2650.0, stop_loss=2649.9, take_profit=2660.0
        ).geometry_valid(SYMBOL)

    def test_off_tick_grid_invalidates(self):
        assert not _proposal(stop_loss=2645.205).geometry_valid(SYMBOL)

    def test_zero_volume_fails_engine_gate_not_geometry(self):
        """Volume feasibility is a separate engine gate (position_size_below_minimum);
        geometry only checks entry/SL/TP shape."""
        assert _proposal(suggested_volume=0.0).geometry_valid(SYMBOL)


class TestExpiry:
    def test_valid_before_expiry(self):
        assert _proposal().status(REF_TIME + timedelta(minutes=14)) == "VALID"

    def test_valid_at_expiry_boundary_inclusive(self):
        # the proposal is actionable through its expiry instant (documented)
        p = _proposal()
        assert p.status(p.expires_at) == "VALID"
        assert p.status(p.expires_at + timedelta(microseconds=1)) == "EXPIRED"

    def test_expired_after(self):
        assert _proposal().status(REF_TIME + timedelta(hours=1)) == "EXPIRED"

    def test_expiry_is_explicit_no_global_state(self):
        """Two proposals created together expire independently of any clock."""
        p1 = _proposal()
        p2 = _proposal(expires_at=REF_TIME + timedelta(hours=1))
        at = REF_TIME + timedelta(minutes=30)
        assert p1.status(at) == "EXPIRED"
        assert p2.status(at) == "VALID"


class TestInvalidationConditions:
    def test_default_conditions_cover_required_cases(self):
        text = " ".join(DEFAULT_INVALIDATION_CONDITIONS).lower()
        assert "structure" in text
        assert "spread" in text
        assert "regime" in text
        assert "expire" in text
        assert "data" in text

    def test_conditions_are_persisted_on_every_proposal(self):
        assert len(_proposal().invalidation_conditions) >= 5


class TestEntryProvider:
    def test_buy_uses_ask(self):
        tick = make_tick()
        assert TickEntryProvider().entry_price(AgentDirection.BUY, tick) == tick.ask

    def test_sell_uses_bid(self):
        tick = make_tick()
        assert TickEntryProvider().entry_price(AgentDirection.SELL, tick) == tick.bid

    def test_never_uses_close(self):
        """Live entry is the validated tick, never a historical close."""
        tick = make_tick()
        for direction in (AgentDirection.BUY, AgentDirection.SELL):
            price = TickEntryProvider().entry_price(direction, tick)
            assert price in (tick.bid, tick.ask)
