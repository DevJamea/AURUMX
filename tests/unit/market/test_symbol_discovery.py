"""Symbol discovery tests: gold pattern matching and broker verification (spec §7, §55)."""

from __future__ import annotations

import pytest

from app.brokers.mt5 import MT5Broker
from app.core.exceptions import (
    SymbolDiscoveryError,
    SymbolNotGoldError,
    SymbolVerificationError,
)
from app.market.symbol_discovery import SymbolDiscovery, gold_match_score, is_gold_symbol
from tests.fakes.mt5_fake import FakeMT5, default_symbol


class TestGoldMatching:
    @pytest.mark.parametrize(
        ("name", "expected_score"),
        [
            ("XAUUSD", 100),
            ("xauusd", 100),          # case-insensitive
            ("XAU/USD", 90),
            ("XAUUSDm", 85),          # exness-style suffix
            ("XAUUSD.a", 85),
            ("XAUUSD.raw", 85),
            ("XAUUSD-ECN", 85),
            ("XAUUSD.1", 85),
            ("XAUUSD+", 85),
            ("GOLD", 80),
            ("GOLDm", 70),
            ("GOLD.raw", 70),
            ("GOLDmicro", 70),
            ("GOLD-M", 70),
        ],
    )
    def test_gold_names_score(self, name: str, expected_score: int):
        assert gold_match_score(name) == expected_score

    @pytest.mark.parametrize(
        "name",
        [
            "XAGUSD",       # silver — never gold
            "XAGUSDm",
            "XPTUSD",       # platinum
            "XPDUSD",       # palladium
            "XAUEUR",       # gold but not USD-quoted
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "GC=F",         # futures ticker
            "GLD",          # ETF
            "GOLDEN",       # must NOT strip down to GOLD
            "XAUUSDGIBBERISH",
            "SILVER",
            "SILVERmicro",
            "USDGOLD",
            "",
            "   ",
        ],
    )
    def test_non_gold_names_score_zero(self, name: str):
        assert gold_match_score(name) == 0
        assert is_gold_symbol(name) is False


class TestAutoDiscovery:
    def test_discovers_best_candidate(self, broker: MT5Broker):
        result = SymbolDiscovery(broker).discover("AUTO")
        assert result.chosen.name == "XAUUSD"
        assert result.method == "auto"
        assert result.score == 100

    def test_alternatives_and_rejections_are_recorded(self, broker: MT5Broker):
        result = SymbolDiscovery(broker).discover("AUTO")
        assert {s.name for s in result.alternatives} >= {"XAUUSDm", "GOLD"}
        rejected = {r.name: r.reasons for r in result.rejected}
        assert "XAUUSD.bad" in rejected
        assert rejected["XAUUSD.bad"]  # reasons are non-empty

    def test_falls_back_when_best_candidate_is_broken(self, fake_mt5: FakeMT5):
        broken = fake_mt5.symbols["XAUUSD"]._replace(point=0.0)
        fake_mt5.symbols["XAUUSD"] = broken
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        result = SymbolDiscovery(broker).discover("AUTO")

        assert result.chosen.name == "XAUUSDm"
        rejected_names = {r.name for r in result.rejected}
        assert "XAUUSD" in rejected_names

    def test_raises_when_no_gold_symbol_exists(self):
        fake = FakeMT5(symbols=[default_symbol("EURUSD", point=0.00001, digits=5)])
        broker = MT5Broker(mt5_module=fake)
        broker.connect()

        with pytest.raises(SymbolDiscoveryError, match="No gold symbol"):
            SymbolDiscovery(broker).discover("AUTO")

    def test_raises_when_all_candidates_fail_verification(self):
        fake = FakeMT5(
            symbols=[
                default_symbol("XAUUSD", point=0.0),
                default_symbol("GOLD", point=0.0),
            ]
        )
        broker = MT5Broker(mt5_module=fake)
        broker.connect()

        with pytest.raises(SymbolVerificationError):
            SymbolDiscovery(broker).discover("AUTO")

    def test_invisible_gold_symbol_is_selected_then_verified(self):
        fake = FakeMT5(
            symbols=[default_symbol("XAUUSD", visible=False, selected=False)]
        )
        broker = MT5Broker(mt5_module=fake)
        broker.connect()

        result = SymbolDiscovery(broker).discover("AUTO")

        assert result.chosen.name == "XAUUSD"
        assert result.chosen.visible is True  # selected during discovery

    def test_disabled_symbol_fails_verification(self, fake_mt5: FakeMT5):
        from tests.fakes.mt5_fake import SYMBOL_TRADE_MODE_DISABLED

        fake_mt5.symbols["XAUUSD"] = fake_mt5.symbols["XAUUSD"]._replace(
            trade_mode=SYMBOL_TRADE_MODE_DISABLED
        )
        broker = MT5Broker(mt5_module=fake_mt5)
        broker.connect()

        result = SymbolDiscovery(broker).discover("AUTO")
        assert result.chosen.name == "XAUUSDm"  # XAUUSD rejected (trading disabled)


class TestExplicitDiscovery:
    def test_explicit_gold_symbol(self, broker: MT5Broker):
        result = SymbolDiscovery(broker).discover("XAUUSDm")
        assert result.chosen.name == "XAUUSDm"
        assert result.method == "explicit"

    def test_explicit_non_gold_symbol_is_rejected(self, broker: MT5Broker):
        with pytest.raises(SymbolNotGoldError, match="gold"):
            SymbolDiscovery(broker).discover("EURUSD")

    def test_explicit_silver_is_rejected(self, broker: MT5Broker):
        with pytest.raises(SymbolNotGoldError):
            SymbolDiscovery(broker).discover("XAGUSD")

    def test_explicit_unknown_symbol_raises(self, broker: MT5Broker):
        with pytest.raises(SymbolDiscoveryError, match="not found"):
            SymbolDiscovery(broker).discover("XAUUSD.zzz")

    def test_explicit_broken_symbol_raises_verification(self, broker: MT5Broker):
        with pytest.raises(SymbolVerificationError):
            SymbolDiscovery(broker).discover("XAUUSD.bad")
