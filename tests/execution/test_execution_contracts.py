"""Execution contract tests (spec §7A.1/§7A.2, §12, §16, §33)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.core.enums import AgentDirection, Direction, TradingMode
from app.execution import (
    AURUMX_MAGIC,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStage,
    ExecutionStatus,
    derive_request_id,
)
from tests.execution.conftest import make_approved_decision
from tests.unit.risk.conftest import (
    make_account,  # noqa: F401  (fixture helper parity)
    make_proposal,
)


class TestExecutionRequest:
    def test_from_proposal_copies_verbatim(self):
        proposal = make_proposal()
        decision = make_approved_decision(proposal)
        request = ExecutionRequest.from_proposal(
            proposal, risk_decision_id=decision.gate_decision_id
        )
        # §16: symbol/direction/volume/SL/TP may not change
        assert request.symbol == proposal.symbol
        assert request.direction is Direction.LONG  # BUY -> LONG
        assert request.volume == proposal.suggested_volume
        assert request.entry_price == proposal.entry_price
        assert request.stop_loss == proposal.stop_loss
        assert request.take_profit == proposal.take_profit
        assert request.proposal_id == proposal.decision_id
        assert request.risk_decision_id == decision.gate_decision_id
        assert request.fingerprint == proposal.fingerprint

    def test_sell_maps_to_short(self):
        proposal = make_proposal(
            direction=AgentDirection.SELL,
            entry_price=2650.0, stop_loss=2655.0, take_profit=2640.0,
        )
        request = ExecutionRequest.from_proposal(
            proposal, risk_decision_id="gate123"
        )
        assert request.direction is Direction.SHORT

    def test_request_id_is_deterministic(self):
        proposal = make_proposal()
        r1 = ExecutionRequest.from_proposal(proposal, risk_decision_id="gate123")
        r2 = ExecutionRequest.from_proposal(proposal, risk_decision_id="gate123")
        assert r1.request_id == r2.request_id
        assert len(r1.request_id) == 16

    def test_request_id_changes_with_content(self):
        proposal = make_proposal()
        r1 = ExecutionRequest.from_proposal(proposal, risk_decision_id="gate123")
        r2 = ExecutionRequest.from_proposal(proposal, risk_decision_id="gate999")
        assert r1.request_id != r2.request_id
        other = ExecutionRequest.from_proposal(
            make_proposal(suggested_volume=0.10), risk_decision_id="gate123"
        )
        assert r1.request_id != other.request_id

    def test_missing_risk_decision_id_refused(self):
        proposal = make_proposal()
        with pytest.raises(ValueError, match="risk_decision_id"):
            ExecutionRequest.from_proposal(proposal, risk_decision_id="")

    def test_neutral_direction_refused(self):
        with pytest.raises(ValidationError, match="NEUTRAL"):
            ExecutionRequest(
                symbol="XAUUSD", direction=Direction.NEUTRAL, volume=0.1,
                entry_price=2650.0, stop_loss=2645.0, take_profit=2660.0,
                proposal_id="p1", risk_decision_id="g1", fingerprint="f1",
                request_id="r1",
            )

    @pytest.mark.parametrize("bad_volume", [0.0, -0.1, float("nan"), float("inf")])
    def test_invalid_volumes_refused(self, bad_volume):
        with pytest.raises(ValidationError):
            ExecutionRequest(
                symbol="XAUUSD", direction=Direction.LONG, volume=bad_volume,
                entry_price=2650.0, stop_loss=2645.0, take_profit=2660.0,
                proposal_id="p1", risk_decision_id="g1", fingerprint="f1",
                request_id="r1",
            )

    @pytest.mark.parametrize(
        "field", ["entry_price", "stop_loss", "take_profit"]
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_prices_refused(self, field, bad):
        kwargs = dict(
            symbol="XAUUSD", direction=Direction.LONG, volume=0.1,
            entry_price=2650.0, stop_loss=2645.0, take_profit=2660.0,
            proposal_id="p1", risk_decision_id="g1", fingerprint="f1",
            request_id="r1",
        )
        kwargs[field] = bad
        with pytest.raises(ValidationError):
            ExecutionRequest(**kwargs)

    def test_symbol_is_normalized(self):
        request = ExecutionRequest(
            symbol=" xauusd ", direction=Direction.LONG, volume=0.1,
            entry_price=2650.0, stop_loss=2645.0, take_profit=2660.0,
            proposal_id="p1", risk_decision_id="g1", fingerprint="f1",
            request_id="r1",
        )
        assert request.symbol == "XAUUSD"

    def test_magic_and_comment_defaults(self):
        proposal = make_proposal()
        request = ExecutionRequest.from_proposal(
            proposal, risk_decision_id="gate123"
        )
        assert request.magic == AURUMX_MAGIC
        assert request.comment.startswith("AURUMX ")
        assert len(request.comment) <= 31

    def test_matches_proposal_detects_every_change(self):
        """§16 consistency: any drift between proposal and request is a
        named violation."""
        proposal = make_proposal()
        request = ExecutionRequest.from_proposal(
            proposal, risk_decision_id="gate123"
        )
        assert request.matches_proposal(proposal) == []

        drifted = request.model_copy(update={"volume": request.volume + 0.01})
        assert any("volume changed" in i for i in drifted.matches_proposal(proposal))
        drifted = request.model_copy(update={"stop_loss": 2640.0})
        assert any("stop_loss changed" in i for i in drifted.matches_proposal(proposal))
        drifted = request.model_copy(update={"take_profit": 2670.0})
        assert any("take_profit changed" in i for i in drifted.matches_proposal(proposal))
        drifted = request.model_copy(update={"symbol": "XAUUSDm"})
        assert any("symbol changed" in i for i in drifted.matches_proposal(proposal))
        drifted = request.model_copy(update={"direction": Direction.SHORT})
        assert any("direction changed" in i for i in drifted.matches_proposal(proposal))


class TestExecutionResult:
    def _result(self, **overrides) -> ExecutionResult:
        base = dict(
            request_id="r1", proposal_id="p1", risk_decision_id="g1",
            symbol="XAUUSD", direction=Direction.LONG, volume=0.08,
            entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
            mode=TradingMode.DRY_RUN, status=ExecutionStatus.DRY_RUN,
            stage=ExecutionStage.SIMULATED,
        )
        base.update(overrides)
        return ExecutionResult(**base)

    def test_status_taxonomy_is_the_spec_set(self):
        assert [s.value for s in ExecutionStatus] == [
            "NOT_ATTEMPTED", "DRY_RUN", "CHECK_FAILED", "SEND_FAILED",
            "REJECTED_BY_BROKER", "ACCEPTED", "FILLED", "PARTIALLY_FILLED",
            "UNKNOWN",
        ]

    def test_only_verified_statuses_are_success(self):
        assert ExecutionStatus.FILLED.is_verified_success
        assert ExecutionStatus.PARTIALLY_FILLED.is_verified_success
        assert not ExecutionStatus.ACCEPTED.is_verified_success  # §20!
        assert not ExecutionStatus.UNKNOWN.is_verified_success
        assert not ExecutionStatus.DRY_RUN.is_verified_success

    def test_ok_requires_verification_or_dry_run(self):
        assert self._result().ok  # dry run
        assert self._result(
            status=ExecutionStatus.FILLED, stage=ExecutionStage.VERIFIED
        ).ok
        assert not self._result(
            status=ExecutionStatus.ACCEPTED, stage=ExecutionStage.SENT
        ).ok
        assert not self._result(
            status=ExecutionStatus.REJECTED_BY_BROKER, stage=ExecutionStage.SENT
        ).ok

    def test_summary_is_json_safe_and_complete(self):
        import json

        summary = self._result(
            order_ticket=None, deal_ticket=None, position_ticket=None
        ).summary()
        json.dumps(summary)
        for key in (
            "request_id", "proposal_id", "risk_decision_id", "symbol",
            "direction", "volume", "entry_price", "stop_loss", "take_profit",
            "mode", "status", "stage", "retcode", "order_ticket",
            "deal_ticket", "position_ticket", "fill_price", "filled_volume",
            "message", "timestamp",
        ):
            assert key in summary, key


class TestDeterminism:
    def test_derive_request_id_pure(self):
        a = derive_request_id(
            proposal_id="p", risk_decision_id="g", fingerprint="f",
            symbol="XAUUSD", direction="LONG", volume=0.08,
            entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
        )
        b = derive_request_id(
            proposal_id="p", risk_decision_id="g", fingerprint="f",
            symbol="XAUUSD", direction="LONG", volume=0.08,
            entry_price=2650.2, stop_loss=2645.2, take_profit=2660.2,
        )
        assert a == b
        assert all(c in "0123456789abcdef" for c in a)

    def test_magic_is_documented_value(self):
        # ASCII "AURX" big-endian, positive int32 — see contracts.py §43 note
        assert AURUMX_MAGIC == int.from_bytes(b"AURX", "big")
        assert 0 < AURUMX_MAGIC <= 2_147_483_647

    def test_no_floating_point_surprises_in_ids(self):
        """NaN/inf can never enter an id (model rejects them upstream)."""
        with pytest.raises(ValidationError):
            ExecutionRequest(
                symbol="XAUUSD", direction=Direction.LONG, volume=math.nan,
                entry_price=2650.0, stop_loss=2645.0, take_profit=2660.0,
                proposal_id="p", risk_decision_id="g", fingerprint="f",
                request_id="r",
            )
