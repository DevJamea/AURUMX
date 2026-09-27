"""Risk state input for the decision engine (Phase-3 §15/§16).

A pure snapshot of account/limit state supplied BY THE CALLER — the engine
never fetches it from a broker and never mutates it.  This is the interface
the future worker (Phase 4+) fills from the account; backtests supply it
directly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RiskState(BaseModel):
    """Everything the engine needs to know about current risk occupancy.

    All fields default to "unknown / nothing open" so partial states are
    representable; the engine treats unknown equity as a safety ABORT.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    #: account equity in account currency (required for sizing)
    equity: float | None = Field(default=None, ge=0)
    #: realized + unrealized loss accumulated today (monetary, positive = loss)
    daily_loss: float = Field(default=0.0)
    #: explicit monetary daily loss limit; None -> derived from equity & config
    daily_loss_limit: float | None = Field(default=None, ge=0)
    #: consecutive closed losing trades so far
    consecutive_losses: int = Field(default=0, ge=0)
    #: currently open positions (count)
    open_positions: int = Field(default=0, ge=0)
    #: currently pending orders (count)
    pending_orders: int = Field(default=0, ge=0)
    #: active setup fingerprints (anti-overtrading; supplied by the journal /
    #: execution layer, never derived from wall-clock time)
    active_setup_fingerprints: frozenset[str] = Field(default_factory=frozenset)

    @property
    def has_equity(self) -> bool:
        return self.equity is not None and self.equity > 0


class AccountState(BaseModel):
    """Execution-side account + safety state for the Phase-4 RiskGate.

    Pure snapshot supplied BY THE CALLER (the worker/execution layer fills
    it from the broker account; backtests supply it directly) — the gate
    never fetches it itself.  Together with ``RiskState`` and the
    ``TradeProposal`` this is the complete input triple of the Phase-4
    safety barrier: enough information to independently reject a dangerous
    proposal (margin, exposure, halts, account sanity) without trusting
    anything the decision layer concluded.
    """

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    balance: float = Field(default=0.0)
    equity: float = Field(default=0.0)
    margin: float = Field(default=0.0)
    margin_free: float = Field(default=0.0)
    margin_level: float | None = Field(default=None)
    leverage: int = Field(default=100, ge=1)
    currency: str = "USD"
    #: currently open positions (count; cross-checked against RiskState)
    open_positions: int = Field(default=0, ge=0)
    #: currently pending orders (count; cross-checked against RiskState)
    pending_orders: int = Field(default=0, ge=0)
    #: monetary value of open positions (total exposure input)
    open_positions_notional: float = Field(default=0.0, ge=0)
    #: account/terminal allows trading at all (trade mode, connection state)
    trade_allowed: bool = True
    #: persistent operator emergency stop is active (blocks everything)
    emergency_stop_active: bool = False
    #: global kill switch is active (config-level halt, blocks everything)
    kill_switch_active: bool = False

    @property
    def has_sane_equity(self) -> bool:
        """Equity present and positive (a gate input sanity primitive)."""
        return self.equity > 0
