"""The broker contract (spec §29).

``BrokerInterface`` fixes the surface every broker implementation must provide.
The trading core (agents, decision, risk, execution service) depends on this
interface only — never on ``MetaTrader5`` — so the execution backend can be
replaced (different MT5 bridge, paper broker, another platform) without
touching any other layer.

Phase-1 scope is the read-only half (lifecycle, account, symbols, market data,
portfolio queries).  The execution half is *declared* here to fix the contract,
but the MT5 implementation is a hard stub until Phase 5 — by design, nothing
built on Phases 1–4 can place an order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.core.enums import TimeFrame
from app.core.models import (
    AccountSnapshot,
    CandleSeries,
    MarketOrderRequest,
    MarketTick,
    OrderResult,
    PendingOrder,
    PendingOrderRequest,
    Position,
    SymbolSpec,
)


class BrokerInterface(ABC):
    """Abstract broker.  All methods are blocking; implementors must be
    thread-safe (the worker and the API query the same instance)."""

    #: short name used in logs / UI ("mt5", "paper", ...)
    name: str = "base"

    # ---- lifecycle ------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Establish the connection.  Raises a ``BrokerError`` subclass on failure."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection.  Never raises for an already-closed broker."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the broker connection is currently usable."""

    # ---- account & symbols (read) ---------------------------------------
    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """Current account snapshot (never contains credentials)."""

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """All symbol names known to the broker (visibility varies)."""

    @abstractmethod
    def get_symbol(self, name: str) -> SymbolSpec | None:
        """Broker metadata for one symbol, or ``None`` if unknown."""

    @abstractmethod
    def select_symbol(self, name: str) -> bool:
        """Make the symbol visible/selected in the broker terminal (safe,
        non-trading operation)."""

    # ---- market data (read) ---------------------------------------------
    @abstractmethod
    def get_tick(self, symbol: str) -> MarketTick | None:
        """Latest quote, or ``None`` when unavailable."""

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        """Candles oldest->newest.  By default the still-forming bar is
        excluded so analysis only ever sees closed bars."""

    # ---- portfolio (read) -------------------------------------------------
    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Open positions (optionally filtered by symbol)."""

    @abstractmethod
    def get_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        """Pending orders (optionally filtered by symbol)."""

    # ---- execution (Phase 5) ----------------------------------------------
    # Declared now to fix the contract.  Implementations that do not support
    # execution yet must raise ExecutionNotImplementedError.
    @abstractmethod
    def place_market_order(self, request: MarketOrderRequest) -> OrderResult: ...

    @abstractmethod
    def place_pending_order(self, request: PendingOrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_position(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: int, *, volume: float | None = None) -> OrderResult:
        """Close fully (volume=None) or partially (spec §26)."""

    @abstractmethod
    def partial_close(self, ticket: int, volume: float) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, ticket: int) -> OrderResult: ...

    # ---- optional services -------------------------------------------------
    def server_time(self) -> datetime | None:
        """Broker server time if the implementation can provide it."""
        return None

    def is_trading_allowed(self) -> bool | None:
        """Whether the terminal/account allows trading right now (e.g. MT5
        AutoTrading enabled).  ``None`` = unknown — consumers must treat
        unknown as NOT allowed (fail-closed evidence, spec §54)."""
        return None
