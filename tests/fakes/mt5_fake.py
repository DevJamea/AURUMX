"""A configurable fake of the ``MetaTrader5`` python package.

Implements exactly the API subset ``MT5Broker`` uses, with the same semantics
(position 0 = current forming bar, epoch-second times, namedtuples, numpy
structured rate arrays).  This lets the whole system be tested deterministically
without a Windows terminal — and it will later power the paper-trading mode.

Determinism: bars are generated with a seeded random walk, and tests inject a
fixed clock, so every test run sees identical data.
"""

from __future__ import annotations

import random
from collections import namedtuple
from datetime import datetime
from typing import Any

import numpy as np

# ---- constants (same values as the real package) --------------------------
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408

TF_MINUTES: dict[int, int] = {
    TIMEFRAME_M1: 1,
    TIMEFRAME_M5: 5,
    TIMEFRAME_M15: 15,
    TIMEFRAME_M30: 30,
    TIMEFRAME_H1: 60,
    TIMEFRAME_H4: 240,
    TIMEFRAME_D1: 1440,
}

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2

SYMBOL_TRADE_MODE_DISABLED = 0
SYMBOL_TRADE_MODE_FULL = 4

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0

# trade retcodes (same values as the real package)
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_CANCEL = 10007
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_DONE_PARTIAL = 10010
TRADE_RETCODE_ERROR = 10011
TRADE_RETCODE_TIMEOUT = 10012
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_PRICE = 10015
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_TRADE_DISABLED = 10017
TRADE_RETCODE_MARKET_CLOSED = 10018
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_PRICE_CHANGED = 10020
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_INVALID_FILL = 10030
TRADE_RETCODE_CONNECTION = 10031

RATES_DTYPE = np.dtype(
    [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i8"),
        ("real_volume", "i8"),
    ]
)

FakeTick = namedtuple(
    "FakeTick", ["time", "bid", "ask", "last", "volume", "time_msc", "flags", "volume_real"]
)
FakeSymbolInfo = namedtuple(
    "FakeSymbolInfo",
    [
        "name", "custom", "select", "visible", "digits", "point",
        "trade_tick_size", "trade_tick_value", "trade_tick_value_profit",
        "trade_tick_value_loss", "trade_contract_size", "volume_min",
        "volume_max", "volume_step", "trade_stops_level", "trade_freeze_level",
        "trade_mode", "currency_profit", "currency_margin", "description", "path",
    ],
)
FakeAccountInfo = namedtuple(
    "FakeAccountInfo",
    [
        "login", "trade_mode", "leverage", "currency", "balance", "equity",
        "margin", "margin_free", "margin_level", "name", "server", "profit",
    ],
)
FakeTerminalInfo = namedtuple(
    "FakeTerminalInfo", ["connected", "trade_allowed", "name", "company", "maxbars"]
)
FakePosition = namedtuple(
    "FakePosition",
    [
        "ticket", "symbol", "type", "volume", "price_open", "price_current",
        "sl", "tp", "profit", "swap", "time", "comment", "magic",
    ],
)
FakeOrder = namedtuple(
    "FakeOrder",
    [
        "ticket", "symbol", "type", "volume_current", "price_open", "sl", "tp",
        "time_setup", "time_expiration", "comment", "magic",
    ],
)
FakeOrderCheckResult = namedtuple(
    "FakeOrderCheckResult",
    ["retcode", "balance", "equity", "margin", "margin_free", "profit", "comment"],
)
FakeOrderSendResult = namedtuple(
    "FakeOrderSendResult",
    [
        "retcode", "deal", "order", "volume", "price", "bid", "ask",
        "comment", "request_id", "retcode_external", "request",
    ],
)


def default_symbol(
    name: str = "XAUUSD",
    *,
    visible: bool = True,
    selected: bool = True,
    point: float = 0.01,
    digits: int = 2,
    trade_mode: int = SYMBOL_TRADE_MODE_FULL,
) -> FakeSymbolInfo:
    """A realistic XAUUSD symbol info (100 oz contract, 0.01 lot steps)."""
    return FakeSymbolInfo(
        name=name,
        custom=False,
        select=selected,
        visible=visible,
        digits=digits,
        point=point,
        trade_tick_size=point,
        trade_tick_value=1.0,
        trade_tick_value_profit=1.0,
        trade_tick_value_loss=1.0,
        trade_contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_stops_level=20,
        trade_freeze_level=10,
        trade_mode=trade_mode,
        currency_profit="USD",
        currency_margin="USD",
        description="Gold vs US Dollar",
        path=f"Metals\\{name}",
    )


def default_account(
    *, login: int = 12345678, trade_mode: int = ACCOUNT_TRADE_MODE_DEMO
) -> FakeAccountInfo:
    return FakeAccountInfo(
        login=login,
        trade_mode=trade_mode,
        leverage=100,
        currency="USD",
        balance=10_000.0,
        equity=10_000.0,
        margin=0.0,
        margin_free=10_000.0,
        margin_level=0.0,
        name="AurumX Test",
        server="AurumX-Demo",
        profit=0.0,
    )


def generate_bars(
    n: int,
    timeframe: int,
    *,
    end_epoch: int,
    base_price: float = 2650.0,
    seed: int = 42,
    spread_points: int = 20,
) -> list[dict[str, Any]]:
    """Deterministic random-walk bars, oldest -> newest.

    The newest bar is the *forming* bar (its open time is ``end_epoch``), which
    is what a live terminal serves for ``start_pos=0``.
    """
    minutes = TF_MINUTES[timeframe]
    step = minutes * 60
    times = [end_epoch - i * step for i in range(n)][::-1]
    rng = random.Random(seed)
    price = base_price
    bars: list[dict[str, Any]] = []
    for t in times:
        open_price = price
        close_price = open_price * (1 + rng.uniform(-0.0025, 0.0025))
        high = max(open_price, close_price) * (1 + rng.uniform(0.0, 0.0008))
        low = min(open_price, close_price) * (1 - rng.uniform(0.0, 0.0008))
        bars.append(
            {
                "time": t,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "tick_volume": rng.randint(80, 900),
                "spread": spread_points,
                "real_volume": 0,
            }
        )
        price = close_price
    return bars


class FakeMT5:
    """Stateful fake of the ``MetaTrader5`` module."""

    def __init__(
        self,
        *,
        symbols: list[FakeSymbolInfo] | None = None,
        account: FakeAccountInfo | None = None,
        terminal: FakeTerminalInfo | None = None,
        init_result: bool = True,
        last_error_value: tuple[int, str] = (0, "no error"),
        execution_enabled: bool = False,
        check_retcode: int | None = None,
        send_retcode: int | None = None,
        send_exception: Exception | None = None,
        auto_open_position: bool = True,
    ) -> None:
        self.symbols: dict[str, FakeSymbolInfo] = {s.name: s for s in (symbols or [])}
        self.rates: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.tick: FakeTick | None = None
        self.positions: list[FakePosition] = []
        self.orders: list[FakeOrder] = []
        self.account = account or default_account()
        self.terminal = terminal or FakeTerminalInfo(
            connected=True, trade_allowed=True, name="AurumX Fake Terminal",
            company="AurumX", maxbars=100000,
        )
        self.init_result = init_result
        self._last_error = last_error_value
        self.initialize_calls: list[dict[str, Any]] = []
        self.shutdown_called = False
        self._initialized = False
        self._next_ticket = 900001
        # ---- Phase-5 execution simulation (opt-in; read-only by default) --
        self.execution_enabled = execution_enabled
        self.check_retcode = check_retcode
        self.send_retcode = send_retcode
        self.send_exception = send_exception
        self.auto_open_position = auto_open_position
        self.order_checks: list[dict[str, Any]] = []
        self.order_sends: list[dict[str, Any]] = []

    # ---- lifecycle ------------------------------------------------------
    def initialize(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        portable: bool = False,
    ) -> bool:
        provided = {
            key: value
            for key, value in {
                "path": path, "login": login, "password": password,
                "server": server, "timeout": timeout,
            }.items()
            if value is not None
        }
        self.initialize_calls.append(provided)
        self._initialized = self.init_result
        return self.init_result

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._initialized = False

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def version(self) -> tuple[int, int, int]:
        return (5, 0, 4990)

    def terminal_info(self) -> FakeTerminalInfo:
        return self.terminal if self._initialized else None

    def account_info(self) -> FakeAccountInfo:
        return self.account if self._initialized else None

    # ---- symbols ----------------------------------------------------------
    def symbols_get(self, group: str | None = None) -> tuple[FakeSymbolInfo, ...]:
        if not self._initialized:
            return ()
        return tuple(self.symbols.values())

    def symbol_info(self, symbol: str) -> FakeSymbolInfo | None:
        return self.symbols.get(symbol)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        info = self.symbols.get(symbol)
        if info is None:
            self._last_error = (-1, f"unknown symbol {symbol}")
            return False
        if enable:
            self.symbols[symbol] = info._replace(select=True, visible=True)
        return True

    def symbol_info_tick(self, symbol: str) -> FakeTick | None:
        if not self._initialized or symbol not in self.symbols:
            return None
        return self.tick

    # ---- rates --------------------------------------------------------------
    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray | None:
        if not self._initialized:
            return None
        info = self.symbols.get(symbol)
        if info is None or not info.select:
            self._last_error = (-1, f"symbol {symbol} not selected")
            return None
        bars = self.rates.get((symbol, timeframe))
        if bars is None:
            self._last_error = (4401, "no history data")
            return None
        if count <= 0:
            return np.empty(0, dtype=RATES_DTYPE)
        # start_pos=0 is the current (forming) bar; newest last.
        end = len(bars) - start_pos
        start = max(0, end - count)
        rows = bars[start:end]
        array = np.empty(len(rows), dtype=RATES_DTYPE)
        for i, row in enumerate(rows):
            for field in RATES_DTYPE.names:  # type: ignore[union-attr]
                array[i][field] = row[field]
        return array

    # ---- portfolio ----------------------------------------------------------
    def positions_get(self, symbol: str | None = None) -> tuple[FakePosition, ...]:
        if not self._initialized:
            return ()
        return tuple(p for p in self.positions if symbol is None or p.symbol == symbol)

    def orders_get(self, symbol: str | None = None) -> tuple[FakeOrder, ...]:
        if not self._initialized:
            return ()
        return tuple(o for o in self.orders if symbol is None or o.symbol == symbol)

    # ---- execution (Phase 5; read-only safety net unless opted in) --------
    def _require_execution_enabled(self, func: str) -> None:
        if not self.execution_enabled:
            raise AssertionError(
                f"FakeMT5.{func} must never be called without "
                "execution_enabled=True: read-only tests must stay read-only. "
                "Phase-5 execution tests opt in explicitly."
            )

    def order_check(self, request: dict[str, Any]) -> FakeOrderCheckResult:
        """Simulated MT5 order_check.  Default: basic sanity validation;
        `check_retcode` overrides the verdict for failure-injection tests."""
        self._require_execution_enabled("order_check")
        self.order_checks.append(dict(request))
        if self.check_retcode is not None:
            return FakeOrderCheckResult(
                retcode=self.check_retcode, balance=self.account.balance,
                equity=self.account.equity, margin=self.account.margin,
                margin_free=self.account.margin_free, profit=0.0, comment="",
            )
        retcode = 0
        symbol = request.get("symbol")
        if symbol not in self.symbols:
            retcode = TRADE_RETCODE_INVALID
        elif not (request.get("volume", 0.0) or 0.0) > 0:
            retcode = TRADE_RETCODE_INVALID_VOLUME
        elif request.get("action") != TRADE_ACTION_DEAL:
            retcode = TRADE_RETCODE_INVALID
        return FakeOrderCheckResult(
            retcode=retcode, balance=self.account.balance,
            equity=self.account.equity, margin=self.account.margin,
            margin_free=self.account.margin_free, profit=0.0,
            comment="" if retcode == 0 else "check failed",
        )

    def order_send(self, request: dict[str, Any]) -> FakeOrderSendResult:
        """Simulated MT5 order_send.  Records every send; on success opens
        a matching FakePosition (unless `auto_open_position` is False) so
        verification/reconciliation can observe the resulting state.
        Never fabricates identifiers on failure paths."""
        self._require_execution_enabled("order_send")
        if self.send_exception is not None:
            raise self.send_exception
        self.order_sends.append(dict(request))
        retcode = self.send_retcode if self.send_retcode is not None else TRADE_RETCODE_DONE

        order_ticket = 0
        deal_ticket = 0
        volume = float(request.get("volume", 0.0) or 0.0)
        price = float(request.get("price", 0.0) or 0.0)
        bid = self.tick.bid if self.tick else 0.0
        ask = self.tick.ask if self.tick else 0.0

        if retcode in (TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL):
            order_ticket = self._next_ticket
            self._next_ticket += 1
            deal_ticket = self._next_ticket
            self._next_ticket += 1
            if retcode is not None and retcode == TRADE_RETCODE_DONE_PARTIAL:
                volume = round(volume / 2.0, 2) or volume
            position_type = (
                ORDER_TYPE_BUY if request.get("type") == ORDER_TYPE_BUY else ORDER_TYPE_SELL
            )
            if self.auto_open_position and request.get("action") == TRADE_ACTION_DEAL:
                self.positions.append(
                    FakePosition(
                        ticket=order_ticket,
                        symbol=str(request.get("symbol", "")),
                        type=position_type,
                        volume=volume,
                        price_open=price,
                        price_current=price,
                        sl=float(request.get("sl", 0.0) or 0.0),
                        tp=float(request.get("tp", 0.0) or 0.0),
                        profit=0.0,
                        swap=0.0,
                        time=(self.tick.time if self.tick else 0),
                        comment=str(request.get("comment", "")),
                        magic=int(request.get("magic", 0) or 0),
                    )
                )
        return FakeOrderSendResult(
            retcode=retcode,
            deal=deal_ticket or 0,
            order=order_ticket or 0,
            volume=volume,
            price=price,
            bid=bid,
            ask=ask,
            comment="",
            request_id=0,
            retcode_external=0,
            request=dict(request),
        )

    # ---- test helpers -----------------------------------------------------------
    def set_tick(self, symbol: str, *, epoch: float, bid: float, ask: float) -> None:
        self.tick = FakeTick(
            time=int(epoch),
            bid=bid,
            ask=ask,
            last=0.0,
            volume=0,
            time_msc=int(epoch * 1000),
            flags=2,
            volume_real=0.0,
        )

    def add_rates(
        self, symbol: str, timeframe: int, bars: list[dict[str, Any]]
    ) -> None:
        self.rates[(symbol, timeframe)] = bars

    def serve_market(
        self,
        symbol: str,
        *,
        now: datetime,
        timeframes: list[int] | None = None,
        candle_count: int = 500,
        bid: float = 2650.0,
        ask: float = 2650.20,
        base_price: float = 2650.0,
        seed: int = 42,
    ) -> None:
        """Wire up a coherent live-market scenario anchored at ``now``.

        The forming bar opens at ``floor(now, timeframe)`` and the tick is
        ``now`` minus 2 seconds.
        """
        now_epoch = int(now.timestamp())
        for timeframe in timeframes or [TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4]:
            minutes = TF_MINUTES[timeframe]
            forming_open = now_epoch - (now_epoch % (minutes * 60))
            self.add_rates(
                symbol,
                timeframe,
                generate_bars(
                    candle_count, timeframe, end_epoch=forming_open,
                    base_price=base_price, seed=seed + minutes,
                ),
            )
        self.set_tick(symbol, epoch=now_epoch - 2, bid=bid, ask=ask)
