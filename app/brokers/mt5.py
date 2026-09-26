"""MetaTrader 5 broker adapter — the ONLY module importing ``MetaTrader5``.

Phase-1 scope: **read-only** (connect/disconnect, account, symbols, ticks,
candles, positions, orders).  Every execution method is a hard stub raising
``ExecutionNotImplementedError`` until Phase 5 — a deliberate safety property:
nothing built on Phases 1–4 can send an order, even accidentally.

Fail-closed rules (learned from the TradingAgents-Gold adapter):

* The ``MetaTrader5`` package is Windows-only.  When it is missing the adapter
  raises ``MT5UnavailableError`` — it never fakes data.
* A failed ``initialize()`` raises with ``last_error()`` details.
* Unknown enum values map to the *safe* side (unknown symbol trade mode ->
  DISABLED, unknown account mode -> REAL so real-account warnings fire).
* Credentials are accepted but never logged; the logger redacts secrets anyway.

Testing: ``mt5_module=`` injects a fake module implementing the small
MetaTrader5 API subset (see ``tests/fakes/mt5_fake.py``), which is how the
whole system is tested without a Windows terminal.
"""

from __future__ import annotations

import importlib
import threading
from datetime import UTC, datetime
from typing import Any

from app.brokers.interface import BrokerInterface
from app.core.enums import (
    AccountTradeMode,
    Direction,
    OrderType,
    SymbolTradeMode,
    TimeFrame,
)
from app.core.exceptions import (
    ExecutionNotImplementedError,
    MarketDataError,
    MT5ConnectionError,
    MT5NotConnectedError,
    MT5UnavailableError,
)
from app.core.logging import get_logger
from app.core.models import (
    AccountSnapshot,
    Candle,
    CandleSeries,
    MarketOrderRequest,
    MarketTick,
    OrderResult,
    PendingOrder,
    PendingOrderRequest,
    Position,
    SymbolSpec,
)

log = get_logger("brokers.mt5")

#: Default import name of the official package.
MT5_IMPORT_NAME = "MetaTrader5"

# ---- MetaQuotes integer constants (mirrored so mapping works even when the
# injected module does not expose the enum names) --------------------------------
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2

SYMBOL_TRADE_MODE_DISABLED = 0
SYMBOL_TRADE_MODE_LONGONLY = 1
SYMBOL_TRADE_MODE_SHORTONLY = 2
SYMBOL_TRADE_MODE_CLOSEONLY = 3
SYMBOL_TRADE_MODE_FULL = 4

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5
ORDER_TYPE_BUY_STOP_LIMIT = 6
ORDER_TYPE_SELL_STOP_LIMIT = 7

#: Fallback timeframe constants (same values as the package's enums).
_TIMEFRAME_FALLBACK: dict[TimeFrame, int] = {
    TimeFrame.M1: 1,
    TimeFrame.M5: 5,
    TimeFrame.M15: 15,
    TimeFrame.M30: 30,
    TimeFrame.H1: 16385,
    TimeFrame.H4: 16388,
    TimeFrame.D1: 16408,
}

_ACCOUNT_TRADE_MODES: dict[int, AccountTradeMode] = {
    ACCOUNT_TRADE_MODE_DEMO: AccountTradeMode.DEMO,
    ACCOUNT_TRADE_MODE_CONTEST: AccountTradeMode.CONTEST,
    ACCOUNT_TRADE_MODE_REAL: AccountTradeMode.REAL,
}

_SYMBOL_TRADE_MODES: dict[int, SymbolTradeMode] = {
    SYMBOL_TRADE_MODE_DISABLED: SymbolTradeMode.DISABLED,
    SYMBOL_TRADE_MODE_LONGONLY: SymbolTradeMode.LONG_ONLY,
    SYMBOL_TRADE_MODE_SHORTONLY: SymbolTradeMode.SHORT_ONLY,
    SYMBOL_TRADE_MODE_CLOSEONLY: SymbolTradeMode.CLOSE_ONLY,
    SYMBOL_TRADE_MODE_FULL: SymbolTradeMode.FULL,
}

_ORDER_TYPES: dict[int, OrderType] = {
    ORDER_TYPE_BUY: OrderType.BUY,
    ORDER_TYPE_SELL: OrderType.SELL,
    ORDER_TYPE_BUY_LIMIT: OrderType.BUY_LIMIT,
    ORDER_TYPE_SELL_LIMIT: OrderType.SELL_LIMIT,
    ORDER_TYPE_BUY_STOP: OrderType.BUY_STOP,
    ORDER_TYPE_SELL_STOP: OrderType.SELL_STOP,
    ORDER_TYPE_BUY_STOP_LIMIT: OrderType.BUY_STOP_LIMIT,
    ORDER_TYPE_SELL_STOP_LIMIT: OrderType.SELL_STOP_LIMIT,
}

_EXECUTION_PHASE_5_MESSAGE = (
    "MT5 order execution is implemented in Phase 5 (execution layer). "
    "Phases 1-4 are read-only by construction."
)


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Defensive attribute access — broker builds and fake modules vary."""
    value = getattr(obj, attr, default)
    return default if value is None else value


class MT5Broker(BrokerInterface):
    """MetaTrader 5 implementation of ``BrokerInterface``.

    Parameters mirror ``AppConfig`` MT5 settings; use ``MT5Broker.from_config``
    to build one from the application configuration.  ``mt5_module`` injects a
    (fake) module for testing; without it the real ``MetaTrader5`` package is
    imported lazily on ``connect()``.
    """

    name = "mt5"

    def __init__(
        self,
        *,
        terminal_path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout_ms: int = 60000,
        mt5_module: Any | None = None,
        import_name: str = MT5_IMPORT_NAME,
    ) -> None:
        self._terminal_path = terminal_path
        self._login = login
        self._password = password  # never logged (logger redaction is the 2nd line of defense)
        self._server = server
        self._timeout_ms = timeout_ms
        self._module_override = mt5_module
        self._import_name = import_name
        self._mt5: Any | None = None
        self._connected = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Any, *, mt5_module: Any | None = None) -> MT5Broker:
        """Build from an ``AppConfig`` (reads its MT5_* fields)."""
        return cls(
            terminal_path=config.mt5_terminal_path,
            login=config.mt5_login,
            password=config.mt5_password.get_secret_value() if config.mt5_password else None,
            server=config.mt5_server,
            timeout_ms=config.mt5_timeout_ms,
            mt5_module=mt5_module,
        )

    def _load_module(self) -> Any:
        if self._module_override is not None:
            return self._module_override
        try:
            return importlib.import_module(self._import_name)
        except ImportError as exc:  # pragma: no cover - depends on host platform
            raise MT5UnavailableError(
                f"The '{self._import_name}' python package is not installed. "
                "It is Windows-only: install it on the machine running the MT5 "
                "terminal (pip install MetaTrader5) or run the system in "
                "backtest/paper mode."
            ) from exc

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            mt5 = self._load_module()

            kwargs: dict[str, Any] = {}
            if self._terminal_path:
                kwargs["path"] = self._terminal_path
            if self._login is not None:
                if not self._password:
                    raise MT5ConnectionError(
                        "MT5_LOGIN is configured without MT5_PASSWORD — refusing to connect."
                    )
                kwargs["login"] = self._login
                kwargs["password"] = self._password
                if self._server:
                    kwargs["server"] = self._server
            if self._timeout_ms:
                kwargs["timeout"] = self._timeout_ms

            log.info(
                "connecting to MetaTrader 5 terminal",
                event="MT5_CONNECTING",
                terminal_path=self._terminal_path,
                login=self._login,
                server=self._server,
            )
            if not mt5.initialize(**kwargs):
                last_error = mt5.last_error()
                raise MT5ConnectionError(
                    f"MT5 initialize() failed: {last_error}. Check that the terminal "
                    "is installed/running and AutoTrading is enabled."
                )

            self._mt5 = mt5
            self._connected = True

            try:
                version = mt5.version()
                log.info("MT5 terminal connected", event="MT5_CONNECTED", version=str(version))

                terminal = mt5.terminal_info()
                if terminal is None:
                    log.warning("terminal_info() returned no data", event="MT5_TERMINAL_INFO_MISSING")
                else:
                    if not _get(terminal, "connected", True):
                        log.warning(
                            "terminal reports 'not connected' to the trade server",
                            event="MT5_SERVER_DISCONNECTED",
                        )
                    if not _get(terminal, "trade_allowed", True):
                        log.warning(
                            "AutoTrading is DISABLED in the terminal — order requests "
                            "would be rejected (read-only data is unaffected)",
                            event="MT5_AUTOTRADING_DISABLED",
                        )

                account = self._account_raw()
                if account is None:
                    log.warning(
                        "no account info available (terminal running without a logged-in account?)",
                        event="MT5_NO_ACCOUNT",
                    )
                else:
                    snapshot = self.get_account()
                    if snapshot.is_real:
                        log.warning(
                            "REAL account detected — AurumX targets DEMO accounts; "
                            "real execution is refused by default (Phase 5)",
                            event="MT5_REAL_ACCOUNT",
                            login=snapshot.login,
                            server=snapshot.server,
                        )
                    else:
                        log.info(
                            "account connected",
                            event="MT5_ACCOUNT_OK",
                            login=snapshot.login,
                            server=snapshot.server,
                            trade_mode=snapshot.trade_mode.value,
                            currency=snapshot.currency,
                            equity=snapshot.equity,
                        )
            except Exception:
                # A post-connect check failing must not leave a half-open state.
                self._connected = False
                self._mt5 = None
                try:
                    mt5.shutdown()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                raise

    def disconnect(self) -> None:
        with self._lock:
            if self._mt5 is not None and self._connected:
                try:
                    self._mt5.shutdown()
                finally:
                    log.info("MT5 terminal disconnected", event="MT5_DISCONNECTED")
            self._connected = False
            self._mt5 = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require(self) -> Any:
        if not self._connected or self._mt5 is None:
            raise MT5NotConnectedError("MT5 broker is not connected (call connect() first)")
        return self._mt5

    # ------------------------------------------------------------------
    # account & symbols
    # ------------------------------------------------------------------
    def _account_raw(self) -> Any | None:
        mt5 = self._require()
        try:
            return mt5.account_info()
        except Exception as exc:  # pragma: no cover - terminal-level failure
            raise MT5ConnectionError(f"account_info() failed: {exc}") from exc

    def get_account(self) -> AccountSnapshot:
        info = self._account_raw()
        if info is None:
            raise MT5ConnectionError(
                "account_info() returned no data — terminal not logged in to an account"
            )
        trade_mode_raw = int(_get(info, "trade_mode", ACCOUNT_TRADE_MODE_REAL))
        margin_level = _get(info, "margin_level")
        return AccountSnapshot(
            login=int(_get(info, "login", 0)),
            # unknown account modes resolve to REAL => fail-closed warnings
            trade_mode=_ACCOUNT_TRADE_MODES.get(trade_mode_raw, AccountTradeMode.REAL),
            server=str(_get(info, "server", "")),
            currency=str(_get(info, "currency", "USD")),
            leverage=int(_get(info, "leverage", 0)),
            balance=float(_get(info, "balance", 0.0)),
            equity=float(_get(info, "equity", 0.0)),
            margin=float(_get(info, "margin", 0.0)),
            margin_free=float(_get(info, "margin_free", 0.0)),
            margin_level=float(margin_level) if margin_level else None,
            name=str(_get(info, "name", "")),
        )

    def list_symbols(self) -> list[str]:
        mt5 = self._require()
        infos = mt5.symbols_get() or ()
        return [str(_get(info, "name", "")) for info in infos if _get(info, "name")]

    def get_symbol(self, name: str) -> SymbolSpec | None:
        mt5 = self._require()
        info = mt5.symbol_info(name)
        if info is None:
            return None
        trade_mode_raw = int(_get(info, "trade_mode", SYMBOL_TRADE_MODE_DISABLED))
        tick_value = _get(info, "trade_tick_value", 0.0) or _get(info, "trade_tick_value_profit", 0.0)
        return SymbolSpec(
            name=str(_get(info, "name", name)),
            description=str(_get(info, "description", "")),
            visible=bool(_get(info, "visible", False)),
            selected=bool(_get(info, "select", False)),
            # unknown symbol modes resolve to DISABLED => fail-closed verification
            trade_mode=_SYMBOL_TRADE_MODES.get(trade_mode_raw, SymbolTradeMode.DISABLED),
            digits=int(_get(info, "digits", 0)),
            point=float(_get(info, "point", 0.0)),
            tick_size=float(_get(info, "trade_tick_size", 0.0)),
            tick_value=float(tick_value),
            contract_size=float(_get(info, "trade_contract_size", 0.0)),
            volume_min=float(_get(info, "volume_min", 0.0)),
            volume_max=float(_get(info, "volume_max", 0.0)),
            volume_step=float(_get(info, "volume_step", 0.0)),
            stops_level_points=int(_get(info, "trade_stops_level", 0)),
            freeze_level_points=int(_get(info, "trade_freeze_level", 0)),
            currency_profit=str(_get(info, "currency_profit", "")),
            currency_margin=str(_get(info, "currency_margin", "")),
        )

    def select_symbol(self, name: str) -> bool:
        mt5 = self._require()
        try:
            return bool(mt5.symbol_select(name, True))
        except Exception as exc:
            log.warning("symbol_select failed", event="MT5_SYMBOL_SELECT_FAILED", symbol=name, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------
    def get_tick(self, symbol: str) -> MarketTick | None:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        time_ms = int(_get(tick, "time_msc", 0) or 0)
        epoch_seconds = (time_ms / 1000.0) if time_ms else float(_get(tick, "time", 0.0))
        if epoch_seconds <= 0:
            return None
        return MarketTick(
            symbol=symbol,
            time=datetime.fromtimestamp(epoch_seconds, tz=UTC),
            bid=float(_get(tick, "bid", 0.0)),
            ask=float(_get(tick, "ask", 0.0)),
            last=float(_get(tick, "last", 0.0)) or None,
            volume=float(_get(tick, "volume", 0.0)) or None,
            source_time_ms=time_ms or None,
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int,
        *,
        include_forming: bool = False,
    ) -> CandleSeries:
        mt5 = self._require()
        const = self._timeframe_const(mt5, timeframe)
        # start_pos=0 always includes the forming bar as the newest element;
        # fetch one extra and drop it so callers only ever see closed bars.
        fetch_count = count if include_forming else count + 1
        rates = mt5.copy_rates_from_pos(symbol, const, 0, fetch_count)
        if rates is None:
            raise MarketDataError(
                f"copy_rates_from_pos({symbol!r}, {timeframe.value}) returned no data: "
                f"{mt5.last_error()}"
            )
        candles = [
            Candle(
                time=datetime.fromtimestamp(int(bar["time"]), tz=UTC),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                tick_volume=float(bar["tick_volume"]),
                real_volume=float(bar["real_volume"]),
                spread_points=int(bar["spread"]) if bar["spread"] else None,
            )
            for bar in rates
        ]
        if not include_forming and candles:
            candles = candles[:-1]
        return CandleSeries(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            include_forming=include_forming,
        )

    @staticmethod
    def _timeframe_const(mt5: Any, timeframe: TimeFrame) -> int:
        const = getattr(mt5, f"TIMEFRAME_{timeframe.value}", None)
        if isinstance(const, int) and const > 0:
            return const
        return _TIMEFRAME_FALLBACK[timeframe]

    # ------------------------------------------------------------------
    # portfolio
    # ------------------------------------------------------------------
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        mt5 = self._require()
        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        positions: list[Position] = []
        for pos in raw or ():
            position_type = int(_get(pos, "type", POSITION_TYPE_SELL))
            direction = {
                POSITION_TYPE_BUY: Direction.LONG,
                POSITION_TYPE_SELL: Direction.SHORT,
            }.get(position_type, Direction.NEUTRAL)
            if direction is Direction.NEUTRAL:
                log.warning(
                    "unknown position type", event="MT5_UNKNOWN_POSITION_TYPE",
                    ticket=_get(pos, "ticket"), type_raw=position_type,
                )
            positions.append(
                Position(
                    ticket=int(_get(pos, "ticket", 0)),
                    symbol=str(_get(pos, "symbol", "")),
                    direction=direction,
                    volume=float(_get(pos, "volume", 0.0)),
                    price_open=float(_get(pos, "price_open", 0.0)),
                    price_current=float(_get(pos, "price_current", 0.0)) or None,
                    price_sl=float(_get(pos, "sl", 0.0)) or None,
                    price_tp=float(_get(pos, "tp", 0.0)) or None,
                    profit=float(_get(pos, "profit", 0.0)),
                    swap=float(_get(pos, "swap", 0.0)),
                    time=datetime.fromtimestamp(int(_get(pos, "time", 0)), tz=UTC),
                    comment=str(_get(pos, "comment", "")),
                    magic=int(_get(pos, "magic", 0)),
                )
            )
        return positions

    def get_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        mt5 = self._require()
        raw = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        orders: list[PendingOrder] = []
        for order in raw or ():
            order_type_raw = int(_get(order, "type", -1))
            order_type = _ORDER_TYPES.get(order_type_raw, OrderType.UNKNOWN)
            if order_type is OrderType.UNKNOWN:
                log.warning(
                    "unknown order type", event="MT5_UNKNOWN_ORDER_TYPE",
                    ticket=_get(order, "ticket"), type_raw=order_type_raw,
                )
            expiration = _get(order, "time_expiration", 0)
            orders.append(
                PendingOrder(
                    ticket=int(_get(order, "ticket", 0)),
                    symbol=str(_get(order, "symbol", "")),
                    order_type=order_type,
                    volume=float(_get(order, "volume_current", 0.0)),
                    price_open=float(_get(order, "price_open", 0.0)),
                    price_sl=float(_get(order, "sl", 0.0)) or None,
                    price_tp=float(_get(order, "tp", 0.0)) or None,
                    time_setup=datetime.fromtimestamp(int(_get(order, "time_setup", 0)), tz=UTC),
                    time_expiration=(
                        datetime.fromtimestamp(int(expiration), tz=UTC)
                        if expiration
                        else None
                    ),
                    comment=str(_get(order, "comment", "")),
                    magic=int(_get(order, "magic", 0)),
                )
            )
        return orders

    # ------------------------------------------------------------------
    # execution — HARD STUBS until Phase 5 (intentional safety property)
    # ------------------------------------------------------------------
    def place_market_order(self, request: MarketOrderRequest) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)

    def place_pending_order(self, request: PendingOrderRequest) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)

    def modify_position(
        self,
        ticket: int,
        *,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)

    def close_position(self, ticket: int, *, volume: float | None = None) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)

    def partial_close(self, ticket: int, volume: float) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)

    def cancel_order(self, ticket: int) -> OrderResult:
        raise ExecutionNotImplementedError(_EXECUTION_PHASE_5_MESSAGE)
