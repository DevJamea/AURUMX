"""Exception hierarchy.

Fail-closed principle: when in doubt, raise.  The trading worker treats any
``AurumXError`` as "do nothing this cycle", and never as "try a trade anyway".
"""

from __future__ import annotations


class AurumXError(Exception):
    """Base class for every AurumX error."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class ConfigurationError(AurumXError):
    """The configuration is invalid."""


class UnsafeConfigurationError(ConfigurationError):
    """The configuration asks for something unsafe (e.g. real trading without
    the explicit confirmation phrase) and is refused."""


# --------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------
class BrokerError(AurumXError):
    """Base class for broker-layer failures."""


class MT5UnavailableError(BrokerError):
    """The ``MetaTrader5`` python package or the terminal is not available.

    The MetaTrader5 package is Windows-only; the system must never fake data
    when it is missing — it fails closed.
    """


class MT5ConnectionError(BrokerError):
    """Connecting to the MT5 terminal failed."""


class MT5NotConnectedError(BrokerError):
    """Operation attempted while the broker connection is down."""


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------
class SymbolDiscoveryError(AurumXError):
    """No tradable gold symbol could be discovered / verified."""


class SymbolNotGoldError(SymbolDiscoveryError):
    """Gold-only protection (spec §55): the symbol is not a broker-equivalent
    XAUUSD gold symbol (e.g. silver, another FX pair, an ETF)."""


class SymbolVerificationError(SymbolDiscoveryError):
    """The symbol exists but failed broker-metadata verification."""


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
class MarketDataError(AurumXError):
    """Market data could not be retrieved or is unusable."""


# --------------------------------------------------------------------------
# Execution (stubbed until Phase 5)
# --------------------------------------------------------------------------
class ExecutionError(AurumXError):
    """Base class for execution failures."""


class ExecutionNotImplementedError(ExecutionError):
    """Execution is intentionally not implemented yet.

    Phases 1–4 are read-only by construction: every order-placing method on the
    broker raises this error.  Phase 5 replaces the stubs with the full
    validate -> risk check -> order_check -> send -> verify -> reconcile flow.
    """
