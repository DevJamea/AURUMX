"""AurumX — modular multi-agent XAUUSD trading platform for MetaTrader 5.

Operating ladder (only ever advanced explicitly, never by default):

    READ-ONLY  ->  BACKTEST  ->  PAPER/DRY-RUN  ->  MT5 DEMO

A fresh installation is read-only: ``TRADING_ENABLED=false`` and ``DRY_RUN=true``.
"""

__version__ = "0.1.0"
