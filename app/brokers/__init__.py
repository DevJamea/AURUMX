"""Broker layer.

Only this package may import a broker SDK (``MetaTrader5``).  Every other layer
depends on ``BrokerInterface`` and broker-agnostic domain models from
``app.core.models`` — that is what makes MT5 replaceable later (spec §2, §29).
"""

from app.brokers.interface import BrokerInterface
from app.brokers.mt5 import MT5Broker

__all__ = ["BrokerInterface", "MT5Broker"]
