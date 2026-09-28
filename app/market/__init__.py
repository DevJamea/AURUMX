"""Market layer: symbol discovery, tick/candle validation, market state.

Broker-agnostic: everything here works against ``BrokerInterface``.  MT5
specifics live in ``app.brokers.mt5``.
"""

from app.market.candles import CandleValidator, fetch_timeframes
from app.market.data_service import MarketDataService
from app.market.market_state import MarketSnapshot
from app.market.symbol_discovery import (
    SymbolDiscovery,
    SymbolDiscoveryResult,
    gold_match_score,
    is_gold_symbol,
)
from app.market.tick import TickValidator

__all__ = [
    "CandleValidator",
    "MarketDataService",
    "MarketSnapshot",
    "SymbolDiscovery",
    "SymbolDiscoveryResult",
    "TickValidator",
    "fetch_timeframes",
    "gold_match_score",
    "is_gold_symbol",
]
