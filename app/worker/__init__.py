"""Trading worker (Phase 4+).

Planned modules: ``trading_loop`` (state-machine driven cycle: connect ->
analyze -> decide -> risk -> execute -> manage), ``scheduler``,
``health_monitor`` (MT5 connection, tick/candle freshness, API, DB, execution
errors — stale data blocks trading).
"""

