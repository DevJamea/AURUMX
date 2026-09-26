# AurumX Architecture

Status: Phase 1 (foundation) implemented. This document describes the target
architecture and marks what exists today.

## 1. Layer model

The absolute rule (spec §2): **agents never execute trades; only the execution
layer talks to a broker.**

```text
┌──────────────────────────────────────────────────────────────────┐
│                          MARKET DATA                             │
│  app/market:  SymbolDiscovery · TickValidator · CandleValidator  │
│               MarketDataService  →  MarketSnapshot               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ reads through
┌──────────────────────────▼───────────────────────────────────────┐
│                     BROKER LAYER                                 │
│  app/brokers:  BrokerInterface (ABC)                             │
│                MT5Broker  ← the ONLY importer of MetaTrader5     │
│                (PaperBroker — Phase 5)                           │
└──────────────────────────▲───────────────────────────────────────┘
                           │ snapshots only (never brokers)
┌──────────────────────────┴───────────────────────────────────────┐
│                    ANALYSIS AGENTS (Phase 2)                     │
│  Trend · Momentum · Structure · Liquidity · Volatility ·         │
│  MeanReversion · Macro — pure functions of a MarketSnapshot      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ AgentResult (direction, signal_strength, reasons…)
┌──────────────────────────▼───────────────────────────────────────┐
│              DECISION ENGINE (Phase 3)                           │
│  Regime detection → regime-weighted evidence → BUY/SELL/HOLD     │
│  + explanations + decision journal (app/storage)                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │ proposed trade (entry/SL/TP/risk)
┌──────────────────────────▼───────────────────────────────────────┐
│                 HARD RISK GATE (Phase 4)                         │
│  PASS / BLOCK — code, not a vote. Never modifies decisions.      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ only approved requests
┌──────────────────────────▼───────────────────────────────────────┐
│              EXECUTION SERVICE (Phase 5)                         │
│  validate → risk check → order_check → send → verify → reconcile │
│  + position management (BE, partial close, trailing)             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                     Broker Adapter → MT5
```

## 2. What exists today (Phase 1)

| Component | File | Notes |
|---|---|---|
| Configuration | `app/core/config.py` | pydantic-settings, spec-named env vars, unsafe configs unconstructible |
| Domain models | `app/core/models.py` | broker-agnostic: `MarketTick`, `Candle`, `CandleSeries`, `SymbolSpec`, `AccountSnapshot`, `Position`, `PendingOrder`, execution request/result contracts, `ValidationReport` |
| Structured logging | `app/core/logging.py` | JSON lines with `timestamp/level/component/event/details`, secret redaction |
| Event bus | `app/core/events.py` | in-process pub/sub with handler isolation |
| Broker contract | `app/brokers/interface.py` | read-only + execution halves; MT5 replaceable |
| MT5 adapter | `app/brokers/mt5.py` | lazy Windows-only import, fail-closed mappings, thread-safe, execution stubbed |
| Symbol discovery | `app/market/symbol_discovery.py` | gold pattern scoring + full broker-metadata verification |
| Tick validation | `app/market/tick.py` | zero/inverted/stale/future checks, spread in points |
| Candle validation | `app/market/candles.py` | OHLC sanity, ordering, per-timeframe freshness, closed-bars-only |
| Market snapshot | `app/market/market_state.py` | `trading_data_ok` data-quality gate, session inference |
| Data service | `app/market/data_service.py` | orchestrates the above; server-clock-offset correction |
| Diagnostics | `scripts/diagnostics.py` | read-only health checks (launcher's "Run Diagnostics") |

## 3. Key design decisions

### 3.1 Broker-agnostic everywhere

Every layer above `app/brokers` works with `BrokerInterface` and domain
models. Swapping MT5 for another bridge (or the local paper broker) requires
no changes anywhere else. The `MetaTrader5` package is imported lazily inside
`MT5Broker.connect()` because it only installs on Windows — backtests, paper
trading, the API and the whole test suite run on any OS.

### 3.2 Data quality is a report, not an exception

Validators return `ValidationReport` objects (errors block trading, warnings
don't). `MarketSnapshot.trading_data_ok` is the single boolean the future risk
gate and health monitor will consume: symbol verified + tick valid & fresh +
every timeframe valid & fresh. Stale data ⇒ block trading (spec §52).

### 3.3 Closed candles only (no repainting, no look-ahead)

`MT5Broker.get_candles(..., include_forming=False)` (the default) drops the
still-forming bar. Agents will therefore only ever evaluate decisions on
information that was complete at bar close — the live-side twin of the
backtester's "candle-close-i may only influence entry at i+1" rule.

### 3.4 Broker clock-offset correction

MT5 servers usually run UTC+2/+3. The data service adopts a server-clock
offset from *future-stamped* ticks only (bounded ±6h): an offset that would
make an old tick look fresh can never be adopted, so a stale feed stays stale
(fail-closed). Freshness for ticks and candles is then judged in the server's
frame of reference. Known limitation: a server clock moving *backwards*
(DST edge case) surfaces as stale data (trading blocked) until restart.

### 3.5 Safety by construction

* Unsafe configurations raise `UnsafeConfigurationError` on construction and
  on mutation (`validate_assignment=True`): live trading without the exact
  confirmation phrase cannot exist as an object.
* Execution methods on `MT5Broker` raise `ExecutionNotImplementedError` until
  Phase 5 — no code path in Phases 1–4 can send an order, even accidentally.
  The test fake's `order_send` also asserts if reached.
* Unknown broker enum values map to the safe side (unknown symbol trade mode →
  `DISABLED`, unknown account mode → `REAL` so real-account alarms fire).

### 3.6 Gold-only protection (spec §55)

`gold_match_score()` recognizes broker spellings (`XAUUSD`, `XAUUSDm`,
`XAUUSD.a`, `XAUUSD.raw`, `GOLD`, `GOLDm`, `XAU/USD`, …) with conservative
suffix stripping (`GOLDEN` is not gold; `XAUUSDGIBBERISH` is not gold), and
explicitly denies other metals (`XAG*`, `XPT*`, `XPD*`). Explicitly configured
non-gold symbols raise `SymbolNotGoldError`.

### 3.7 Determinism & testability

* All market tests run against `tests/fakes/mt5_fake.py` — a deterministic
  fake terminal with seeded random-walk bars and injectable failure modes.
* Clocks are injectable (`clock=`), so freshness tests never depend on wall
  time.
* The same fake will power the paper-trading mode (Phase 5/6).

## 4. State machine (Phase 4+)

The worker will drive the explicit `SystemState` enum (`STARTING`, `CONNECTED`,
`ANALYZING`, `SIGNAL_READY`, `RISK_CHECK`, `EXECUTING`, `MANAGING`, `PAUSED`,
`EMERGENCY_STOP`, `ERROR`, `DISCONNECTED`) — never scattered booleans.
Emergency stop will be persisted so a restart cannot silently re-enable
trading.

## 5. Observability

* Every important event is a structured JSON log line with a stable event code
  (`MT5_CONNECTED`, `SYMBOL_DISCOVERED`, `MARKET_DATA_STALE`, …) — the same
  codes the event bus publishes and the dashboard will display.
* Secrets are redacted at the logging layer (defense in depth behind
  `SecretStr`).

## 6. Deployment shapes (target)

* **Mode A — Windows local:** MT5 + backend + worker + dashboard on one
  machine (`http://127.0.0.1:8000`), started by the launcher (Phase 8).
* **Mode B — VPS:** Docker services `aurumx-api`, `aurumx-worker`, `aurumx-db`,
  `aurumx-web` behind a reverse proxy (Phase 9). The MetaTrader5 package is
  Windows-only — in split deployments the broker/worker runs on the Windows MT5
  host and the API/DB/web may run in Docker.
* **Mode C — Remote control:** browser → HTTPS → API → trading core → MT5,
  with mandatory authentication; the UI never assumes MT5 is local and never
  sees broker credentials (Phase 7/9).
