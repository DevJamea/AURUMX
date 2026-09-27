# AurumX Architecture

Status: Phases 1 (foundation), 2 (deterministic agent layer), 3
(decision engine, proposals, journal — hardened) and 4 (hard risk gate:
18 checks, kill switch, emergency stop, risk events) implemented.
Phase 4 is proposal-only: it approves/rejects, it never executes.
This document describes the target architecture and marks
what exists today.

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
│                DECISION ENGINE (Phases 2–3)                      │
│  Regime → agents → synthesis → 14 deterministic gates →          │
│  BUY/SELL/HOLD/ABORT + TradeProposal (entry/SL/TP/volume)        │
│  + DecisionJournal (full provenance, every evaluation)           │
│  Phase 3 is proposal-only: NO execution, NO broker access        │
└──────────────────────────┬───────────────────────────────────────┘
                           │ proposed trade (pure data, expires)
┌──────────────────────────▼───────────────────────────────────────┐
│                 HARD RISK GATE (Phase 4 — implemented)                         │
│  HardRiskGate: 18 checks, fail-closed, independent risk math.      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ APPROVED / REJECTED / EMERGENCY_STOP · never modifies proposals
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

## 2b. What exists today (Phase 2 — agent layer)

| Component | File | Notes |
|---|---|---|
| Agent contract | `app/agents/base.py` | `AgentResult` with provenance (`reasons`, `features`, `warnings`, `data_quality`, `source_time`); bad-tick guard |
| Shared feature layer | `app/agents/features.py` | EMA/RSI/MACD/ROC/ATR/BB/ADX, Wilder smoothing, ATR-buffered comparisons, fractal swings with 2-candle confirmation + prominence filter, BOS/CHOCH events, closed-candles only |
| Multi-timeframe context | `app/agents/context.py` | M15/H1/H4 with explicit roles (ENTRY/STRUCTURE/MACRO); features computed once per timeframe |
| TrendAgent | `app/agents/trend.py` | EMA20/50/200 + ADX + slope + H4 + structure evidence table; E1 + ≥2 confirmations required |
| MomentumAgent | `app/agents/momentum.py` | regime-conditional (trend/range modes); exhaustion + conflict guards; no naive RSI reversals |
| StructureAgent | `app/agents/structure.py` | confirmed swings only, BOS/CHOCH, invalidation levels, documented confirmation delay |
| LiquidityAgent | `app/agents/liquidity.py` | sweeps/wick rejections/breakouts/failed breakouts vs H1 swings, prev-day levels, M15 extremes; volume-confirmed where volume exists |
| VolatilityAgent | `app/agents/volatility.py` | LOW/NORMAL/HIGH/EXTREME + expansion state; contextual, never directional |
| MeanReversionAgent | `app/agents/mean_reversion.py` | RANGE-gated, L1/L2/L3 gates, no martingale/averaging/DCA by construction |
| MacroAgent | `app/agents/macro.py` | provider protocol, blackout window, NEUTRAL/NO_DATA without a source, never fabricates |
| Agent registry | `app/agents/registry.py` | ordered, enable/disable, failure-isolated (raising agent → INVALID NEUTRAL) |
| Regime engine | `app/decision/regime.py` | TREND/RANGE/volatility/UNCERTAIN + per-agent relevance matrix (HIGH 1.25 / NORMAL 1.0 / REDUCED 0.5 / DISABLED 0) |
| Synthesis contract | `app/decision/synthesis.py` | deterministic reference aggregation; disagreement preserved (supporting/opposing/neutral/disabled + conflicts) |

See `docs/AGENTS.md` for the full agent reference (evidence tables, gates,
limitations, examples).

## 2c. What exists today (Phase 3 — decision engine)

| Component | File | Notes |
|---|---|---|
| Engine config | `app/decision/config.py` | every threshold in one validated model; engineering defaults, not statistically optimal (see `docs/DECISIONS.md`) |
| Timeframe alignment | `app/decision/alignment.py` | H4/H1/M15 weighted agreement; renormalization is policy-gated (strict H4 default, `decision.timeframe`); records effective weights |
| SL/TP levels | `app/decision/levels.py` | hierarchy: structure → ATR(2×) → broker minimum; TP methods `TP_BY_RR` / `TP_BY_STRUCTURE` (structure-derived RR genuinely gated); tick-grid snapping; Phase-1 validators |
| Trade proposal | `app/decision/proposal.py` | pure data: entry (validated ask/bid), SL, TP, RR, volume, invalidation conditions, TTL; `EntryPriceProvider` protocol (tick now, historical adapter in Phase 6) |
| Decision engine | `app/decision/engine.py` | 17-gate pipeline; deterministic BUY/SELL/HOLD/ABORT; slices every series to candles closed by `now` (no look-ahead); agent rosters canonically ordered |
| Risk sizing | `app/risk/sizing.py` | pure calculator: equity × risk% / loss-per-lot, floored to step (never rounded up); martingale structurally impossible (closed signature) |
| Risk state | `app/risk/state.py` | caller-supplied snapshot (daily loss, limits, open/pending counts, active fingerprints); engine reads, never mutates |
| Journal contract | `app/decision/journal.py` | `DecisionRecord` with full evidence (agents, synthesis, gates, config, snapshot ref); JSON-safe, credential-free |
| SQLite journal | `app/storage/decision_journal.py` | WAL, indexed (ts/symbol), flat schema → PostgreSQL-ready |
| Phase-4 gate contract | `app/risk/gate.py` | `RiskGate` Protocol (`evaluate(proposal, risk_state, account_state) -> RiskDecision`), `REQUIRED_CHECKS` (14 contract-level names), `RiskCheck`/`RiskDecision` models, `AccountState` evidence input (None-defaults, `trade_allowed=False`) |
| Phase-4 gate implementation | `app/risk/engine.py` | `HardRiskGate`: 18 ordered checks (`IMPLEMENTED_CHECKS`), tiered precedence (emergency > kill switch > trading > account > safety > approve), independent risk/exposure math, optional `event_sink`, `RiskGateConfig` (safe defaults; `max_total_exposure` never defaulted) |
| Kill switch / emergency stop | `app/risk/kill_switch.py` | `KillSwitchState` + `EmergencyStopState` (STANDBY/ACTIVE/TRIGGERED/RESET_REQUIRED), `KillSwitchStore` Protocol (future GUI/VPS persistence), `halt_flags()` bridge to `AccountState` |
| Risk events | `app/risk/events.py` | `RISK_APPROVED` / `RISK_REJECTED` / `EMERGENCY_STOP` / `KILL_SWITCH_ACTIVE` data objects — no Telegram/HTTP (receivers stamp time) |

See `docs/DECISIONS.md` for the Phase-3 decision record, and
`docs/RISK_GATE.md` for the Phase-4 gate reference (all 18 checks,
precedence, fail-closed table, kill switch, limitations). (HOLD vs ABORT
semantics, threshold rationale, SL/TP methodology, fingerprinting,
determinism guarantees, known limitations).

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
