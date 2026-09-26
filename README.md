# AurumX

**Modular multi-agent XAUUSD trading platform for MetaTrader 5.**

AurumX is a trading *research and execution* platform for gold (XAUUSD), built
around one principle: correctness, risk control, reproducibility and
observability come before anything else. It is designed to say **NO TRADE**
when the evidence is insufficient — a HOLD is a valid decision.

> **Not financial advice.** AurumX is experimental software. Default behavior
> is strictly read-only; real-money trading is refused unless explicitly,
> unmistakably enabled.

---

## Operating ladder

The system only ever advances explicitly — never by default:

```text
READ-ONLY  →  BACKTEST  →  PAPER / DRY-RUN  →  MT5 DEMO  →  (MT5 REAL, gated)
```

A fresh installation is **read-only**:

```env
TRADING_ENABLED=false
DRY_RUN=true
```

## Current status

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: config, logging, events, models, MT5 adapter (read-only), symbol discovery, tick/candle validation, market snapshots, diagnostics | ✅ **complete — 156 tests passing** |
| 2 | Analysis agents (trend, momentum, structure, liquidity, volatility, mean-reversion, macro) | ⏳ next |
| 3 | Decision engine (regime, weighted evidence, explanations, journal) | planned |
| 4 | Hard risk gate, position sizing, daily loss, kill switch | planned |
| 5 | Execution (orders, BE, partial close, trailing, pending orders) | planned |
| 6 | Backtesting + walk-forward | planned |
| 7 | Dashboard (React) + FastAPI | planned |
| 8 | Windows launcher | planned |
| 9 | Docker / VPS deployment | planned |
| 10 | Full QA + final report | planned |

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the full
plan, including the review of the two reference projects this architecture
learns from.

## Architecture (short version)

```text
Market Data (broker adapters)          ← only app/brokers touches MetaTrader5
        ↓
Analysis Agents                        ← never see a broker, never trade
        ↓
Decision Engine  (weighted evidence + market regime)
        ↓
Risk Gate        (hard gate — PASS/BLOCK, not a vote)
        ↓
Execution Service                      ← only layer allowed to trade
        ↓
Broker Adapter (MT5)
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Key safety properties (all enforced by code + tests today):

* **Read-only by default** — `TRADING_ENABLED=false`, `DRY_RUN=true`; unsafe
  configurations raise and cannot even be constructed.
* **Execution physically stubbed** — every order method on the broker raises
  `ExecutionNotImplementedError` until Phase 5.
* **Gold-only** — non-gold symbols (silver, FX, futures tickers…) are rejected.
* **Fail-closed MT5** — missing Windows package / dead terminal / unknown enum
  values all resolve to the safe side, never to fake data.
* **Closed candles only** — the forming bar is always excluded, so analysis and
  backtests see only information that existed at bar close.
* **Secrets never logged** — `MT5_PASSWORD` is a `SecretStr`; the logger
  redacts password/token/key fields.

## Quickstart

```bash
# Python 3.11+
python -m venv .venv
.venv/bin/pip install -e ".[dev]"          # Windows:  pip install -e ".[dev,mt5]"

# run the test suite (no MT5 terminal required — a deterministic fake is used)
.venv/bin/python -m pytest

# read-only diagnostics against a real terminal (Windows with MT5 installed)
.venv/bin/python scripts/diagnostics.py
```

Configure via `.env` (see [`.env.example`](.env.example)); every risky setting
is documented there.

## Project layout

```text
app/
├── core/       config, enums, domain models, events, exceptions, structured logging
├── brokers/    BrokerInterface + MT5Broker (the ONLY MetaTrader5 importer)
├── market/     symbol discovery, tick/candle validation, market snapshots
├── agents/     specialized analysis agents            (Phase 2)
├── decision/   regime + weighted decision engine      (Phase 3)
├── risk/       hard risk gate                         (Phase 4)
├── execution/  order lifecycle                        (Phase 5)
├── backtest/   no-look-ahead simulator + walk-forward (Phase 6)
├── storage/    repositories (SQLite → PostgreSQL)     (Phase 3+)
├── worker/     trading loop + health monitor          (Phase 4+)
└── api/        FastAPI + websocket                    (Phase 7)
frontend/       React + TypeScript dashboard           (Phase 7)
scripts/        diagnostics and operational tooling
tests/          unit / integration / backtest / risk / execution
```

## Testing philosophy

The implementation is not "done because it runs". Every phase ships with
tests; MT5-dependent code is tested against a deterministic fake terminal
(`tests/fakes/mt5_fake.py`) so the whole pipeline — discovery, validation,
freshness gating, clock-offset correction — is verified without Windows.
Edge cases covered today include: zero ticks, inverted quotes, stale ticks,
future-stamped ticks, missing timeframes, broken symbol metadata, invisible
symbols, real-account warnings, credential redaction and restart-safe
configuration guards.

## License

MIT (see `pyproject.toml`).
