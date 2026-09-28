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
| 2 | Analysis agents (trend, momentum, structure, liquidity, volatility, mean-reversion, macro) | ✅ **complete — 359 tests passing** |
| 3 | Decision engine: deterministic gates, trade proposals, sizing, SL/TP, risk-state interface, anti-overtrading, decision journal (+ hardening: explicit TP/RR semantics, strict H4 policy, gate classification, Phase-4 RiskGate contract) | ✅ **complete — 587 tests passing; proposal-only, no execution** |
| 4 | Hard risk gate, kill switch, exposure limits, emergency stop | ✅ **complete — 789 tests passing; approve/reject only, no execution** |
| 5 | MT5 execution (market orders), DRY_RUN, DEMO guard, reconciliation, Windows control plane | ✅ **complete — 1048 tests passing; offline-verified; real accounts blocked** (position management: later phase) |
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
Risk Gate        (hard gate — APPROVED/REJECTED/EMERGENCY_STOP, not a vote; done)
        ↓
Execution Service                      ← only layer allowed to trade (Phase 5 done)
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
├── agents/     7 deterministic analysis agents + feature layer (Phase 2, done)
├── decision/   regime, synthesis, gates, proposals, journal  (Phases 2–3 done)
├── risk/       sizing + risk state + gate contract · HardRiskGate 18 checks (Phase 4 done)
├── execution/  contracts + service + journal + reconciliation (Phase 5 done)
├── control/    engine runtime + local control API + GUI     (Phase 5 done)
├── backtest/   no-look-ahead simulator + walk-forward (Phase 6)
├── storage/    decision journal (SQLite, done) → PostgreSQL (later)
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

**Phase 2 (agent layer)** adds the deterministic analysis stack: seven agents
(trend, momentum, structure, liquidity, volatility, mean-reversion, macro
interface), a regime engine and a synthesis contract — all pure functions of
the market snapshot.  Verified properties include: determinism (same input →
same result), purity (analysis never mutates the snapshot), wall-clock
independence, no-look-ahead structure signals (prefix-consistency proofs),
disagreement preservation in synthesis, and a full adversarial matrix (NaN /
infinities / duplicate, unordered or gapped series / spikes / constant prices /
zero volume / bad ticks — all fail safe to NEUTRAL).  See
[`docs/AGENTS.md`](docs/AGENTS.md) for the agent reference.

**Phase 3 (decision engine)** adds the deterministic decision layer on top:
a 17-gate pipeline that turns agent evidence into BUY / SELL / HOLD / ABORT
with a full audit trail — data-quality and session gates (fail-closed:
UNKNOWN session or excessive spread means ABORT, never "assume it's fine"),
timeframe-alignment and conflict gates that preserve disagreement, SL/TP
construction (structure → ATR → broker-minimum hierarchy), pure position
sizing floored to the broker step, setup fingerprints to stop duplicate
proposals, and a decision journal that records every evaluation with the
complete evidence needed to answer *why* the bot decided anything. The
engine is provably deterministic and look-ahead-free:
`decision(prefix) == decision(full history, evaluated at the prefix's time)`.
**Phase 3 produces proposals and journal entries only — it cannot execute
anything.** The hardening pass made the TP/RR semantics explicit
(`TP_BY_RR` default with a documented self-referential RR gate;
`TP_BY_STRUCTURE` with a genuinely enforced minimum RR), made missing-H4
handling a strict policy decision (HOLD `missing_primary_context` unless
renormalization is explicitly enabled — and then journaled with the
effective weights), classified every gate, and contracted the Phase-4
boundary (`app/risk/gate.py`: `RiskGate.evaluate(proposal, risk_state,
account_state) -> APPROVED / REJECTED / EMERGENCY_STOP`, interface only —
Phase 4 re-verifies all 14 safety checks independently). See
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the design record and known
limitations.

Edge cases covered today include: zero ticks, inverted quotes, stale ticks,
future-stamped ticks, missing timeframes, broken symbol metadata, invisible
symbols, real-account warnings, credential redaction and restart-safe
configuration guards.

### Phase 5 — execution + control plane (complete)

The execution layer turns a gate-APPROVED proposal into a controlled MT5
market order — or a provably order-free DRY_RUN simulation — and verifies
what actually happened against the broker:

- `app/execution` — `ExecutionRequest` (verbatim from the approved
  proposal, deterministic ids), `ExecutionService` (approval verification,
  independent local validation, DRY_RUN, DEMO-guarded MT5 path with
  `order_check` before `order_send`, explicit retcode verification, state
  verification before `FILLED`, no retries), reconciliation
  (`MATCHED / MISSING_IN_MT5 / MISSING_IN_JOURNAL / MISMATCH / UNKNOWN`,
  mismatch halts execution until explicitly resolved), and the execution
  journal with the full correlation chain.
- `app/control` — the local Windows control plane: engine runtime,
  localhost control API + GUI (`python -m app.control`).  The GUI talks
  only to the API; there is no endpoint that can send an order.
- `order_send` lives exclusively in `app/brokers/mt5.py` (the only
  MetaTrader5 importer) — AST-enforced.
- Real accounts are refused unconditionally; demo execution requires
  `TRADING_ENABLED=true`, `DRY_RUN=false`, the confirmation phrase and a
  runtime-verified DEMO account.

See [`docs/EXECUTION.md`](docs/EXECUTION.md) for the full reference,
including how to run the opt-in real-MT5 demo test
(`tests/live_mt5`, `AURUMX_LIVE_MT5=1`).  Phase 5 does **not** authorize
real-money trading.

## License

MIT (see `pyproject.toml`).
