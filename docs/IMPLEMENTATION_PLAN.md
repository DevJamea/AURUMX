# AurumX — Implementation Plan

Status: **Phases 1–3 implemented and tested.** This document records the reference-project
review, the architecture decisions, and the phase-by-phase plan. Each phase must pass its
tests before the next phase starts.

---

## 1. Reference Project Review (required first task)

Two existing projects were inspected before any AurumX code was written:

### 1.1 `hmmodtech/snipbot` (crypto, ~1.5k LOC)

A Flask-based crypto bot with 8 "agents" (deterministic TA functions), weighted
consensus, a DCA/grid strategy set, an exchange proxy, a Telegram bot and a static
dashboard, deployed via Docker services on Railway.

**Reusable concepts (extracted into AurumX)**

| Concept | Where it lands in AurumX |
|---|---|
| Weighted agent registry with per-agent `weight` + `enabled` flags | `app/agents` registry + `AGENT_WEIGHTS` configuration (Phase 2) |
| `BaseStrategy` / `Signal` dataclass interface | `BaseAgent` / `AgentResult` interface (Phase 2) |
| Periodic scan loop over pairs/timeframes | `app/worker/trading_loop.py` (Phase 4+) |
| Component separation (engine / proxy / dashboard) | API-first backend + worker + frontend separation |
| Data-source failover idea | `BrokerInterface` abstraction (many brokers later) |

**Architectural weaknesses (deliberately NOT copied)**

1. **Monolithic `main.py`** — all agents are free functions in one module with global
   mutable state (`_latest_evaluations`). AurumX: strict package separation, no global
   mutable analysis state.
2. **Agents hold exchange credentials directly** (`SniperEngine.__init__` takes
   API keys). AurumX: only the broker layer ever sees credentials.
3. **Arbitrary "confidence" numbers** (`"confidence": 88`) with no calibration.
   AurumX: agents emit `signal_strength`, never a fake probability.
4. **DCA / grid / martingale-flavored strategies** that buy falling prices.
   AurumX: mean-reversion is regime-gated, no averaging down, no martingale (§56).
5. **Fake "AI" branding** of deterministic rules. AurumX calls them *Specialized
   Analysis Agents* (§57).
6. **No tests, no decision persistence, no auth, wide-open CORS.**

### 1.2 `DevJamea/TradingAgents-Gold` (gold slice of an LLM framework, ~4k LOC)

A fork of TauricResearch/TradingAgents (LangChain/LangGraph LLM multi-agent framework)
with an added deterministic `tradingagents/gold` package: MT5 adapter, risk gate,
market structure, paper engine, research splits and 100+ test files.

**Reusable concepts (extracted into AurumX)**

| Concept | Where it lands in AurumX |
|---|---|
| Deterministic **risk gate**: output is only APPROVED / NO TRADE, never modifies the decision, fail-closed, check-by-check audit trail | `app/risk` — **implemented in Phase 4** (`HardRiskGate`, 18 checks, see `docs/RISK_GATE.md`) |
| **Fractal swing detection** (left/right window) + HH/HL/LH/LL classification + S/R levels | `app/agents/structure.py` (Phase 2) |
| **Provenance-carrying data models** + `ValidationReport` (errors/warnings) + data-freshness budgets | `app/core/models.py`, `app/market/*` validators (Phase 1) |
| **Fail-closed MT5 adapter**: demo-account verification, re-validation before send, `TRADE_RETCODE_DONE` verification, never fake data when the package is missing | `app/brokers/mt5.py` (Phase 1 read-only, Phase 5 execution) |
| Gold symbol alias set (`XAUUSD`, `GOLD`, …) | `app/market/symbol_discovery.py` (Phase 1, extended to true auto-discovery) |
| Paper engine driven bar-by-bar (`process_bar`) | `app/backtest/simulator.py` + paper engine (Phase 6) |
| **Chronological splits** (in-sample vs out-of-sample) | `app/backtest` walk-forward (Phase 6) |
| Test discipline (100+ focused test files) | `tests/` layout (every phase) |

**Architectural weaknesses (deliberately NOT copied)**

1. **Hard dependency on LangChain/LangGraph + paid LLM APIs** — the deterministic gold
   core cannot run without the LLM framework around it. AurumX: **zero LLM dependency**;
   the whole pipeline is deterministic (§47).
2. **Entanglement with the parent monorepo** (`gold` imports from `tradingagents.graph`).
   AurumX: self-contained project, no framework lock-in.
3. **Yahoo `GC=F` futures proxy** as the price source for spot XAUUSD analysis — a
   provenance mismatch. AurumX reads the **broker's own gold symbol** via MT5.
4. **Single hardcoded `DEFAULT_MT5_SYMBOL="XAUUSD"`** — no real symbol discovery.
   AurumX: automatic broker symbol discovery with verification (§7).
5. **CLI-only** — no API, no dashboard, no long-running worker separation.

---

## 2. AurumX Architecture

### 2.1 Layer rule (absolute)

```text
Market Data (broker adapters)
        ↓
Analysis Agents  (never touch a broker)
        ↓
Decision Engine  (weighted evidence + regime)
        ↓
Risk Gate        (hard gate, code not votes)
        ↓
Execution Service (only layer allowed to trade)
        ↓
Broker Adapter (MT5)  ← the ONLY module importing MetaTrader5
```

Only `app/brokers/*` may import the `MetaTrader5` package. Agents receive already
validated, broker-agnostic models (`MarketTick`, `CandleSeries`, `SymbolSpec`).

### 2.2 Safety model

- Fresh install is **read-only**: `TRADING_ENABLED=false`, `DRY_RUN=true`.
- Real-money execution is refused unless an explicit confirmation phrase is configured
  **and** the account verifies as demo/real at runtime (Phase 5).
- Gold-only: any symbol that is not a broker-equivalent XAUUSD gold symbol is rejected.
- Execution methods on `MT5Broker` are **hard stubs that raise** until Phase 5 — nothing
  built on Phases 1–4 can send an order, even by accident.
- Every MT5 call is logged (structured JSON) with secrets redacted.

### 2.3 Structural adaptations from the spec

The spec's tree is followed with two adaptations (allowed by §6):

- `app/market/mt5_data.py` → **`app/market/data_service.py`** — the market-data service
  is broker-agnostic (it talks to `BrokerInterface`), so an "mt5_" filename would be
  misleading. MT5 specifics live only in `app/brokers/mt5.py`.
- Later-phase packages (`agents`, `decision`, `risk`, `execution`, `backtest`, `storage`,
  `worker`, `api`) exist as documented skeleton packages; their modules are created in
  their own phase, not left as dead stubs.

### 2.4 Key Phase-1 design decisions

| Decision | Rationale |
|---|---|
| Broker-agnostic domain models (`MarketTick`, `Candle`, `CandleSeries`, `SymbolSpec`, `AccountSnapshot`) | MT5 can be replaced later without touching agents (§2) |
| `MetaTrader5` imported lazily inside `MT5Broker.connect()` | The package is Windows-only; the rest of the codebase (tests, backtests, API) runs anywhere |
| `mt5_module` injection parameter on `MT5Broker` | Deterministic tests without a terminal; same pattern will support the paper broker |
| Forming candle always dropped (`include_forming=False` default) | Agents and backtests only ever see **closed** candles → deterministic, no repainting |
| Server-clock offset estimation from the freshest tick | Broker servers run UTC+2/+3; freshness math without offset correction is wrong |
| Validation as reports (`ValidationReport`), not exceptions | Data quality is a first-class output (stale/spread/invalid) consumed by the health monitor and risk gate |
| Timeframe constants resolved via `getattr` with numeric fallback | Works with real package and test fakes alike |
| Pydantic-settings flat config matching the spec's `.env` names | `TRADING_ENABLED`, `RISK_PER_TRADE`, … work exactly as documented in §49 |
| `validate_assignment` + safety validator on the config | Unsafe configurations cannot exist, even via mutation |

---

## 3. Phase Plan

| Phase | Scope | Status |
|---|---|---|
| **1 — Foundation** | structure, config, structured logging, events, domain models, `BrokerInterface`, `MT5Broker` (read-only), symbol discovery + verification, tick/candle validation, market snapshot, diagnostics script, test suite | **DONE — all tests pass** |
| 2 — Agents | Trend, Momentum, Structure, Liquidity, Volatility, Mean Reversion, Macro interface + synthesis agent interface; agent registry & weights config | **DONE — 203 new tests, all passing** |
| 3 — Decision engine | deterministic engine + gates, trade proposals, sizing, SL/TP, risk-state interface, anti-overtrading, decision journal; hardened (explicit TP/RR, strict H4 policy, gate classification, Phase-4 contract) | **DONE — 228 tests (167 + 61 hardening), all passing; proposal-only, NO execution** |
| 4 — Risk | hard risk gate, position sizing, limits, daily loss, spread/exposure checks, kill switch | pending |
| 5 — Execution | order validation → risk check → `order_check` → send → verify → reconcile; BE, partial close, trailing, pending orders; dry-run + paper broker | pending |
| 6 — Backtesting | no-look-ahead simulator, metrics, reports, walk-forward splits, agent performance tracking | pending |
| 7 — Dashboard & API | FastAPI routes, websocket, React frontend, auth | pending |
| 8 — Windows launcher | start/stop/open/diagnostics/logs | pending |
| 9 — Docker/VPS | Dockerfile, compose, reverse proxy docs | pending |
| 10 — Full QA | security tests, end-to-end demo run, final report (§63) | pending |

### Phase 1 acceptance criteria (all met)

- [x] Project structure with separation of concerns (no monolith)
- [x] Safe defaults: `TRADING_ENABLED=false`, `DRY_RUN=true`; unsafe configs raise
- [x] Structured JSON logging with `timestamp/level/component/event/details` + secret redaction
- [x] In-process event bus for observability
- [x] Broker-agnostic domain models with validation reports
- [x] `BrokerInterface` + `MT5Broker` (connect/disconnect, account, symbols, tick,
      candles, positions, orders — read-only; execution stubbed and guarded)
- [x] Gold symbol auto-discovery with full broker-metadata verification
      (visible, trade allowed, digits, point, tick size/value, contract size,
      volume min/max/step, stops & freeze level)
- [x] Gold-only protection (non-gold symbols rejected, incl. silver XAG*)
- [x] Tick validation: zero tick, ask < bid, stale tick, future tick, spread in points
- [x] Candle validation: OHLC sanity, ordering, duplicates, freshness, forming-bar exclusion
- [x] `MarketSnapshot` with data-quality gate (`trading_data_ok`) and session inference
- [x] Read-only diagnostics script (`scripts/diagnostics.py`)
- [x] Unit + integration tests, all passing (**156 passed**), no MT5 terminal
      required (deterministic fake MT5 module, injectable clocks)
- [x] Ruff lint clean

### Phase 2 acceptance criteria (all met)

- [x] Seven agents + MacroAgent interface + Synthesis interface; agents touch
      no MT5/credentials/orders/positions/LLM/internet/global state
      (statically enforced by source-hygiene tests)
- [x] Common contract: name, direction, `signal_strength` (never
      probability/confidence), timeframe, reasons, features, warnings, data
      quality, decision-candle + snapshot timestamps
- [x] M15/H1/H4 roles explicit and tested (ENTRY/STRUCTURE/MACRO)
- [x] TrendAgent: EMA20/50/200, ADX, slope, HH/HL; EMA20>EMA50 alone never
      signals; bullish/bearish/sideways/insufficient tests
- [x] MomentumAgent: no naive RSI<30→BUY; regime-conditional; exhausted-trend,
      range and conflicting-indicator tests
- [x] StructureAgent: confirmed swings only, BOS/CHOCH, documented 2-candle
      confirmation delay, explicit look-ahead (prefix-consistency) tests
- [x] LiquidityAgent (no order-book claims): sweep/rejection/breakout/failed
      breakout patterns, session levels, volume degradation tests
- [x] VolatilityAgent: LOW/NORMAL/HIGH/EXTREME, expansion/contraction,
      contextual (never directional on high volatility)
- [x] MeanReversionAgent: RANGE-only gates, no martingale/averaging/DCA,
      stays neutral in trends (tested), reversal confirmation required
- [x] MacroAgent interface: NEUTRAL/NO_DATA provenance, blackout warning,
      never fabricates
- [x] Regime detection with per-agent relevance (not equal averaging)
- [x] Synthesis contract with disagreement preserved end-to-end
- [x] Determinism, purity and wall-clock independence verified per agent
- [x] Adversarial matrix: NaN/inf/dup/unordered/gapped/short/zero-volume/
      spike/constant/flat/abnormal-ATR — all fail safe (bad-tick guard added)
- [x] Backtest-compatible (no MT5/wall-clock/network anywhere in the layer)
- [x] docs/AGENTS.md reference written; ARCHITECTURE.md updated
- [x] **359 tests passing** (156 Phase-1 + 203 Phase-2), ruff clean

### Phase 3 acceptance criteria (all met)

- [x] Full pipeline MT5→BrokerAdapter→MarketSnapshot→Validation→Regime→
      Agents→Synthesis→DecisionEngine→RiskSizing→TradeProposal→DecisionJournal
      (integration-tested end-to-end with the deterministic FakeMT5)
- [x] Deterministic BUY/SELL/HOLD/ABORT from agent evidence, regime
      compatibility, signal strength, conflict, data quality, timeframe
      alignment and risk constraints — no majority voting
- [x] Every threshold in `DecisionEngineConfig` (validated, documented,
      overridable; reused from `AppConfig` via `from_app_config`) — initial
      engineering parameters, historical validation documented as required
- [x] TF roles H4 context / H1 directional / M15 entry; H4+H1 BUY + M15
      strongly bearish → HOLD `timeframe_conflict`; lower configured
      threshold explicitly permits (tested both ways)
- [x] Disagreement preserved: `supporting_agents`, `opposing_agents`,
      `conflict_score` on every decision; conflict above tolerance → HOLD
- [x] Data-quality gate before BUY/SELL (tick/candles/sessions/spread);
      critical invalid → ABORT (fail-closed), tested per case
- [x] Session OPEN→normal, CLOSED→HOLD, UNKNOWN→ABORT (never assumed OPEN)
- [x] Spread gate vs `max_spread_points` → ABORT `spread_too_high`
- [x] `TradeProposal` pure data with entry/SL/TP/RR/volume/reasons/
      invalidation_conditions/created_at/expires_at — nothing is ever sent
      anywhere; BUY=validated ask, SELL=validated bid (provider protocol,
      historical backtest adapter documented for Phase 6)
- [x] SL hierarchy structure → ATR(2×) → broker minimum; TP by RR target
      or opposing structure; validated against direction/tick size/digits/
      stops level/freeze level (Phase-1 validators reused)
- [x] RR computed pre-rounding; `RR < minimum_rr` → HOLD (both TP methods
      and the raised-minimum path tested)
- [x] Sizing pure calculator floored to volume step (never rounded up),
      below minimum → HOLD `position_size_below_minimum`; martingale
      structurally impossible (closed signature, tested)
- [x] `RiskState` interface (daily loss, limits, open/pending, fingerprints)
      read-only; violations ABORT; engine never mutates it (tested)
- [x] Position limits (max 1 open, max 2 pending) → no new proposal
- [x] Anti-overtrading via setup fingerprint (no time delays); duplicate →
      HOLD `duplicate_setup`; new setup allowed (tested)
- [x] Proposal expiry VALID/EXPIRED as a pure function of time
- [x] Journal records every evaluation with full provenance (agents,
      synthesis, gates, config, snapshot ref) — no credentials (tested);
      in-memory + SQLite (WAL, indexed) backends
- [x] No look-ahead: `decision(prefix) == decision(full series @ prefix
      time)` — future candles present in the snapshot cannot change a
      finalized decision (tested with future-anchored series)
- [x] Determinism: repeated evaluation, rebuilt engine and reordered agent
      rosters all reproduce identical decisions (tested)
- [x] Scenarios A–J pinned (BUY/SELL/timeframe-conflict/spread/invalid
      tick/closed/unknown/insufficient RR/daily loss/position limit)
- [x] Decision layer contains no MT5/network/wall-clock/GUI (source-hygiene
      AST tests extended to `app/decision` + `app/risk`)
- [x] docs/DECISIONS.md written; ARCHITECTURE.md/README updated
- [x] **526 tests passing** (156 + 203 + 167), ruff clean
- [x] **Phase 3 does NOT execute trades** — no `order_send` anywhere in the
      decision layer; execution remains Phase 5 behind the risk gate

### Phase-3 hardening pass (complete)

Post-review hardening of the decision layer (no Phase-4 work, no execution):

- [x] TP/RR semantics explicit: `TPMethod.TP_BY_RR` (default, unchanged) /
      `TPMethod.TP_BY_STRUCTURE`; the RR gate's self-referential nature for
      TP_BY_RR is documented on every such proposal; structure-derived TP
      produces a market-determined RR that the `minimum_rr` gate genuinely
      enforces (engine-level tests prove HOLD/BLOCK at the threshold)
- [x] Invalid structure targets (wrong side of entry / below broker
      minimum) are used verbatim, flagged and rejected — never clamped
- [x] Missing H4 is now a policy decision: strict default
      (`require_h4=true`, `allow_missing_h4_renormalization=false`) →
      HOLD `missing_primary_context`; explicit opt-in renormalization
      journals the missing timeframe and the effective weights in force;
      both modes deterministic and tested
- [x] Every Phase-3 gate classified (DATA SAFETY / MARKET CONDITION /
      DECISION QUALITY / RISK CONTROL / PROPOSAL VALIDATION) —
      docs/DECISIONS.md §13
- [x] Phase-4 boundary documented and contracted: `app/risk/gate.py`
      (`RiskGate.evaluate(proposal, risk_state, account_state) ->
      RiskDecision{APPROVED|REJECTED|EMERGENCY_STOP}`, `REQUIRED_CHECKS`
      with the 14 independent verifications, `AccountState` input model);
      interface only — no implementation
- [x] Boundary enforced by tests: decision layer cannot reach the gate or
      any execution surface; no MT5/network/wall-clock in decision or risk
      layers (AST checks)
- [x] **587 tests passing** (526 + 61 hardening), ruff clean

### Phase 4 acceptance criteria (all met)

The hard risk gate — an independent barrier between the decision layer and
(any future) execution. Full reference: `docs/RISK_GATE.md`.

- [x] **Fail-closed barrier**: missing evidence (equity, spread, exposure,
      margin, symbol metadata) ⇒ REJECTED, never APPROVED; unknown
      configurations (unset `max_total_exposure`, unset spread limit) ⇒
      REJECTED; `RiskGateConfig` safe default `trading_enabled=False`
- [x] **Contract** (`app/risk/gate.py`): `RiskGate` Protocol,
      `REQUIRED_CHECKS` (14), `RiskCheck{status: PASS/FAIL/WARN/NOT_EVALUATED,
      severity: CRITICAL/ADVISORY}`, `RiskDecision` with deterministic
      `gate_decision_id`, warnings, risk amount, exposure, config snapshot,
      kill/emergency state; `AccountState` evidence semantics (monetary
      fields default `None`, `trade_allowed` defaults `False`)
- [x] **Implementation** (`app/risk/engine.py`): `HardRiskGate`, 18 ordered
      checks (`IMPLEMENTED_CHECKS`), tiered precedence — emergency stop >
      kill switch > trading disabled > account > safety checks > approve;
      halted tiers record remaining checks NOT_EVALUATED (visible)
- [x] **Independent risk math**: `loss_per_lot = |entry − SL| / tick_size ×
      tick_value`; never reads `proposal.sizing.risk_amount` (lying-proposal
      test); never rounds volume/risk into validity; exposure =
      current + volume × (entry / tick_size) × tick_value
- [x] **Gold-only reuse**: `is_gold_symbol()` from Phase 1 — no second
      implementation; symbol must ALSO have registered verified metadata
- [x] **Kill switch & emergency stop** (`app/risk/kill_switch.py`): two
      distinct caller-supplied halts → EMERGENCY_STOP, statuses
      STANDBY/ACTIVE/TRIGGERED/RESET_REQUIRED, `KillSwitchStore` Protocol
      for future GUI/VPS persistence, `halt_flags()` bridge; no invented
      auto-triggers
- [x] **Risk events** (`app/risk/events.py`): RISK_APPROVED / RISK_REJECTED /
      EMERGENCY_STOP / KILL_SWITCH_ACTIVE as data objects via optional
      `event_sink` — no Telegram/HTTP/network
- [x] **No mutation**: proposal byte-identical before/after evaluation
      (model_dump + model_dump_json tests); AST scan forbids attribute
      assignment on proposals in `app/risk`
- [x] **No martingale/recovery/loss-scaling anywhere** (AST identifier
      scans + risk-math purity test: consecutive losses can only block)
- [x] **Determinism**: repeated + rebuilt evaluations byte-identical;
      no uuid/random/wall-clock/network/MT5 in `app/risk` (AST tests)
- [x] Full matrix tested: symbols (incl. XAGUSD/GOLDEN/GC=F), geometry
      (NaN/inf/zero/off-grid/under-minimum/wrong side), risk
      (below/exact/above/lying proposal), exposure, daily loss (at-limit
      rejects), consecutive losses, position/pending boundaries, spread
      (missing/50/51), volume (min/max/step/invalid), account evidence,
      adversarial combos (valid BUY + each violation, kitchen sink),
      precedence, events, pipeline integration (engine → gate)
- [x] **789 tests passing** (587 + 202 Phase-4), ruff clean
- [x] **Phase 4 does NOT execute trades** — gate output is data; execution
      remains Phase 5

### Phase 5 acceptance criteria

The execution layer + Windows control plane.  Full reference:
`docs/EXECUTION.md`.  Verdict: **offline proof complete; real-MT5 demo
verification pending** (requires a Windows MT5 terminal — the sandbox
environment has none; the opt-in harness is `tests/live_mt5` with
`AURUMX_LIVE_MT5=1`).

- [x] **MT5 execution isolated behind the adapter**: `order_check`/
      `order_send` appear only in `app/brokers/mt5.py`
      `place_market_order` — AST-enforced across `app/**`
- [x] **order_check precedes order_send**; a failed check provably
      prevents the send (fake-terminal call-order tests)
- [x] **Retcodes explicitly verified** using the installed package's
      constants (numeric fallbacks documented); unknown retcodes fail
      closed; every category tested (DONE, PARTIAL, REJECT, INVALID*,
      MARKET_CLOSED, NO_MONEY, REQUOTE, TIMEOUT, unknown)
- [x] **DRY_RUN never calls order_send** — proven by call-counting at
      unit, service and integration level; DRY_RUN uses the same pipeline
      (request construction, validation, risk verification, journal) and
      fabricates no broker identifiers
- [x] **Demo execution explicitly opt-in**: TRADING_ENABLED + DRY_RUN=
      false + REAL_TRADING_CONFIRMED phrase + runtime-verified DEMO
      account; real accounts blocked unconditionally; no single boolean
      bypasses the chain
- [x] **Reconciliation** detects MATCHED / MISSING_IN_MT5 /
      MISSING_IN_JOURNAL / MISMATCH / UNKNOWN with documented tolerances;
      mismatches halt execution until explicitly resolved (audited);
      nothing silently repaired or auto-adopted
- [x] **Unknown execution state is never success** — accepted-but-
      unverified fills are UNKNOWN, not FILLED; journal accuracy pinned
- [x] **Proposal immutable** through execution; request derived verbatim
      (§16 consistency tests: symbol/direction/volume/SL/TP/identity)
- [x] **No execution bypass**: `execute(request, decision)` requires an
      APPROVED RiskDecision with matching ids/fingerprint; GUI cannot
      reach MT5 (AST) and the control API has no execution endpoint
- [x] **Windows control plane**: `python -m app.control` — engine runtime,
      localhost control API, static GUI (same-origin only), safe defaults
      (READ_ONLY, stopped), dry-run ratchet (de-escalation only)
- [x] **Live-MT5 suite separated and opt-in** (`tests/live_mt5`,
      collect-ignored without `AURUMX_LIVE_MT5=1`)
- [x] **No retries** after any outcome (§42); no martingale/recovery
      anywhere (AST identifier scans + risk-math purity)
- [x] **1048 tests passing** (789 baseline + 259 Phase-5), ruff clean,
      `python -m compileall .` clean, `python -m app.control --help` works
- [x] Phase-4 limits preserved (the gate is unchanged and still the only
      approval authority; execution re-validates independently)

### Validation semantics note

`valid` and `fresh` are deliberately orthogonal: a *stale* tick/candle series is
**valid** (sane data) but **not fresh** — staleness is recorded as an ERROR in
the `ValidationReport` (it blocks trading via `trading_data_ok`) while `valid`
reflects data sanity only. This lets the dashboard say "data OK but market
closed" instead of "data corrupted".
