# Phase 3 — Decision Record

This document records every consequential design decision of the Phase-3
decision engine, so a reviewer can challenge the reasoning rather than the
code. Phase 3 is **proposal-only**: the engine decides, never executes.
There is no path from `DecisionEngine` to `order_send` — the first execution
code arrives with Phase 4+ behind the risk gate and the broker adapter.

Pipeline implemented:

```text
MT5 → BrokerAdapter → MarketSnapshot → Validation → Regime → Agents
    → Synthesis → DecisionEngine (17 gates) → RiskSizing → TradeProposal
    → DecisionJournal                 (→ RiskGate is Phase 4 — §14, NOT built)
```

> **Hardening pass (this revision):** §4 H4 policy made explicit (strict
> default), §6 TP/RR semantics made explicit (`TP_BY_RR` /
> `TP_BY_STRUCTURE`), §13 gate classification, §14 the Phase-4 RiskGate
> contract (interface only).

---

## 1. HOLD vs ABORT — the fail-closed split

Two ways to not trade, with different meanings:

| | HOLD | ABORT |
|---|---|---|
| meaning | market conditions are not good enough *right now* | inputs or safety state are broken — deciding is impossible/unsafe |
| examples | `no_edge`, `below_threshold`, `weak_signal`, `timeframe_conflict`, `insufficient_rr`, `market_closed`, `duplicate_setup`, `missing_entry_timeframe` (M15), `insufficient_candles`, `conflict_exceeds_tolerance` | `missing_symbol`, `invalid_tick`, `stale_tick`, `stale_candles:*`, `invalid_series:*`, `no_candle_data`, `missing_primary_timeframe_h1`, `session_unknown`, `spread_too_high`, `missing_equity`, `daily_loss_limit`, `consecutive_losses`, `position_limit`, `pending_order_limit`, `invalid_entry_price` |
| next cycle | may trade when conditions change | must stay out until the fault clears |

Spread exceeding `max_spread_points` is **ABORT**, not HOLD: a snapshot with
an unusable spread cannot support any price-based decision (§8). A CLOSED
session is HOLD (the market itself may reopen normally); an UNKNOWN session
is **ABORT** — the engine never assumes UNKNOWN means OPEN (§7).

Gate order is data-quality → session → freshness/report → spread → risk
state → analysis (H1 depth, M15 presence, H4 policy) → edge → signal →
alignment → conflict → duplicate → entry → RR → sizing → geometry. Inputs are validated before risk state;
risk state before any agent work (agents do not even run when the risk
state blocks — visible in the journal as empty `agent_results`).

## 2. Thresholds are engineering defaults, not statistics

Every threshold lives in `DecisionEngineConfig` (`app/decision/config.py`)
with bounds and documentation; the engine contains no magic numbers:

| key | default | role |
|---|---|---|
| `buy_threshold` / `sell_threshold` | 1.00 | minimum relevance-weighted side score |
| `net_threshold` | 0.80 | minimum side − opposing score |
| `minimum_signal_strength` | 0.45 | strongest single supporting agent |
| `minimum_timeframe_alignment` | 0.80 | weighted multi-TF agreement |
| `max_conflict` | 0.35 | max opposing share of weighted evidence |
| `minimum_rr` / `target_rr` | 1.50 / 2.00 | reward:risk floor / TP construction |
| `atr_stop_multiple` | 2.00 | ATR SL distance |
| `risk_per_trade_pct` | 0.50 | fraction of equity risked per trade |
| `max_spread_points` | 50 | quoted spread ceiling |
| `max_open_positions` / `max_pending_orders` | 1 / 2 | exposure limits |
| `max_consecutive_losses` | 3 | stand-down after a losing streak |
| `daily_loss_limit_pct` | 2.00 | daily budget when RiskState has no explicit limit |
| `proposal_ttl_minutes` | 15 | proposal validity window |

**These values are initial engineering parameters.** They were chosen to be
deliberately conservative, not fitted to data. They **require validation
against historical data** (walk-forward, Phase 6) before any live use.
Operators can override all of them; `from_app_config` reuses the Phase-1
`AppConfig` risk fields so there is one configuration surface.

## 3. Deciding from evidence — no majority voting

`synthesize()` (Phase 2) turns agent results into a weighted evidence
summary; the engine then applies the gates above. There is no vote counting:
a single strong opposing agent can HOLD a trade (`conflict_exceeds_tolerance`)
while the disagreement is *preserved* — `Decision.supporting_agents`,
`opposing_agents`, and `conflict_score` (opposing weighted share) are
computed and journaled on every decision, including BUYs and SELLs.

Direction is taken from the synthesis side score; the leading agent
(max `signal_strength × regime relevance`) names the setup type in the
proposal fingerprint.

## 4. Timeframe roles and the conflict policy (§3)

H4 = context (weight .30), H1 = directional primary (.45), M15 = entry (.25).

* An opposing **M15** caps agreement at .75 < .80 → `timeframe_conflict`
  HOLD by default. A configured lower `minimum_timeframe_alignment`
  (e.g. 0.70) explicitly permits the trade — policy, not accident.
* **Missing H1** → ABORT `missing_primary_timeframe_h1` (no direction
  without the primary timeframe).
* **Missing M15** → HOLD `missing_entry_timeframe` (no entry-timing
  context → no proposal, but the data isn't broken).
* **Missing H4** → HOLD `missing_primary_context` — **strict by default**
  (hardening §2: no implicit fallback). The operator may explicitly enable
  `decision.timeframe.allow_missing_h4_renormalization=true`, in which case
  alignment renormalizes over the present timeframes (H1+M15 agreeing →
  1.0) and **every decision and journal record carries the fact**: the
  missing timeframe (`alignment_detail.missing`), the renormalization flag
  and the effective weights in force (`H1=0.643, M15=0.357`), plus a
  warning on the decision. Both modes are deterministic and tested; the
  policy itself is journaled in `config_snapshot.timeframe`.

Alignment reads are EMA20/EMA50 direction per timeframe (Phase-2 features);
a read is "strong" when the EMA gap exceeds `strong_alignment_atr` ATRs.

## 5. Live entry vs backtest entry (§10)

`TickEntryProvider` prices a live BUY at the **validated ask** and a SELL at
the **validated bid** — never a historical close, never a mid. The entry is
re-checked (`invalid_entry_price` ABORT if non-finite/≤0). Backtesting
(Phase 6) will plug a `HistoricalEntryProvider` (next-bar open / bar close
per the backtest contract) into the same `EntryPriceProvider` protocol —
the engine code is identical, only the provider differs. The journal records
which provider produced the entry via the proposal's entry fields.

## 6. SL/TP methodology (§11/§12)

`compute_levels` implements the hierarchy:

1. **Structure invalidation** — the confirmed swing level from the structure
   agent, used only when it is *valid for the direction*, at least the
   broker minimum distance away, and no farther than the ATR stop.
2. **ATR stop** — `entry ± atr_stop_multiple × ATR(H1)`.
3. **Broker minimum** — `max(stops_level, freeze_level) × point`, the
   documented fallback of last resort (never an arbitrary
   "entry − 10 points").

TP method is an explicit enum (`TPMethod`, config `tp_method`):

* **`TP_BY_RR`** (`"rr"`, **the default — unchanged by the hardening
  pass**): TP = `entry ± target_rr × risk_distance`. Deterministic and
  geometry-clean, but the resulting RR is derived from the target ratio
  itself, so the `minimum_rr` gate is **self-referential** for this
  method: passing it proves valid geometry and broker distances, **not a
  market-quality RR opportunity**. Every such proposal carries an explicit
  note saying so (in `proposal.reasons`, hence in the journal). The gate
  still runs for this method — it catches config contradictions
  (`minimum_rr > target_rr`) and tick-snapping drift — but no
  market-quality claim is made from it.
* **`TP_BY_STRUCTURE`** (`"structure"`): TP = the opposing confirmed swing
  level (last swing high for a long, last swing low for a short), derived
  **independently** of the risk distance. The actual RR is computed
  afterwards, so `minimum_rr` is a **genuine constraint**: a structure
  target that offers less reward than the configured minimum HOLDs with
  `insufficient_rr` (tested with real engine runs — e.g. swing 1.2 above
  entry vs a 4.7 ATR stop → RR ≈ 0.25 → HOLD; swing 11.8 above → RR 2.51
  → BUY). A structure level on the wrong side of entry, or closer than
  the broker minimum, is an **invalid target**: it is used verbatim,
  flagged with a note, and rejected by the geometry gate — never silently
  clamped or replaced. When no opposing level exists at all, the
  documented fallback is the RR target (never a silent methodology swap).

All levels are snapped to the symbol's tick grid and validated by the
Phase-1 validators (direction, min stop distance, tick alignment).
`risk_distance`, `reward_distance`, `risk_reward` are computed from raw
values **before** any rounding, and `RR < minimum_rr` → HOLD (never a BUY
with a bad payoff).

## 7. Sizing (§14)

Pure calculator, closed signature `(equity, risk_per_trade_pct, entry,
stop_loss, symbol)`:

```text
risk_amount   = equity × risk_per_trade_pct / 100
loss_per_lot  = risk_distance / tick_size × tick_value
raw_volume    = risk_amount / loss_per_lot
volume        = floor(raw_volume / volume_step) × volume_step   # NEVER rounded up
```

Floored to the broker step (never rounded up — the risk budget is a cap);
below `volume_min` → infeasible `volume_below_minimum` → HOLD
`position_size_below_minimum` (the minimum is never forced); above
`volume_max` → clamped and flagged. The full `RiskSizing` provenance
(raw, normalized, monetary, percentage) rides on the proposal.

**No martingale is possible**: the signature contains no loss history,
streak counters or multipliers — structurally, not by convention.
`RiskState` (which carries loss information) is an engine *gate* input, not
a sizing input; loss history can only *reduce* activity (consecutive-loss
stand-down, daily budget), never increase risk.

## 8. Risk state (§15/§16)

`RiskState` is supplied by the caller per evaluation; the engine reads it
and never mutates it. Daily limit = `daily_loss_limit` if provided, else
`equity × daily_loss_limit_pct / 100`. Violations ABORT. Position limits
(`max_open_positions`, `max_pending_orders`, `max_consecutive_losses`) ABORT
before agents run.

## 9. Anti-overtrading without clocks (§23/§24)

A setup fingerprint (`sha256` of symbol | last closed H1 candle | direction
| regime | structure bias | last structure event | leading agent, first 16
hex chars) identifies *the same setup*. If it is already in
`RiskState.active_setup_fingerprints` → HOLD `duplicate_setup`. There are
**no time-based cooldowns** — a cooldown would corrupt backtest determinism.
A new candle, direction, regime or structure event mints a new fingerprint;
the same candle can only produce one proposal per direction.

## 10. Determinism and no look-ahead (§20/§21)

* `now` is an explicit `evaluate()` parameter; the engine has no wall-clock,
  RNG, network or filesystem access (AST-enforced by the contract tests).
* Agent rosters are sorted by name at engine construction, so roster order
  can never leak into a decision.
* Every series is sliced to candles **closed by `now`** before analysis
  (`_slice_to_now`), so `decision(prefix) == decision(full data, now)`.
* `decision_id = sha256(symbol | now | fingerprint)` — unique per
  evaluation; the fingerprint is stable per setup (tested separately).

## 11. Journal (§19)

Every evaluation — BUY, SELL, HOLD, ABORT — is recorded
(`DecisionRecord`): decision id, timestamp, symbol, regime, session,
spread, conflict, alignment (score *and* structured detail: missing
timeframes, renormalization flag, effective weights), supporting/opposing
agents, **every agent result**, the synthesis payload, all gate outcomes,
reasons, the full proposal + sizing, fingerprint, setup type, a snapshot
summary reference and the config snapshot in force. A record answers *"why did the bot
decide this"* without re-running anything. Backends: `InMemoryDecisionJournal`
(tests/backtests) and `SQLiteDecisionJournal` (WAL, flat schema ready to
become PostgreSQL per spec §38). Records are JSON-safe and
credential-free by construction — no password/token/login-shaped keys
anywhere in a record (tested).

## 12. Proposal lifecycle (§17/§18)

`TradeProposal` is pure data. Every proposal carries explicit
`invalidation_conditions` (structure invalidated, spread exceeded, regime
changed, signal expired, SL distance invalid, data quality degraded) and a
validity window `created_at → expires_at` (`proposal_ttl_minutes`).
`status(at)` is a pure function — VALID through `expires_at` (inclusive),
EXPIRED after; no global state. Expiry marks the proposal *stale*, never
cancels anything at a broker — Phase 3 sends nothing anywhere.

## 13. Phase-3 gate classification (hardening §3)

Every gate, its outcome and its **class**. The class tells Phase 4 which
territory it must re-verify independently (all RISK CONTROL and PROPOSAL
VALIDATION logic is deliberately duplicated in the Phase-4 barrier —
defense in depth, never "the engine already checked it").

| Gate | Rejection reason(s) | Outcome | Class |
|---|---|---|---|
| data_validity | `missing_symbol`, `invalid_tick`, `no_candle_data`, `invalid_series:*`, `missing_primary_timeframe_h1` | ABORT | DATA SAFETY |
| session | `session_unknown` | ABORT | DATA SAFETY |
| session | `market_closed` | HOLD | MARKET CONDITION |
| freshness / report | `stale_tick`, `stale_candles:*`, `validation_errors` | ABORT | DATA SAFETY |
| spread | `spread_too_high` | ABORT | MARKET CONDITION |
| risk_state | `missing_equity` | ABORT | DATA SAFETY (input) |
| risk_state | `daily_loss_limit`, `consecutive_losses` | ABORT | RISK CONTROL |
| risk_state | `position_limit`, `pending_order_limit` | ABORT | RISK CONTROL |
| analysis | `insufficient_candles` | HOLD | DATA SAFETY |
| analysis | `missing_entry_timeframe` (M15) | HOLD | DATA SAFETY |
| timeframe_policy | `missing_primary_context` (H4, strict default) | HOLD | DATA SAFETY (policy) |
| edge | `no_edge`, `below_threshold` | HOLD | DECISION QUALITY |
| signal_strength | `weak_signal` | HOLD | DECISION QUALITY |
| timeframe_alignment | `timeframe_conflict` | HOLD | DECISION QUALITY |
| conflict | `conflict_exceeds_tolerance` | HOLD | DECISION QUALITY |
| duplicate_setup | `duplicate_setup` | HOLD | RISK CONTROL (anti-overtrading) |
| entry_price | `invalid_entry_price` | ABORT | DATA SAFETY |
| reward_risk | `insufficient_rr` | HOLD | PROPOSAL VALIDATION |
| position_size | `position_size_below_minimum` | HOLD | PROPOSAL VALIDATION |
| geometry | `invalid_geometry` | HOLD | PROPOSAL VALIDATION |

Classification rules used:

* **DATA SAFETY** — the inputs are missing, stale, insane or against
  policy; no trustworthy decision is possible. Always ABORT except where
  the *absence* of data is a normal market state (HOLD).
* **MARKET CONDITION** — the market itself (session, spread) makes trading
  inappropriate right now; the data is fine, the world isn't.
* **DECISION QUALITY** — evidence quality gates (edge, strength,
  alignment, conflict). Pure Phase-3 territory; Phase 4 does NOT re-check
  these (they are opinions, not safety).
* **RISK CONTROL** — exposure/loss/fingerprint limits. Phase 3 checks
  them to avoid *proposing*; **Phase 4 re-verifies every one independently
  before execution**.
* **PROPOSAL VALIDATION** — the constructed trade's geometry, RR and
  volume. Phase 4 re-verifies SL/TP presence, geometry and volume limits
  on its own evidence.

## 14. Phase-4 boundary — the RiskGate contract (hardening §4/§5)

**Not implemented.** The contract lives in `app/risk/gate.py`
(`RiskGate` Protocol, `RiskDecision`/`RiskCheck` models,
`REQUIRED_CHECKS`) and is enforced by boundary tests: `app/decision`
never imports or names the gate, and no Phase-3 code can call an
implementation.

```text
DecisionEngine        Phase 3 — decision correctness (this document)
      ↓
TradeProposal         pure data; nothing sent anywhere
      ↓
RiskGate              Phase 4 — INDEPENDENT safety barrier
      ↓
RiskDecision          APPROVED / REJECTED / EMERGENCY_STOP
      ↓
execution service     Phase 5 — only APPROVED proposals
```

The pure contract:

```python
class RiskGate(Protocol):
    def evaluate(
        self,
        proposal: TradeProposal,
        risk_state: RiskState,
        account_state: AccountState,
    ) -> RiskDecision: ...
```

* Inputs are complete and pure: the proposal carries entry/SL/TP/RR,
  volume, sizing provenance, symbol, expiry and fingerprint; `RiskState`
  carries equity, daily loss/limit, losing streak, open/pending counts and
  active fingerprints; `AccountState` (new, `app/risk/state.py`) carries
  balance/equity/margin, exposure notional, `trade_allowed`,
  `emergency_stop_active` and `kill_switch_active`. No wall clock, no
  network, no broker calls — the same gate evaluates backtest proposals
  unchanged.
* Output is one of exactly `APPROVED`, `REJECTED` (with named failed
  `RiskCheck`s) or `EMERGENCY_STOP` (operator halt active). The gate
  **never modifies a proposal** — a gate that rewrites trades is a gate
  that can be tuned into unsafety.
* The gate is **fail-closed**: a check that cannot be evaluated (missing
  data) is a FAILED check.

**Phase 4 must independently verify (REQUIRED_CHECKS, hardening §5) —
never assuming Phase 3 already checked anything:**

1. `max_risk_per_trade` — proposal risk ≤ configured % of equity
2. `max_total_exposure` — open exposure + proposal ≤ limit
3. `daily_loss_limit` — today's loss budget
4. `consecutive_loss_protection` — losing-streak stand-down
5. `max_open_positions`
6. `max_pending_orders`
7. `max_spread` — current quoted spread vs limit
8. `symbol_restriction` — gold-only
9. `volume_limits` — volume_min ≤ vol ≤ volume_max, step-aligned
10. `sl_presence` — a usable stop loss exists
11. `tp_validity` — TP present and geometrically valid
12. `emergency_stop` — persistent operator stop
13. `kill_switch` — global kill switch
14. `account_safety` — account state allows trading (margin, trade mode)

The principle:

```text
Phase 3 = decision correctness   (is this a good trade to propose?)
Phase 4 = independent safety barrier (is this trade SAFE to execute?)
```

The engine's risk gates exist so the system does not *propose* unsafe
trades; the RiskGate exists so it cannot *execute* them. The duplication
is deliberate (defense in depth).

## Known limitations

* Thresholds unvalidated on historical data (see §2) — Phase 6 walk-forward
  will tune them.
* With `TP_BY_RR` (the default) the RR gate is self-referential (RR ≈
  target by construction) — documented on every such proposal; use
  `TP_BY_STRUCTURE` for a market-derived RR (see §6).
* H1-only regime/volatility inputs: the detector reads the primary
  timeframe; M15/H4 influence decisions through alignment and agent
  features only.
* The `InMemoryDecisionJournal` is bounded (deque) — production uses SQLite.
* The repo does not pass `ruff format --check` (never did — Phase 1/2
  established `ruff check` as the lint bar); reformatting 60+ files is
  deferred to a dedicated style-only commit.
* H4 is strict by default (§4): a broker feed hiccup on H4 now HOLDs the
  engine. Operators who accept the quality trade-off must opt in
  explicitly; renormalized decisions are fully journaled (missing
  timeframe + effective weights).
