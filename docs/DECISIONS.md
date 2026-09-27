# Phase 3 — Decision Record

This document records every consequential design decision of the Phase-3
decision engine, so a reviewer can challenge the reasoning rather than the
code. Phase 3 is **proposal-only**: the engine decides, never executes.
There is no path from `DecisionEngine` to `order_send` — the first execution
code arrives with Phase 4+ behind the risk gate and the broker adapter.

Pipeline implemented:

```text
MT5 → BrokerAdapter → MarketSnapshot → Validation → Regime → Agents
    → Synthesis → DecisionEngine (14 gates) → RiskSizing → TradeProposal
    → DecisionJournal
```

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
state → analysis → edge → signal → alignment → conflict → duplicate →
entry → RR → sizing → geometry. Inputs are validated before risk state;
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
* **Missing H4** → proceed: alignment renormalizes over the present
  timeframes (H1+M15 agreeing → 1.0). H4 is context, not a prerequisite.
  Documented trade-off: H4 context improves quality but demanding it would
  make the engine unusable whenever the broker's H4 feed hiccups.

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

TP: `tp_method="rr"` (default) projects `target_rr × risk_distance`;
`tp_method="structure"` targets the opposing confirmed swing level
(falling back to the RR target when no level exists — documented).
All levels are snapped to the symbol's tick grid and validated by the
Phase-1 validators (direction, min stop distance, tick alignment).
`risk_distance`, `reward_distance`, `risk_reward` are computed from raw
values **before** any rounding, and `RR < minimum_rr` → HOLD (never a BUY
with a bad payoff). With `tp_method="rr"` the RR is ~`target_rr` by
construction, so the `minimum_rr` gate only binds when `minimum_rr >
target_rr` or `tp_method="structure"` — both paths are tested.

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
spread, conflict, alignment, supporting/opposing agents, **every agent
result**, the synthesis payload, all gate outcomes, reasons, the full
proposal + sizing, fingerprint, setup type, a snapshot summary reference
and the config snapshot in force. A record answers *"why did the bot
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

## Known limitations

* Thresholds unvalidated on historical data (see §2) — Phase 6 walk-forward
  will tune them.
* `tp_method="rr"` makes the RR gate mostly tautological (RR ≈ target by
  construction); it binds under `tp_method="structure"` or when
  `minimum_rr > target_rr`.
* H1-only regime/volatility inputs: the detector reads the primary
  timeframe; M15/H4 influence decisions through alignment and agent
  features only.
* The `InMemoryDecisionJournal` is bounded (deque) — production uses SQLite.
* The repo does not pass `ruff format --check` (never did — Phase 1/2
  established `ruff check` as the lint bar); reformatting 60+ files is
  deferred to a dedicated style-only commit.
* Missing H4 proceeds (§4) — the journal records the missing timeframe in
  the snapshot reference so post-hoc analysis can filter those decisions.
