# AurumX Risk Gate (Phase 4)

Status: **implemented** — `app/risk` — 789 tests passing, proposal-only.
**Phase 4 does NOT execute trades. Phase 5 is NOT started.**

The risk gate is the hard safety barrier between the decision layer and
(any future) execution:

```text
DecisionEngine ──▶ TradeProposal ──▶ HardRiskGate ──▶ RiskDecision
                                     (app/risk)        APPROVED / REJECTED /
                                                       EMERGENCY_STOP
```

The gate receives a **finished** proposal and independently decides whether
it is SAFE to execute. It is code, not a vote: no agent score, signal
strength or trend conviction can push an unsafe proposal through it.
It never proposes trades, never repairs them, and never executes them.

## 1. Contract (`app/risk/gate.py`)

```python
class RiskGate(Protocol):
    def evaluate(
        self, proposal: TradeProposal, risk_state: RiskState,
        account_state: AccountState,
    ) -> RiskDecision: ...
```

* **`RiskCheck`** — one check's result: `name`, `status`
  (`PASS` / `FAIL` / `WARN` / `NOT_EVALUATED`), `severity`
  (`CRITICAL` / `ADVISORY`), `reason`, `observed_value`, `limit`.
  A check blocks only on `FAIL` (or `WARN` for `CRITICAL`-severity checks).
* **`RiskDecision`** — `action` (`APPROVED` / `REJECTED` / `EMERGENCY_STOP`),
  the full ordered check list, `reasons` (deterministic strings),
  `warnings`, `risk_amount` (the gate's own computation), `exposure`
  (`current` / `proposed` / `total` / `limit`), `gate_decision_id`
  (deterministic SHA-256-derived, no uuid/random/clock), `config_snapshot`,
  kill-switch / emergency-stop state, and the derived partitions
  `passed_checks` / `failed_checks` / `not_evaluated_checks`.
* **`AccountState`** — the evidence input. Monetary evidence (`equity`,
  `balance`, margin fields, `open_positions_notional`, `spread_points`,
  position/pending counts) defaults to `None`: **missing evidence is
  evidence of nothing** and fails closed. `trade_allowed` defaults to
  `False` — permission must be affirmative, never assumed.
* **`REQUIRED_CHECKS`** — the 14 contract-level check names every
  implementation must cover; `app/risk/engine.py` implements these plus 4
  more granular ones (18 total, `IMPLEMENTED_CHECKS`).

## 2. Fail-closed principle

Every unknown fails toward REJECTED, never toward APPROVED:

| Situation | Result |
|---|---|
| Missing equity evidence | REJECTED (`max_risk_per_trade`) |
| Missing spread evidence | REJECTED (`max_spread`) — never substituted with 0 |
| Missing exposure evidence | REJECTED (`max_total_exposure`) |
| Missing/zero symbol metadata | REJECTED (`symbol_restriction`) |
| Exposure limit not configured | REJECTED (`max_total_exposure` not configured) |
| Margin evidence missing (policy on) | REJECTED (`margin_safety`) |
| Conflicting evidence (state vs account) | REJECTED (inconsistent evidence) |
| `trade_allowed=False` or absent | REJECTED (`trading_not_allowed`) |
| Trading globally disabled | REJECTED (`trading_enabled`) — the safe default |

The gate never **repairs** anything: a volume of 0.105 is rejected, not
rounded to 0.11; an SL 0.1 points away is rejected, not widened; the
proposal object is guaranteed byte-identical before and after evaluation
(pinned by tests in both `model_dump` and `model_dump_json` form).

## 3. The 18 checks (`app/risk/engine.py`)

| # | Check | Meaning |
|---|---|---|
| 1 | `emergency_stop` | operator emergency stop active → EMERGENCY_STOP |
| 2 | `kill_switch` | kill switch engaged → EMERGENCY_STOP |
| 3 | `trading_enabled` | global enable; `RiskGateConfig` safe default is `False` |
| 4 | `account_safety` | `trade_allowed` + evidence consistency between `RiskState` and `AccountState` |
| 5 | `symbol_restriction` | `is_gold_symbol()` (Phase-1 protection, no second implementation) AND a registered, verified `SymbolSpec` |
| 6 | `direction_valid` | only BUY / SELL executable; NEUTRAL (and defensively HOLD/ABORT) rejected |
| 7 | `entry_valid` | finite, positive, on the symbol's tick grid |
| 8 | `sl_presence` | SL exists, correct side of entry (BUY: SL < entry < TP; SELL: TP < entry < SL), finite, on grid, ≥ broker minimum stop distance, non-zero distance |
| 9 | `tp_validity` | TP exists, correct side, finite, on grid, ≥ broker minimum |
| 10 | `volume_limits` | > 0, finite, within symbol min/max, aligned to step — never rounded into validity |
| 11 | `max_risk_per_trade` | independent risk computation (below), ≤ min(pct-of-equity budget, absolute cap if set); WARN at ≥ 90 % of budget |
| 12 | `max_total_exposure` | current open notional + proposed notional ≤ configured limit; **limit must be explicitly configured** |
| 13 | `daily_loss_limit` | monetary daily loss vs explicit amount or pct-of-equity; **at the limit rejects** |
| 14 | `consecutive_loss_protection` | ≥ limit consecutive losses rejects; losses never scale risk (no martingale) |
| 15 | `max_open_positions` | open positions (state + account must agree) + 1 ≤ limit |
| 16 | `max_pending_orders` | pending orders ≤ limit |
| 17 | `max_spread` | spread in points ≤ limit; missing evidence rejects |
| 18 | `margin_safety` | advisory by default; `min_free_margin_percent` set → CRITICAL margin-level check; missing evidence rejects |

## 4. Independent risk computation (§ never trust the proposal)

The gate **never reads `proposal.sizing.risk_amount`**. It recomputes from
first principles using the verified symbol spec:

```text
loss_per_lot = |entry − stop_loss| / tick_size × tick_value
actual_risk  = suggested_volume × loss_per_lot
```

A proposal that claims $40 risk while its own geometry implies $500 is
rejected with the true number. Non-finite inputs produce a FAIL, never a
NaN in the decision. Exposure uses the same independence:
`proposed_notional = volume × (entry / tick_size) × tick_value`.

## 5. Decision precedence

Exactly one outcome, evaluated in tiers; a failing tier **stops** the walk
and all remaining checks are recorded `NOT_EVALUATED` (visible, never hidden):

1. **EMERGENCY_STOP** — operator emergency stop
2. **EMERGENCY_STOP** — kill switch
3. **REJECTED** — trading globally disabled
4. **REJECTED** — account not allowed / inconsistent evidence
5. **REJECTED** — any safety-check failure (all failures listed together)
6. **APPROVED** — everything passed (advisory WARNs recorded as warnings)

## 6. Kill switch & emergency stop (`app/risk/kill_switch.py`)

Two **distinct** halts, both caller-supplied — the gate never invents
catastrophic conditions:

* `KillSwitchState` (source `config`) and `EmergencyStopState`
  (source `operator`) with statuses `STANDBY` / `ACTIVE` / `TRIGGERED` /
  `RESET_REQUIRED`. Engaged states require explicit reset — a restart
  cannot silently re-enable trading.
* `KillSwitchStore` Protocol (+ in-memory implementation) is the
  persistence interface for the future GUI/VPS control plane; the gate
  itself only consumes the two boolean flags via `halt_flags()`.
* `EmergencyStopState`/`KillSwitchState` statuses feed `AccountState`
  flags through the documented `halt_flags()` bridge.

## 7. Risk events (`app/risk/events.py`)

`RiskEvent` data objects — `RISK_APPROVED`, `RISK_REJECTED`,
`EMERGENCY_STOP`, `KILL_SWITCH_ACTIVE` — emitted through an optional
`event_sink` callback (exactly one per evaluation). No Telegram, no HTTP,
no network: receiving layers (log, bus, storage) stamp time and transport.

## 8. Configuration (`RiskGateConfig`)

Safe defaults: `trading_enabled=False`, `dry_run=True`,
`max_risk_per_trade_pct=0.5`, `daily_loss_limit_pct=2.0`,
`max_consecutive_losses=3`, `max_open_positions=1`,
`max_pending_orders=2`, `max_spread_points=50.0`.
`max_total_exposure` has **no default** — it must be set explicitly
(fail-closed); `from_app_config()` deliberately does not invent one.
`min_free_margin_percent=None` means the margin check is advisory-only.

## 9. Determinism

* No `uuid`, `random`, `datetime.now`, network or MT5 anywhere in
  `app/risk` (AST-enforced by `tests/unit/risk/test_gate_hygiene.py`).
* `gate_decision_id` = deterministic hash of (proposal fingerprint, action,
  failed checks) — identical inputs ⇒ identical decision, repeated or
  rebuilt.
* The same inputs evaluated twice produce byte-identical decisions
  (`model_dump()` equality, tested).

## 10. Limitations & Phase-5 boundary

* The gate judges a proposal against **supplied** evidence; it cannot
  fetch account state, positions or prices itself (by design — all inputs
  arrive via `RiskState` / `AccountState`).
* `RiskState.daily_loss`, `consecutive_losses`, position counts are
  caller-maintained; Phase 5/6 must keep them honest.
* Exposure is notional-based (volume × entry value), a conservative
  approximation — no leverage/margin-usage modeling until Phase 5.
* Margin safety is advisory unless explicitly enabled.
* **APPROVED is data, not action**: nothing in Phase 4 sends orders,
  modifies positions or creates pending orders. Execution is Phase 5
  (validate → risk check → order_check → send → verify → reconcile).
