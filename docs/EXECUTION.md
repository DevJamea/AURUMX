# AurumX Execution Layer (Phase 5)

Status: **implemented (offline-verified)** — `app/execution`, `app/control`,
`app/brokers/mt5.py` (execution half).  **Phase 5 does not authorize
real-money trading.**  Real accounts are refused unconditionally; the only
live path is a verified DEMO account, one explicitly-approved order at a
time.

```text
TradeProposal -> HardRiskGate -> RiskDecision(APPROVED)
                                     |
                              ExecutionRequest
                                     |
                            ExecutionService            <- the ONLY caller of
                                     |                      broker execution
                              MT5Broker.place_market_order
                               (order_check -> order_send)
                                     |
                          retcode verification + state check
                                     |
                              ExecutionResult
                                     |
                            Reconciliation  ->  Journal + events
```

## 1. Where `order_send` lives (and why nothing else can call it)

``order_check`` and ``order_send`` are called in exactly ONE place in the
codebase: ``MT5Broker.place_market_order`` in **``app/brokers/mt5.py``**
— the only module that imports ``MetaTrader5`` (lazily, Windows-only).
Everything above the broker layer goes through ``BrokerInterface``.

Enforced by tests (``tests/execution/test_phase5_hygiene.py``):

| Guarantee | Proof |
|---|---|
| `order_send`/`order_check` called only in the adapter | AST scan of every `app/**.py` |
| `place_market_order` called only by `ExecutionService` | AST scan |
| decision + risk layers have no execution surface | Phase-4 boundary suite (unchanged) + new scans |
| the GUI/control surface imports no broker module, calls no MT5 function | AST scans of `app/control/{api,state,errors,__init__}.py` |
| the runtime uses `BrokerInterface` only (no `MT5Broker` import) | AST scan |
| `MT5Broker(...)` constructed only in the adapter's `from_config` and the `app.control` entry point | scan |
| no execution-like HTTP endpoint exists | route-table assertions |
| dry-run short-circuit precedes the MT5 path, structurally | AST ordering proof on `execute()` |

## 2. ExecutionRequest / ExecutionResult

`ExecutionRequest` (``app/execution/contracts.py``) is derived **verbatim**
from an approved proposal via ``ExecutionRequest.from_proposal`` — symbol,
direction, volume, entry, SL, TP are copied exactly; the request is never
"normalized into validity" (a 0.105 lot volume is rejected, not rounded).
It carries the correlation chain: ``proposal_id -> risk_decision_id ->
request_id`` (+ ``order_ticket -> deal_ticket -> position_ticket`` on the
result).  IDs are sha256-derived — deterministic, no uuid/random/clock.

`ExecutionResult.status` is the exact spec taxonomy:
``NOT_ATTEMPTED / DRY_RUN / CHECK_FAILED / SEND_FAILED / REJECTED_BY_BROKER
/ ACCEPTED / FILLED / PARTIALLY_FILLED / UNKNOWN``.
**Unknown state is never success**: an accepted-but-unverified order is
``UNKNOWN``, not ``FILLED``; ``FILLED`` requires the position to be found
in actual MT5 state with matching direction/volume/SL/TP.

## 3. Risk-gate bypass protection

`ExecutionService.execute(request, decision)` requires:

1. `decision.action is APPROVED` (REJECTED / EMERGENCY_STOP / kill switch /
   trading-disabled decisions all block with ``NOT_ATTEMPTED``);
2. `decision.gate_decision_id == request.risk_decision_id` (non-empty);
3. `decision.proposal_id == request.proposal_id`;
4. `decision.fingerprint == request.fingerprint`;
5. mode gates: `trading_enabled`, no unresolved reconciliation halt;
6. independent local validation (defense in depth): gold-only symbol
   (Phase-1 ``is_gold_symbol``), verified broker `SymbolSpec`, volume vs
   broker min/max/step, geometry (sides, tick grid, broker minimum stop
   distance) — **reject, never repair**.

The proposal object is byte-identical before and after execution (pinned
by tests).

## 3b. Idempotency and single-use approvals

* **Single-use approval.** `EngineRuntime.execute_approved()` takes the last
  approved proposal and risk decision and clears them *before* doing anything
  else. Every attempt — success, refusal, engine stopped, halt, expiry —
  consumes the approval; a second call fails with "no proposal to execute".
* **Expiry.** An approved proposal past `expires_at` is refused as
  `proposal_expired`, and the refusal is journaled (`NOT_ATTEMPTED`).
* **Idempotency by `request_id`.** `ExecutionService.execute()` first asks the
  journal for an earlier record of the same `request_id` that *reached the
  broker* (`FILLED`, `PARTIALLY_FILLED`, `ACCEPTED`, `UNKNOWN`, `SEND_FAILED`,
  `REJECTED_BY_BROKER`, `CHECK_FAILED`). If found, the new call is rejected as
  `duplicate_request` (fresh `NOT_ATTEMPTED` / `BLOCKED` result, no tickets or
  fill price copied) and the broker is never contacted. `NOT_ATTEMPTED` and
  `DRY_RUN` records do **not** count, so DRY_RUN -> DEMO for the same
  proposal, or "trading was disabled, now enabled", remain possible.
* **Durability.** `python -m app.control` uses `SQLiteExecutionJournal` and
  `SQLiteDecisionJournal` under `DATA_DIR` (`executions.db`, `decisions.db`),
  so the duplicate guard survives restarts. The execution journal is
  append-only: rejected duplicates are recorded too. The in-memory journal
  (tests, embedded use) keeps only its most recent 1000 records, so its
  duplicate window is bounded.
* **Serialization.** One re-entrant lock covers `evaluate_cycle` and
  `execute_approved`; control operations (stop, emergency stop, kill switch)
  deliberately do **not** take it, so they never wait behind a slow broker call.

## 4. DRY_RUN (same pipeline, zero orders)

With `DRY_RUN=true` the service runs the identical pipeline — request
construction, risk verification, local validation, journal entry — and
terminates with a **simulated** result: status `DRY_RUN`, fill price =
the proposal's entry, **no fabricated tickets** (`order_ticket`/
`deal_ticket`/`position_ticket` stay `None`).  Neither `order_check` nor
`order_send` is invoked.  Proven by tests that count the fake terminal's
calls (always 0) at unit, service and integration level.

## 5. Demo execution (explicitly guarded)

Real broker execution requires ALL of:

1. `TRADING_ENABLED=true` and `DRY_RUN=false` in configuration;
2. the config-level confirmation phrase
   `REAL_TRADING_CONFIRMED="I ACCEPT REAL TRADING RISK"`
   (`AppConfig` refuses to construct otherwise — even for demo);
3. the **actual broker account is DEMO** — verified at execution time via
   `get_account()`, never inferred from configuration.  REAL accounts are
   blocked with ``NOT_ATTEMPTED / real_account_blocked`` before any
   broker call; UNKNOWN account types fail closed too.

Then: `order_check` (a failed check prevents `order_send` — tested) →
`order_send` → explicit retcode verification → position verification →
`FILLED`/`PARTIALLY_FILLED`/`UNKNOWN`.

### Retcode handling

The adapter classifies retcodes using the installed package's constants
(with documented numeric fallbacks): `DONE` → accepted; `DONE_PARTIAL` →
partial; `REJECT/CANCEL/INVALID*/MARKET_CLOSED/NO_MONEY/REQUOTE/
PRICE_OFF/TRADE_DISABLED/...` → definite rejection; `TIMEOUT/ERROR/
CONNECTION/PLACED` and **any unknown code** → fail closed.  There are NO
automatic retries (§42): exactly one `order_check` + at most one
`order_send` per `execute()` call — a timeout never triggers a second
attempt that could double exposure.

## 6. Reconciliation

``Reconciler`` (``app/execution/reconciliation.py``) compares the
execution journal against actual MT5 state (positions carrying the
`AURUMX_MAGIC` = `0x41555258` "AURX" magic number):

* `MATCHED` — agreement within the documented tolerances;
* `MISSING_IN_MT5` — journal claims a verified fill, no position exists;
* `MISSING_IN_JOURNAL` — an AURUMX-magic position the journal cannot
  explain (**never auto-adopted**);
* `MISMATCH` — both sides exist but disagree beyond tolerance;
* `UNKNOWN` — evidence insufficient (e.g. unconfirmed sends, broker
  unreachable).

Documented tolerances (§24): direction/magic exact; volume within one
`volume_step`; SL/TP within one tick; entry within 50 ticks (slippage).
Nothing is silently repaired: a non-clean report engages the
`ReconciliationGuard`, which blocks further execution until the operator
re-reconciles (clean) or explicitly acknowledges (audited event).

Limitation: Phase 5 reconciles *current* state — a position closed
normally (SL/TP) reads `MISSING_IN_MT5` and requires the explicit
acknowledgement; deal-history reconciliation arrives with a later phase.

## 7. Events, journal, logging

Event codes on the existing `EventBus`: `EXECUTION_REQUESTED /
CHECKED / SENT / SIMULATED / FILLED / REJECTED / FAILED / UNKNOWN`,
`RECONCILIATION_MATCHED / MISMATCH`, and control ops (`CONTROL_STARTED /
STOPPED / DRY_RUN_FORCED / EMERGENCY_STOP / KILL_SWITCH_RESET /
RECONCILIATION_ACKNOWLEDGED`).

The execution journal (`ExecutionRecord`) records every attempt — blocked
and dry-run included — with the full correlation chain and honest
statuses.  Logs use the structured JSON logger with components
`execution`, `brokers.mt5`, `execution.reconciliation`, `control`; no
secrets are ever logged (credentials are `SecretStr` + logger redaction).

## 8. Windows control plane

```text
Windows
├── AURUMX engine      (app/control/runtime.py — EngineRuntime)
├── MT5 terminal       (read via MT5Broker only)
├── Local Control API  (app/control/api.py — stdlib HTTP, 127.0.0.1:8757)
└── GUI                (app/control/static/index.html — served by the API)
```

Start with ``python -m app.control [--host H] [--port P] [--dry-run]``
(help works everywhere; the MT5 import happens only at startup).  The GUI
shows connection, **account type (from the broker, never config)**,
symbol, mode, trading-enabled, kill switch, emergency stop, bid/ask/
spread, agents, decision, risk gate, execution status, open positions,
reconciliation and recent events.  Buttons: START, STOP, DRY RUN,
EMERGENCY STOP, RESET KILL SWITCH, RECONCILE.  **There is no manual
trading and no execute endpoint** — the API cannot send an order; the
only execution paths are the runtime's explicit `execute_approved()`
(gate-approved pair required) and the live demo harness below.

Safe defaults: `TRADING_ENABLED=false`, `DRY_RUN=true`; the control plane
can only **de-escalate** (force DRY_RUN) — returning to demo execution
requires a configuration change and restart.

## 9. Running the tests

```bash
pytest -q            # the whole suite — offline, no MT5 terminal needed
ruff check .
python -m compileall .
```

### Real MT5 demo tests (explicit opt-in — never in CI)

On the Windows MT5 host with a **DEMO** account logged in:

```bat
set AURUMX_LIVE_MT5=1
set TRADING_ENABLED=true
set DRY_RUN=false
set REAL_TRADING_CONFIRMED=I ACCEPT REAL TRADING RISK
set MAX_TOTAL_EXPOSURE_USD=1000000
pytest tests/live_mt5 -m live_mt5 -s
```

The harness (``tests/live_mt5/test_demo_execution.py``) performs exactly
ONE evaluation cycle and at most ONE order — only for a proposal the real
HardRiskGate APPROVED — then verifies the retcode, the resulting MT5
position, the journal record and a clean reconciliation.  No setup in the
market → SKIP (nothing is forced).  With
`AURUMX_LIVE_ALLOW_SYNTHETIC=1` a minimal proposal built from the live
tick may be used — it still passes through the REAL gate (nothing is
bypassed).

## 10. Limitations

* Position management (break-even, trailing, partial close, pending
  orders) remains stubbed — a later phase; Phase 5 executes market orders
  only.
* `RiskState` loss accumulators (`daily_loss`, `consecutive_losses`) are
  caller-supplied; the runtime supplies 0 until the trade-history worker
  (Phase 6) maintains them.
* Exposure is notional-based (no leverage modeling).
* The control API is localhost-only without authentication; remote
  exposure requires the authenticated Phase-7 API.
* Reconciliation covers current open state (see §6).

> **Phase 5 does not authorize real-money trading.**  It proves the
> execution pipeline is safe, observable and reconcilable on a demo
> account.  Real-money deployment is a separate future validation phase.
