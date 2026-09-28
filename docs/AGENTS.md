# AurumX Agent Layer (Phase 2)

Seven deterministic analysis agents plus a macro interface, a regime detector
and a synthesis contract.  This document is the reference for what each agent
does, what it consumes, how its score is built, where it is limited, and how
the regime engine weights its output.

**Everything here is enforced by tests** (`tests/unit/agents/`,
`tests/unit/decision/`, `tests/integration/test_agent_pipeline.py`).

---

## 1. Common contract

Every agent implements `BaseAgent.analyze(context) -> AgentResult`:

| Field                | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `agent`              | stable identifier (`trend`, `momentum`, …)                              |
| `direction`          | `BUY` / `SELL` / `NEUTRAL`                                              |
| `signal_strength`    | deterministic evidence score in **[0, 1]** — *not* a probability; never calibrated, never called one |
| `primary_timeframe`  | the timeframe whose last **closed** candle is the decision candle        |
| `timeframes_used`    | every timeframe the agent actually read                                 |
| `reasons`            | human-readable provenance: *why* this signal                            |
| `features`           | JSON-safe measurements (journal-ready)                                  |
| `warnings`           | data anomalies, conflicts, suppressed-evidence notes                    |
| `data_quality`       | `OK` / `DEGRADED` / `INSUFFICIENT` / `INVALID` / `NO_DATA`              |
| `source_time`        | timestamp of the decision candle (last closed candle of primary tf)     |
| `snapshot_time`      | snapshot the analysis was derived from (deterministic provenance)       |

Hard rules (statically enforced by `test_contract.py::TestSourceHygiene`):

* no MetaTrader5, no network, no wall-clock, no randomness, no file I/O;
* same context in → byte-identical result out (determinism tests);
* the context/snapshot is never mutated (purity tests);
* missing/invalid data ⇒ NEUTRAL + explicit quality status, never an exception;
* an **abnormal decision candle** (body > 20× trailing ATR or > 10 % of price —
  a bad tick) suppresses directional signals entirely (`abnormal_candle_result`).

### Scoring methodology (no fake confidence)

Scores are weighted sums of *documented boolean evidence*.  `+` evidence,
`−` contradicting evidence, regime compatibility is applied later by the
synthesis layer (weights HIGH 1.25 / NORMAL 1.0 / REDUCED 0.5 / DISABLED 0).
A score of 0.85 means "85 % of the documented evidence table fired", nothing
more.  Thresholds are chosen so that single-indicator signals are impossible.

### Multi-timeframe roles (explicit, tested)

| Timeframe | Role      | Used for                                   |
|-----------|-----------|--------------------------------------------|
| H4 (D1)   | MACRO     | primary-trend context, structure agreement |
| H1 (M30)  | STRUCTURE | directional decisions, statistics          |
| M15 (M5/M1)| ENTRY    | entry timing, reversal confirmation        |

The mapping lives in `app/agents/context.py` (`DEFAULT_ROLES`); agents declare
`primary_timeframe` and may only *read* other timeframes through the context.

---

## 2. Shared feature layer (`app/agents/features.py`)

All indicators are computed once per timeframe in `build_market_context`
(instance-scoped, no global caches):

* **EMA 20/50/200** (pandas `ewm`, ATR-buffered comparisons: a gap smaller
  than 0.05 ATR is "flat", not aligned);
* **RSI(14)**, **MACD(12,26,9)**, **ROC(10)** — Wilder smoothing where standard;
* **ATR(14)**, ATR %, ATR ratio vs 100-bar median, Bollinger(20, 2, ddof=0),
  %B (zero-width bands ⇒ 0.5 by convention), BB width percentile (≥ 20 values);
* **ADX(14)** with Wilder-smoothed ±DM (a flat market reads ADX < 20);
* **Fractal swings** (left = right = 2, strict max/min): a swing at bar *i* is
  only knowable from bar *i + 2* — the **confirmation delay of 2 closed
  candles** is part of the contract.  A prominence filter (≥ 0.75 ATR over the
  opposite window extreme) removes wick-jitter "swings";
* **Structure events**: BOS (break in bias direction) / CHOCH (break against
  it), only from *closed* candles breaking *already-confirmed* levels —
  nothing retroactive.  Look-ahead safety is proven by prefix-consistency
  tests: extending history never changes a previously derived event;
* **Bad-tick guard**: `last_candle_anomaly` (see §1).

---

## 3. Agents

### 3.1 TrendAgent — `app/agents/trend.py`

* **Purpose**: classify the H1 trend and its strength; H4 provides macro
  agreement.
* **Inputs**: H1 (primary, ≥ 60 candles), H4 (optional).
* **Evidence table** (bullish; bearish mirrored):

| E  | Evidence (H1)                          | Weight |
|----|----------------------------------------|--------|
| E1 | EMA20 > EMA50 by > 0.05 ATR            | 0.25   |
| E2 | EMA50 > EMA200 by > 0.05 ATR           | 0.15   |
| E3 | ADX ≥ 25 (1.0) / ≥ 20 (0.6)            | 0.20   |
| E4 | 20-bar regression slope > 0.10 ATR/bar | 0.15   |
| E5 | H4 EMA alignment agrees                | 0.10   |
| E6 | H1 structure bias UPTREND (HH/HL)      | 0.15   |

* **Actionable** ⇔ E1 ∧ ≥ 2 of {E2…E6} — *EMA20 > EMA50 alone never signals*
  (explicit test).  States: `STRONG_BULLISH` (E1∧E2∧ADX≥25∧E4),
  `WEAK_BULLISH`, mirrored bearish, else `NEUTRAL`.
* **Limitations**: < 200 candles ⇒ E2 excluded, quality DEGRADED (warning);
  EMA-based trends lag reversals by design.
* **Regime compatibility**: HIGH in TREND_*, REDUCED in RANGE/LOW_VOL/UNCERTAIN.
* **Example** (linear uptrend, 300 bars): `BUY 0.85` — reasons cite every
  evidence item; `trend_state=STRONG_BULLISH`, `ema_alignment_h1=up`.

### 3.2 MomentumAgent — `app/agents/momentum.py`

* **Purpose**: regime-aware momentum; **no naive RSI reversals**.
* **Inputs**: H1 (primary, ≥ 60), M15 (confirmation).
* **TREND mode** (regime TREND_* or local ADX ≥ 20):

| M  | Evidence                                    | Weight |
|----|---------------------------------------------|--------|
| M1 | MACD **line** > 0 — direction               | 0.25   |
| M1b| MACD histogram > 0 — strengthening          | 0.15   |
| M2 | histogram rising vs previous bar            | 0.15   |
| M3 | RSI > 55 / < 45                             | 0.20   |
| M4 | ROC(10) beyond ±0.1 % (same sign)           | 0.15   |
| M5 | M15 MACD line agrees                        | 0.10   |

  BUY ⇔ M1 ∧ ≥ 2 corroborators.  The *line* (not the histogram) is the core
  evidence: the histogram measures acceleration and idles near zero in a
  steady trend.
* **RANGE mode** (regime RANGE/LOW_VOL or local ADX < 20): RSI < 30 is only a
  buy when the last M15 candle is a bullish reversal candle (and mirrored for
  RSI > 70).  In a TREND regime the same RSI 30 is pullback continuation —
  tested to *not* produce a BUY.
* **Guards**: exhaustion (RSI ≥ 75 ∧ meaningful bearish histogram
  (>|line|·0.1) ∧ fading ROC) ⇒ NEUTRAL + warning — do not chase a decelerating
  move.  RSI/MACD conflict ⇒ strength × 0.6 + warning.
* **Limitations**: momentum is a lagging descriptor; conflict and exhaustion
  are heuristics, deliberately conservative.
* **Regime compatibility**: HIGH in TREND_*, NORMAL in RANGE (mode switches).
* **Examples**: steady uptrend → `BUY 0.42`; steady downtrend → `SELL 1.0`;
  decelerating rally → `NEUTRAL` + "bullish momentum exhausted".

### 3.3 StructureAgent — `app/agents/structure.py` (major)

* **Purpose**: deterministic market structure — swings, BOS, CHOCH.
* **Inputs**: H1 (primary, ≥ 30), H4 (agreement).
* **Rules**: only *confirmed* swings (delay 2 candles, documented in every
  result via `confirmation_delay_candles`); BOS = close breaks the latest
  confirmed swing level in bias direction (base 0.45); CHOCH = break against
  the bias (base 0.35 — early reversal, reduced by design); recency ≤ 20
  candles (+0.20 scaled), H4 agreement (+0.20), clean swing sequence (+0.15).
  BOS-vs-bias mismatch ⇒ NEUTRAL ("transitioning").  Output carries the
  **invalidation level** (last HL/LH) for future stop logic.
* **Limitations**: 2-candle confirmation delay means every signal is 2 bars
  later than the extreme; fractal structure needs visible swings (a smooth
  ramp has none → NEUTRAL, by design).
* **Regime compatibility**: HIGH in TREND_*, NORMAL elsewhere.
* **Look-ahead safety**: proven at the feature layer (prefix consistency) and
  at the agent level (events identical to the deterministic event list; no
  event may precede its level's confirmation bar).
* **Examples**: uptrend zigzag → `BUY 1.0` (BOS_UP + bias + H4 + sequence);
  downtrend reversal → `BUY 0.55` (CHOCH_UP, flagged); range → `NEUTRAL`.

### 3.4 LiquidityAgent — `app/agents/liquidity.py`

* **Purpose**: price-action around reference levels.  *Not* an order-book
  agent — no order-book claims, tick volume only.
* **Inputs**: M15 (primary, ≥ 45), H1 swings + previous UTC-day high/low +
  M15 20-bar extremes (a sweep candle never becomes its own level: levels
  exclude the last candle; failed-breakout pre-levels exclude the last two).
* **Patterns** (hand-crafted-candle tests for each):

| Pattern                     | Score          |
|-----------------------------|----------------|
| Sweep (pierce + close back) | ± 0.35         |
| Wick rejection (≥ 60 % wick, close placement) | ± 0.30 |
| Breakout, volume-confirmed (ratio ≥ 1.5) | ± 0.30 |
| Breakout, unconfirmed      | ± 0.20         |
| Failed breakout (2-candle)  | ∓ 0.25         |

* **Action threshold**: |net| ≥ 0.30 — an unconfirmed breakout (0.20) or a lone
  failed breakout (0.25) is *recorded as evidence* but does not signal.
* **Missing volume**: price evidence kept, volume confirmation skipped,
  warning, never DEGRADED-below-usable.
* **Limitations**: pattern detection is scale-free (ATR-relative) — a dead
  market's wick noise can produce proportionally weak micro-signals (≤ 0.35);
  synthetic timeframes have no real order-flow semantics.
* **Regime compatibility**: NORMAL in all regimes (it feeds context, not
  direction).
* **Examples**: bullish sweep of the 20-bar low → `BUY 0.35`; breakout with
  volume ratio 3 → `BUY 0.30`; failed breakout + rejection wick → `SELL 0.60`.

### 3.5 VolatilityAgent — `app/agents/volatility.py`

* **Purpose**: volatility context — LOW / NORMAL / HIGH / EXTREME and
  STABLE / EXPANDING / CONTRACTING.  **Never directional**: high volatility
  alone produces a warning ("risk controls should widen stops / reduce size"),
  not a signal (tested).
* **Inputs**: H1 (primary, ≥ 30 candles; full ranking confidence from ~80).
* **Classification** (either axis): EXTREME ≥ 2.5× ATR-median or BB-width
  percentile ≥ 0.97; HIGH ≥ 1.5× or ≥ 0.85; LOW ≤ 0.6× or ≤ 0.10; else NORMAL.
* **Limitations**: percentile ranking needs history — short series self-degrade
  to DEGRADED + "insufficient history" warning rather than guessing.
* **Regime compatibility**: HIGH in HIGH_VOL/LOW_VOL (it defines the axis),
  NORMAL elsewhere.

### 3.6 MeanReversionAgent — `app/agents/mean_reversion.py`

* **Purpose**: heavily constrained range mean-reversion.  This is the
  deliberate redesign of SnipBot's DCA/grid ideas: **no martingale, no
  averaging down, no pyramiding, no DCA, no loss-dependent sizing** — the
  agent has no notion of positions at all (flags `no_martingale`,
  `no_averaging` in every result).
* **Inputs**: H1 (primary, ≥ 60) + M15 (confirmation).
* **Gates, in order** (failing any ⇒ NEUTRAL with the reason):
  1. regime gate: only RANGE / LOW_VOLATILITY (or, without a regime, local
     ADX < 20 ∧ ATR-normalized EMA gap < 0.5);
  2. statistical edge: %B ≤ 0.05 or deviation ≤ −1.5 ATR (mirrored);
  3. range boundary: within 0.75 ATR of the 50-bar low/high;
  4. M15 reversal confirmation (bullish candle with ≥ 33 % lower wick, or H1
     close back inside the band).  Edge *without* confirmation ⇒ NEUTRAL +
     "no falling-knife catches" warning.
* **Strength** = 0.45·edge-depth + 0.30·boundary-proximity +
  0.25·confirmation-quality (all components in `features`).
* **Limitations**: strictly a range tool — TREND/UNCERTAIN/HIGH_VOL exclude it
  by construction (tested with textbook oversold data inside a trend).
* **Regime compatibility**: HIGH in RANGE, NORMAL in LOW_VOL, DISABLED
  elsewhere.
* **Example**: sine range + close 4 ATR below the mean + M15 hammer →
  `BUY 0.72` with `percent_b=-0.33, deviation_atr=-4.46`.

### 3.7 MacroAgent — `app/agents/macro.py` (interface only)

* **Purpose**: provenance-carrying macro/news context.  Phase 2 ships the
  **interface** (`MacroDataProvider` protocol, `MacroEvent`); future
  implementations (economic calendar, CPI/NFP/FOMC feeds) plug in without
  touching the pipeline.
* **Behaviour**: no provider ⇒ `NEUTRAL` + `NO_DATA` + explicit provenance
  ("provider=none (no macro information fabricated)") — never fabricates.
  High-impact USD events within ±30 min produce a **blackout warning** for the
  future risk gate — the agent is deliberately non-directional.  Provider
  failures fail safe (NEUTRAL, DEGRADED, warning).
* **Limitations**: no calendar source in Phase 2 by design (no paid APIs, no
  LLM, no internet — spec §47).
* **Regime compatibility**: NORMAL in every regime.

---

## 4. Regime engine (`app/decision/regime.py`)

`RegimeDetector.detect(context)` → `RegimeAssessment` (regime, volatility
axis, evidence strings, conflicts).  Needs ≥ 60 H1 candles, else UNCERTAIN
(INSUFFICIENT).

* **TREND_UP / TREND_DOWN**: H1 EMA alignment ∧ ADX ≥ 25 ∧ (H4 agrees or
  structure confirms).  H4 *opposing* ⇒ UNCERTAIN with an explicit conflict —
  never a confident TREND call.
* **RANGE**: ADX ≤ 20 ∧ flat/mixed EMAs ∧ structure bias RANGE/UNKNOWN.
* **HIGH_VOLATILITY / LOW_VOLATILITY**: range-like structure with the
  volatility axis pinned to EXTREME/HIGH or LOW.
* **UNCERTAIN**: everything else — a first-class, safe answer.

**Relevance model** (not equal averaging): TREND regimes give trend/momentum/
structure HIGH (×1.25) and **disable** mean-reversion (×0); RANGE promotes
mean-reversion to HIGH and reduces trend agents (×0.5); volatility agents
define the volatility axis.  The matrix is exhaustive per regime and tested.

## 5. Synthesis (`app/decision/synthesis.py`)

`synthesize(SynthesisInput) → SynthesisOutput`: deterministic reference
aggregation — `buy_score`/`sell_score` are relevance-weighted sums of agent
strengths; `action` = BUY when net ≥ +0.80 ∧ buy ≥ 1.00 (SELL mirrored, else
HOLD — placeholder thresholds, Phase 3 makes them configurable).  **Disagreement
is preserved**: every stance is carried with its direction, relevance and
weighted strength; supporting/opposing/neutral/disabled lists survive even on
HOLD; conflicts are explicit strings.  No trade is executed — outputs are
journal-serializable evidence for Phase 3's decision engine.

## 6. Test catalogue

| Area | File | What is pinned |
|------|------|----------------|
| Feature layer | `test_features.py` | indicator math, swing confirmation delay, prominence, bias, BOS/CHOCH, **prefix look-ahead safety** |
| Trend | `test_trend.py` | bull/bear/sideways/insufficient, E1-alone rule, EMA200 warm-up |
| Momentum | `test_momentum.py` | trend/range modes, no naive RSI buys, exhaustion, conflict, M15 degradation |
| Structure | `test_structure.py` | BOS/CHOCH strengths, invalidation, stale events, confirmed-levels-only |
| Liquidity | `test_liquidity.py` | sweeps, breakouts (volume variants), failed breakouts, wick rejection, ordinary |
| Volatility | `test_volatility.py` | LOW/NORMAL/HIGH/EXTREME, warnings, short-history degradation |
| Mean reversion | `test_mean_reversion.py` | valid setups, all gates, no-martingale flags, falling-knife rule |
| Macro | `test_macro.py` | no-provider provenance, blackout, fail-safe provider |
| Registry | `test_registry.py` | order, enable/disable, failure isolation, events |
| Contract | `test_contract.py` | contract shape, determinism, purity, wall-clock independence, roles, **static source hygiene** |
| Adversarial | `test_adversarial.py` | NaN/inf rejection, dup/unordered/gapped/short series, spikes, constant prices, zero volume, bad-tick guard |
| Regime | `test_regime.py` | all six regimes, relevance matrix, weights |
| Synthesis | `test_synthesis.py` | BUY/SELL/HOLD, disagreement preservation, disabled agents, determinism |
| Integration | `test_agent_pipeline.py` | FakeMT5 → service → context → regime → agents → synthesis, end-to-end |

## 7. Known limitations (Phase 2 scope)

1. Synthesis thresholds (0.80 / 1.00) are placeholders pending Phase 3.
2. MacroAgent has no data source by design; blackout logic is tested with
   stub providers only.
3. Liquidity patterns are ATR-relative and scale-free; micro-structure in dead
   markets yields proportionally weak (≤ 0.35) readings rather than silence.
4. Regime detection requires 60 H1 candles; younger contexts are UNCERTAIN.
5. Swing confirmation delay (2 candles) applies to every structure-derived
   signal — documented, not hidden.
6. The bad-tick guard (20×ATR / 10 % body) suppresses signals on physically
   implausible candles; a genuine once-a-decade shock would also be suppressed
   (fail-safe bias, by design).
