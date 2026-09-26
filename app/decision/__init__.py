"""Decision engine (Phase 2: regime + synthesis contract; Phase 3: full engine).

Implemented so far:

* ``regime``    — deterministic regime detection (TREND_UP/DOWN, RANGE,
  HIGH/LOW_VOLATILITY, UNCERTAIN) + the per-agent relevance matrix that
  decides which agents may influence decisions under each regime.
* ``synthesis`` — the input/output contract aggregating agent evidence while
  **preserving disagreement** (supporting/opposing stances, conflicts).  The
  reference aggregation is deterministic; Phase 3 replaces its internals with
  the full weighted decision engine (configurable weights, entry/SL/TP
  proposal, journal persistence).
"""

from app.decision.regime import (
    AGENT_RELEVANCE_WEIGHTS,
    REGIME_RELEVANCE,
    RegimeAssessment,
    RegimeDetector,
    relevance_for,
    relevance_of,
    weighted_strength,
)
from app.decision.synthesis import (
    ACTION_NET_THRESHOLD,
    AgentStance,
    SynthesisInput,
    SynthesisOutput,
    synthesize,
)

__all__ = [
    "ACTION_NET_THRESHOLD",
    "AGENT_RELEVANCE_WEIGHTS",
    "AgentStance",
    "REGIME_RELEVANCE",
    "RegimeAssessment",
    "RegimeDetector",
    "SynthesisInput",
    "SynthesisOutput",
    "relevance_for",
    "relevance_of",
    "synthesize",
    "weighted_strength",
]
