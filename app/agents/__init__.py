"""Specialized analysis agents (Phase 2).

Planned modules: ``base`` (BaseAgent/AgentResult interface), ``trend``,
``momentum``, ``structure``, ``liquidity``, ``volatility``, ``mean_reversion``,
``macro``.

Invariants:
* Agents receive validated ``MarketSnapshot`` data and return an
  ``AgentResult`` — they NEVER see a broker and NEVER execute trades.
* Agents emit ``signal_strength`` (a normalized evidence score), not a fake
  probability (spec §9, §57).
* Weights are configuration, regime-adjusted by the decision engine (spec §58).
"""

