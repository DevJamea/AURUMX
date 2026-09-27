"""Hard risk gate (Phase 4) and its Phase-3 foundations.

Planned Phase-4 modules: the RiskGate IMPLEMENTATION (the contract already
exists — see ``app/risk/gate.py``, interface only), ``limits``,
``daily_loss``, ``kill_switch`` (emergency stop, persisted across
restarts).

Phase 3 ships the pure, execution-free parts already:

* ``sizing`` — deterministic risk-based position sizing (no martingale, no
  loss-recovery, no DCA: past losses are not even an input);
* ``state``  — the RiskState snapshot the decision engine consumes, plus
  the AccountState input model of the Phase-4 gate;
* ``gate``   — the Phase-4 RiskGate CONTRACT (Protocol + RiskDecision +
  REQUIRED_CHECKS). Deliberately NOT exported here yet and deliberately
  not implemented: nothing in Phase 3 can call it (boundary tests), and
  Phase 4 wires the exports when it implements the barrier.
"""

from app.risk.sizing import RiskSizing, calculate_position_size
from app.risk.state import RiskState

__all__ = ["RiskSizing", "RiskState", "calculate_position_size"]

