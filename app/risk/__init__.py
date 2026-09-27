"""Hard risk gate (Phase 4).

Planned Phase-4 modules: ``engine`` (PASS/BLOCK gate — risk is a gate, not
a vote), ``limits``, ``daily_loss``, ``kill_switch`` (emergency stop,
persisted across restarts).

Phase 3 ships the pure, execution-free parts already:

* ``sizing`` — deterministic risk-based position sizing (no martingale, no
  loss-recovery, no DCA: past losses are not even an input);
* ``state``  — the RiskState snapshot the decision engine consumes.
"""

from app.risk.sizing import RiskSizing, calculate_position_size
from app.risk.state import RiskState

__all__ = ["RiskSizing", "RiskState", "calculate_position_size"]

