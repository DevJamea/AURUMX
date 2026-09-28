"""Execution layer (Phase 5) — the ONLY caller of broker execution methods.

Modules:

* ``contracts``   — ``ExecutionRequest`` (derived verbatim from an approved
                    proposal, deterministic ids) and ``ExecutionResult``
                    (status/stage taxonomy where UNKNOWN is never success);
* ``service``     — ``ExecutionService``: risk-decision verification ->
                    mode gates -> independent local validation -> DRY_RUN
                    simulation or DEMO-guarded MT5 execution -> state
                    verification.  No retries, no proposal mutation;
* ``journal``     — ``ExecutionRecord`` + storage contract (full
                    correlation chain, honest statuses);
* ``events``      — execution/reconciliation/control event codes published
                    on the existing in-process ``EventBus``;
* ``reconciliation`` — journal vs MT5 state comparison (MATCHED /
                    MISSING_IN_MT5 / MISSING_IN_JOURNAL / MISMATCH /
                    UNKNOWN) plus the safety guard that halts execution on
                    unresolved mismatches.

The MT5 ``order_check``/``order_send`` calls themselves live inside the
broker adapter (``app/brokers/mt5.py`` — the only MetaTrader5 importer);
this package reaches the broker exclusively through ``BrokerInterface``.
"""

from app.execution.contracts import (
    AURUMX_MAGIC,
    DEFAULT_DEVIATION_POINTS,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStage,
    ExecutionStatus,
    derive_request_id,
)
from app.execution.journal import (
    ExecutionJournal,
    ExecutionRecord,
    InMemoryExecutionJournal,
)
from app.execution.reconciliation import (
    ExecutionComparison,
    Reconciler,
    ReconciliationGuard,
    ReconciliationReport,
    ReconciliationStatus,
)
from app.execution.service import ExecutionService, ExecutionServiceConfig

__all__ = [
    "AURUMX_MAGIC",
    "DEFAULT_DEVIATION_POINTS",
    "ExecutionComparison",
    "ExecutionJournal",
    "ExecutionRecord",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionService",
    "ExecutionServiceConfig",
    "ExecutionStage",
    "ExecutionStatus",
    "InMemoryExecutionJournal",
    "ReconciliationGuard",
    "ReconciliationReport",
    "ReconciliationStatus",
    "Reconciler",
    "derive_request_id",
]
