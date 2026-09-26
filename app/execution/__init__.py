"""Execution service (Phase 5).

Planned modules: ``service`` (validate -> risk check -> order_check -> send ->
verify -> reconcile), ``orders``, ``position_manager``, ``break_even``,
``trailing``, ``partial_close``.  The ONLY layer allowed to call broker
execution methods.
"""

