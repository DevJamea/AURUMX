"""Opt-in gate for the real-MT5 suite (spec §36).

These tests NEVER run during normal ``pytest`` or CI: they are collected
only when the environment variable ``AURUMX_LIVE_MT5=1`` is set.  They
require a Windows machine with a running MetaTrader 5 terminal logged in
to a DEMO account.
"""

from __future__ import annotations

import os

#: set AURUMX_LIVE_MT5=1 to collect the real-terminal suite
LIVE_MT5_ENV = "AURUMX_LIVE_MT5"

if os.environ.get(LIVE_MT5_ENV) != "1":
    collect_ignore_glob = ["test_*.py"]
