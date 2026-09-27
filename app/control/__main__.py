"""Windows control-plane entry point: ``python -m app.control``.

Starts the local control API (GUI + JSON) wired to a real MT5 broker.
``--help`` works everywhere (no MT5 import happens until the broker is
actually constructed); on non-Windows machines without the MetaTrader5
package the process exits with a clear, fail-closed message.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.control",
        description=(
            "AurumX local control plane (Phase 5): MT5 + engine + local "
            "control API + GUI.  Safe defaults: READ_ONLY until configured "
            "otherwise; demo execution requires explicit configuration and "
            "a verified DEMO account."
        ),
    )
    parser.add_argument(
        "--host", default=None,
        help="control API bind address (default: CONTROL_API_HOST / 127.0.0.1)",
    )
    parser.add_argument(
        "--port", default=None, type=int,
        help="control API port (default: CONTROL_API_PORT / 8757)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="force DRY_RUN regardless of configuration (safety ratchet)",
    )
    args = parser.parse_args(argv)

    from app.core.config import AppConfig
    from app.core.exceptions import AurumXError

    config = AppConfig.load()
    if args.dry_run:
        config.dry_run = True

    from app.brokers.mt5 import MT5Broker
    from app.control.api import LocalControlAPI
    from app.control.runtime import EngineRuntime

    broker = MT5Broker.from_config(config)
    runtime = EngineRuntime(config, broker)
    api = LocalControlAPI(
        runtime,
        host=args.host or config.control_api_host,
        port=args.port or config.control_api_port,
    )

    try:
        runtime.connect()
    except AurumXError as exc:
        print(
            f"AURUMX: could not connect to MetaTrader 5 — {exc}\n"
            "The control plane requires a running MT5 terminal (Windows). "
            "Offline: run the test suite (pytest) or diagnostics "
            "(python scripts/diagnostics.py).",
            file=sys.stderr,
        )
        return 2

    api.start()
    print(f"AURUMX CONTROL running at {api.url}")
    print(f"mode={runtime.status()['mode']}  (Ctrl+C to stop)")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nAURUMX: shutting down.")
    finally:
        api.stop()
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
