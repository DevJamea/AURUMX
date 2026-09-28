"""AurumX read-only diagnostics.

Backs the launcher's "Run Diagnostics" action (spec §43) and doubles as the
Phase-1 acceptance check on a real Windows machine.  It is strictly read-only:
it connects, discovers the gold symbol and validates market data — it never
places orders.

Usage::

    python scripts/diagnostics.py            # uses .env / environment
    python scripts/diagnostics.py --json     # machine-readable output

Exit code: 0 = all checks PASS (warnings allowed), 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.brokers.interface import BrokerInterface  # noqa: E402
from app.brokers.mt5 import MT5Broker  # noqa: E402
from app.core.config import AppConfig  # noqa: E402
from app.core.enums import AccountTradeMode  # noqa: E402
from app.core.exceptions import AurumXError  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402

log = get_logger("scripts.diagnostics")

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class DiagnosticsReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == WARN for c in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))

    def as_dict(self) -> dict:
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": (
                FAIL if self.has_failures else (WARN if self.has_warnings else PASS)
            ),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }


def run_diagnostics(
    *,
    config: AppConfig | None = None,
    broker: BrokerInterface | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DiagnosticsReport:
    """Run all read-only checks.  ``broker`` injects a fake for tests."""
    config = config or AppConfig.load()
    configure_logging(level=config.log_level, log_format=config.log_format)
    report = DiagnosticsReport()

    # 1. Python
    ok = sys.version_info >= (3, 11)
    report.add(
        "python",
        PASS if ok else FAIL,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    # 2. Configuration safety
    report.add(
        "configuration",
        PASS if not config.trading_enabled else WARN,
        f"mode={config.trading_mode.value} symbol={config.symbol}",
    )

    # 3. MT5 availability / connection
    broker = broker or MT5Broker.from_config(config)
    try:
        broker.connect()
    except AurumXError as exc:
        report.add("mt5_connection", FAIL, str(exc)[:300])
        _finish(report)
        return report
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        report.add("mt5_connection", FAIL, f"{type(exc).__name__}: {exc}")
        _finish(report)
        return report
    report.add("mt5_connection", PASS, f"broker={broker.name}")

    try:
        # 4. Account
        try:
            account = broker.get_account()
            if account.trade_mode is AccountTradeMode.REAL:
                report.add(
                    "account",
                    WARN,
                    f"login={account.login} REAL account (demo recommended) "
                    f"server={account.server}",
                )
            else:
                report.add(
                    "account",
                    PASS,
                    f"login={account.login} {account.trade_mode.value} "
                    f"equity={account.equity} {account.currency}",
                )
        except AurumXError as exc:
            report.add("account", FAIL, str(exc)[:300])

        # 5-7. Symbol discovery + market data via the service
        service = MarketDataService.from_config(
            config, broker, clock=clock or (lambda: datetime.now(UTC))
        )
        try:
            spec = service.resolve_symbol()
            report.add(
                "gold_symbol",
                PASS,
                f"{spec.name} digits={spec.digits} point={spec.point} "
                f"contract={spec.contract_size} vol_step={spec.volume_step}",
            )
        except AurumXError as exc:
            report.add("gold_symbol", FAIL, str(exc)[:300])
            _finish(report)
            return report

        try:
            snapshot = service.get_snapshot()
            if snapshot.trading_data_ok:
                spread = (
                    f"{snapshot.spread_points:.0f} pts" if snapshot.spread_points is not None else "n/a"
                )
                report.add(
                    "market_data",
                    PASS,
                    f"bid={snapshot.bid} ask={snapshot.ask} spread={spread} "
                    f"session={snapshot.session_state.value}",
                )
            else:
                issues = snapshot.report.summary()[:300]
                status = WARN if snapshot.session_state.value == "CLOSED" else FAIL
                report.add(
                    "market_data",
                    status,
                    f"data not trading-grade (session={snapshot.session_state.value}): {issues}",
                )
            for tf in config.timeframes:
                check = snapshot.series.get(tf)
                if check is None:
                    report.add(f"candles_{tf.value}", FAIL, "missing")
                elif check.series is not None:
                    # A closed session explains staleness — warn, don't fail.
                    closed = snapshot.session_state is not None and snapshot.session_state.value == "CLOSED"
                    status = PASS if check.valid else (WARN if closed else FAIL)
                    report.add(
                        f"candles_{tf.value}",
                        status,
                        f"{len(check.series.candles)} closed candles, "
                        f"age={check.age_seconds and round(check.age_seconds, 1)}s",
                    )
        except AurumXError as exc:
            report.add("market_data", FAIL, str(exc)[:300])

        # 8. Data directory writable
        try:
            data_dir = Path(config.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            probe = data_dir / ".diagnostics_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.add("data_directory", PASS, str(data_dir.resolve()))
        except OSError as exc:
            report.add("data_directory", FAIL, str(exc))
    finally:
        try:
            broker.disconnect()
        except Exception:  # noqa: BLE001
            pass

    _finish(report)
    return report


def _finish(report: DiagnosticsReport) -> None:
    for check in report.checks:
        log.info(
            f"diagnostics: {check.name} -> {check.status}",
            event="DIAGNOSTICS_CHECK",
            check=check.name,
            status=check.status,
            detail=check.detail,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="AurumX read-only diagnostics")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--env-file", default=".env", help="path to .env file")
    args = parser.parse_args()

    report = run_diagnostics(config=AppConfig.load(env_file=args.env_file))

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print("\nAurumX Diagnostics")
        print("=" * 60)
        for check in report.checks:
            marker = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}[check.status]
            print(f"{marker:<8} {check.name:<18} {check.detail}")
        print("=" * 60)
        print(f"Summary: {report.as_dict()['summary']}\n")

    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
