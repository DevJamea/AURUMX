"""Application configuration (spec §49).

Principles:

* **Safe defaults**: a fresh install is read-only — ``TRADING_ENABLED=false``,
  ``DRY_RUN=true`` (spec §54, absolute).
* **Unsafe configurations cannot exist**: enabling live (non-dry-run) trading
  without the exact confirmation phrase raises ``UnsafeConfigurationError`` on
  construction *and* on attribute mutation (``validate_assignment=True``).
* Flat environment names match the spec exactly (``RISK_PER_TRADE``,
  ``MAX_OPEN_POSITIONS``, …); an ``.env`` file is supported.
* Secrets (``MT5_PASSWORD``) are ``SecretStr`` — they never appear in logs,
  ``repr`` or ``safe_summary()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.enums import TimeFrame, TradingMode
from app.core.exceptions import UnsafeConfigurationError
from app.core.logging import get_logger

log = get_logger("core.config")

#: Exact phrase required to unlock real-money execution (Phase 5 still re-checks
#: the account at runtime; this is the configuration-side confirmation).
REQUIRED_REAL_TRADING_PHRASE = "I ACCEPT REAL TRADING RISK"

_Noneable_STR_FIELDS = (
    "mt5_terminal_path",
    "mt5_server",
    "real_trading_confirmed_is_str_placeholder",
)


class AppConfig(BaseSettings):
    """All runtime settings.  See ``.env.example`` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

    # ---- environment / infrastructure --------------------------------
    app_env: Literal["development", "production", "test"] = "development"
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: Path | None = None
    database_url: str = "sqlite:///data/aurumx.db"

    # ---- trading safety (spec §54) ------------------------------------
    trading_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("TRADING_ENABLED"),
    )
    dry_run: bool = Field(
        default=True,
        validation_alias=AliasChoices("DRY_RUN"),
    )
    real_trading_confirmed: str = Field(
        default="",
        validation_alias=AliasChoices("REAL_TRADING_CONFIRMED"),
    )

    # ---- market data ---------------------------------------------------
    symbol: str = Field(default="AUTO", validation_alias=AliasChoices("SYMBOL"))
    # NoDecode: pydantic-settings must not JSON-parse the env value; the
    # "M15,H1,H4" comma format is parsed by the field validator below.
    timeframes: Annotated[list[TimeFrame], NoDecode] = Field(
        default=[TimeFrame.M15, TimeFrame.H1, TimeFrame.H4],
        validation_alias=AliasChoices("TIMEFRAMES"),
    )
    candle_count: int = Field(
        default=400, ge=60, le=5000, validation_alias=AliasChoices("CANDLE_COUNT")
    )
    max_tick_age_seconds: float = Field(
        default=60.0, gt=0, validation_alias=AliasChoices("MAX_TICK_AGE_SECONDS")
    )
    candle_freshness_multiplier: float = Field(
        default=3.0,
        gt=1.0,
        le=50.0,
        validation_alias=AliasChoices("CANDLE_FRESHNESS_MULTIPLIER"),
    )
    max_spread_points: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("MAX_SPREAD_POINTS"),
    )

    # ---- risk (enforced by the hard risk gate, Phase 4) ----------------
    risk_per_trade_pct: float = Field(
        default=0.5, gt=0, le=10.0, validation_alias=AliasChoices("RISK_PER_TRADE")
    )
    max_open_positions: int = Field(
        default=1, ge=1, le=20, validation_alias=AliasChoices("MAX_OPEN_POSITIONS")
    )
    max_pending_orders: int = Field(
        default=2, ge=0, le=50, validation_alias=AliasChoices("MAX_PENDING_ORDERS")
    )
    max_daily_loss_pct: float = Field(
        default=2.0, gt=0, le=50.0, validation_alias=AliasChoices("MAX_DAILY_LOSS_PCT")
    )
    max_total_risk_pct: float = Field(
        default=2.0, gt=0, le=50.0, validation_alias=AliasChoices("MAX_TOTAL_RISK_PCT")
    )
    max_drawdown_pct: float = Field(
        default=20.0, gt=0, le=100.0, validation_alias=AliasChoices("MAX_DRAWDOWN_PCT")
    )
    min_reward_risk: float = Field(
        default=1.5, gt=0, le=20.0, validation_alias=AliasChoices("MIN_REWARD_RISK")
    )

    # ---- trade management feature flags (spec §25-27) ------------------
    enable_break_even: bool = Field(
        default=True, validation_alias=AliasChoices("ENABLE_BREAK_EVEN")
    )
    enable_trailing: bool = Field(
        default=False, validation_alias=AliasChoices("ENABLE_TRAILING")
    )
    enable_partial_close: bool = Field(
        default=True, validation_alias=AliasChoices("ENABLE_PARTIAL_CLOSE")
    )

    # ---- MetaTrader 5 connection ---------------------------------------
    mt5_terminal_path: str | None = Field(
        default=None, validation_alias=AliasChoices("MT5_TERMINAL_PATH")
    )
    mt5_login: int | None = Field(
        default=None, ge=1, validation_alias=AliasChoices("MT5_LOGIN")
    )
    mt5_password: SecretStr | None = Field(  # never rendered by repr()/logs
        default=None, validation_alias=AliasChoices("MT5_PASSWORD")
    )
    mt5_server: str | None = Field(
        default=None, validation_alias=AliasChoices("MT5_SERVER")
    )
    mt5_timeout_ms: int = Field(
        default=60000, ge=1000, le=600000, validation_alias=AliasChoices("MT5_TIMEOUT_MS")
    )

    # ------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------
    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return "AUTO"
            return value.upper()
        return value

    @field_validator("timeframes", mode="before")
    @classmethod
    def _parse_timeframes(cls, value: object) -> object:
        """Accept ``"M15,H1,H4"`` (env / .env) as well as lists."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator(
        "mt5_terminal_path",
        "mt5_login",
        "mt5_password",
        "mt5_server",
        "max_spread_points",
        "log_file",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        """Treat empty env values (``MT5_LOGIN=``) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_safety(self) -> AppConfig:
        """Hard safety gate on configuration level (spec §54, §21)."""
        if self.trading_enabled and not self.dry_run:
            if self.real_trading_confirmed.strip() != REQUIRED_REAL_TRADING_PHRASE:
                raise UnsafeConfigurationError(
                    "Live (non-dry-run) trading is requested but the explicit "
                    f"confirmation is missing. Set REAL_TRADING_CONFIRMED="
                    f"'{REQUIRED_REAL_TRADING_PHRASE}' if you really mean it. "
                    "AurumX refuses to run otherwise."
                )
            log.warning(
                "REAL TRADING MODE CONFIGURED — every order will be real money",
                event="REAL_TRADING_CONFIGURED",
                app_env=self.app_env,
            )
        if self.trading_enabled and self.dry_run and self.app_env == "production":
            log.info(
                "dry-run mode active in production: full pipeline, no orders",
                event="DRY_RUN_ACTIVE",
            )
        if self.risk_per_trade_pct > 2.0:
            log.warning(
                "risk per trade above 2% is aggressive for gold",
                event="RISK_CONFIG_AGGRESSIVE",
                risk_per_trade_pct=self.risk_per_trade_pct,
            )
        return self

    # ------------------------------------------------------------------
    # derived properties
    # ------------------------------------------------------------------
    @property
    def trading_mode(self) -> TradingMode:
        """The configured operating mode.

        READ_ONLY (default): observe only.  DRY_RUN: full decision pipeline,
        no orders.  When live trading is confirmed the *intended* mode is
        MT5_REAL, but the runtime account check (Phase 5) decides the actual
        mode: a REAL account trades real, a DEMO account trades demo — and
        real accounts are refused unless the confirmation phrase is present.
        """
        if not self.trading_enabled:
            return TradingMode.READ_ONLY
        if self.dry_run:
            return TradingMode.DRY_RUN
        # Live path: only reachable when the confirmation phrase is set.
        # Demo vs real is verified against the account at runtime (Phase 5).
        return TradingMode.MT5_REAL

    @property
    def is_symbol_auto(self) -> bool:
        return self.symbol == "AUTO"

    def safe_summary(self) -> dict[str, object]:
        """A log/API-safe view of the config: no secrets, ever."""
        return {
            "app_env": self.app_env,
            "trading_enabled": self.trading_enabled,
            "dry_run": self.dry_run,
            "trading_mode": self.trading_mode.value,
            "symbol": self.symbol,
            "timeframes": [tf.value for tf in self.timeframes],
            "candle_count": self.candle_count,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_open_positions": self.max_open_positions,
            "max_pending_orders": self.max_pending_orders,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "mt5_login": self.mt5_login,
            "mt5_server": self.mt5_server,
            "mt5_terminal_path": self.mt5_terminal_path,
            "mt5_password": "***set***" if self.mt5_password is not None else None,
        }

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, env_file: str | Path | None = ".env", **overrides: object) -> AppConfig:
        """Load configuration from environment + optional ``.env`` file."""
        return cls(_env_file=env_file, **overrides)  # type: ignore[call-arg]


__all__ = ["AppConfig", "REQUIRED_REAL_TRADING_PHRASE"]
