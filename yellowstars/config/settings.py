"""
Central configuration management for YellowStars.

Loads settings from YAML files and environment variables.
Hierarchy: defaults < config.yaml < environment variables < CLI args
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

from yellowstars.core.models import (
    AlertConfig,
    BrokerConfig,
    DataProviderConfig,
    RiskConfig,
    TaxConfig,
    TradingMode,
)


@dataclass
class StrategySettings:
    """Configuration for strategy parameters."""
    name: str = "malik_white_light"
    enabled: bool = True
    # Trend Following
    fast_ma_period: int = 20
    slow_ma_period: int = 250
    # Mean Reversion / Velocity
    roc_period: int = 20           # Rate of Change lookback
    roc_threshold: float = -5.0    # Velocity threshold for position reduction
    # Position Sizing
    max_position_pct: float = 1.0  # Max allocation (1.0 = 100%)
    min_position_pct: float = 0.0  # Min allocation
    # Assets
    long_asset: str = "TQQQ"
    short_asset: str = "SQQQ"
    benchmark_asset: str = "QQQ"
    underlying_index: str = "NDX"
    # Execution Timing
    execution_minutes_before_close: int = 15
    # Custom params (for user-defined strategies)
    custom_params: dict = field(default_factory=dict)


@dataclass
class BacktestSettings:
    """Backtesting configuration."""
    start_date: str = "1985-01-01"      # 42+ years back
    end_date: str = ""                   # Empty = today
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.05           # 5 basis points
    benchmark_symbol: str = "QQQ"
    use_adjusted_close: bool = True
    include_dividends: bool = False


@dataclass
class Settings:
    """Master settings for the entire platform."""
    # General
    project_name: str = "YellowStars"
    version: str = "0.1.0"
    trading_mode: TradingMode = TradingMode.PAPER
    log_level: str = "INFO"
    data_directory: str = "data"
    reports_directory: str = "reports"

    # Sub-configs
    data_provider: DataProviderConfig = field(default_factory=DataProviderConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    strategies: list[StrategySettings] = field(default_factory=lambda: [StrategySettings()])

    def get_strategy(self, name: str) -> Optional[StrategySettings]:
        """Get a strategy config by name."""
        for s in self.strategies:
            if s.name == name:
                return s
        return None


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(config: dict) -> dict:
    """Override config values from environment variables.

    Convention: YS_SECTION__KEY=value
    Example: YS_DATA_PROVIDER__API_KEY=abc123
    """
    prefix = "YS_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        final_key = parts[-1]

        # Type coercion
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        current[final_key] = value
    return config


def _dict_to_settings(config: dict) -> Settings:
    """Convert a flat dictionary to Settings dataclass."""
    settings = Settings()

    # General settings
    settings.trading_mode = TradingMode(config.get("trading_mode", "paper"))
    settings.log_level = config.get("log_level", "INFO")
    settings.data_directory = config.get("data_directory", "data")
    settings.reports_directory = config.get("reports_directory", "reports")

    # Data provider
    dp = config.get("data_provider", {})
    settings.data_provider = DataProviderConfig(
        name=dp.get("name", "polygon"),
        api_key=dp.get("api_key", ""),
        base_url=dp.get("base_url", ""),
        rate_limit_per_minute=dp.get("rate_limit_per_minute", 5),
        cache_enabled=dp.get("cache_enabled", True),
        cache_directory=dp.get("cache_directory", "data/cache"),
    )

    # Broker
    br = config.get("broker", {})
    settings.broker = BrokerConfig(
        name=br.get("name", "alpaca"),
        api_key=br.get("api_key", ""),
        api_secret=br.get("api_secret", ""),
        base_url=br.get("base_url", ""),
        paper_trading=br.get("paper_trading", True),
        max_order_size=br.get("max_order_size", 100000.0),
        commission_per_trade=br.get("commission_per_trade", 0.0),
        commission_per_share=br.get("commission_per_share", 0.0),
    )

    # Risk
    rk = config.get("risk", {})
    settings.risk = RiskConfig(
        max_portfolio_drawdown_pct=rk.get("max_portfolio_drawdown_pct", 30.0),
        max_single_position_pct=rk.get("max_single_position_pct", 100.0),
        max_daily_loss_pct=rk.get("max_daily_loss_pct", 10.0),
        min_cash_reserve_pct=rk.get("min_cash_reserve_pct", 0.0),
        max_leverage=rk.get("max_leverage", 1.0),
        stop_loss_pct=rk.get("stop_loss_pct"),
    )

    # Tax
    tx = config.get("tax", {})
    settings.tax = TaxConfig(
        country=tx.get("country", "US"),
        short_term_rate=tx.get("short_term_rate", 0.37),
        long_term_rate=tx.get("long_term_rate", 0.20),
        state_tax_rate=tx.get("state_tax_rate", 0.0),
        wash_sale_rule=tx.get("wash_sale_rule", True),
        tax_loss_harvesting=tx.get("tax_loss_harvesting", False),
    )

    # Alerts
    al = config.get("alerts", {})
    settings.alerts = AlertConfig(
        enabled=al.get("enabled", True),
        telegram_bot_token=al.get("telegram_bot_token", ""),
        telegram_chat_id=al.get("telegram_chat_id", ""),
        discord_webhook_url=al.get("discord_webhook_url", ""),
    )

    # Backtest
    bt = config.get("backtest", {})
    settings.backtest = BacktestSettings(
        start_date=bt.get("start_date", "1985-01-01"),
        end_date=bt.get("end_date", ""),
        initial_capital=bt.get("initial_capital", 100000.0),
        commission_per_trade=bt.get("commission_per_trade", 0.0),
        slippage_pct=bt.get("slippage_pct", 0.05),
        benchmark_symbol=bt.get("benchmark_symbol", "QQQ"),
    )

    # Strategies
    strats = config.get("strategies", [])
    if strats:
        settings.strategies = []
        for s in strats:
            settings.strategies.append(StrategySettings(
                name=s.get("name", "default"),
                enabled=s.get("enabled", True),
                fast_ma_period=s.get("fast_ma_period", 20),
                slow_ma_period=s.get("slow_ma_period", 250),
                roc_period=s.get("roc_period", 20),
                roc_threshold=s.get("roc_threshold", -5.0),
                max_position_pct=s.get("max_position_pct", 1.0),
                min_position_pct=s.get("min_position_pct", 0.0),
                long_asset=s.get("long_asset", "TQQQ"),
                short_asset=s.get("short_asset", "SQQQ"),
                benchmark_asset=s.get("benchmark_asset", "QQQ"),
                underlying_index=s.get("underlying_index", "NDX"),
                execution_minutes_before_close=s.get("execution_minutes_before_close", 15),
                custom_params=s.get("custom_params", {}),
            ))

    return settings


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load settings from YAML file, with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Populated Settings dataclass.
    """
    # Load .env file for API keys and secrets
    load_dotenv()

    config = {}

    # Load from YAML if exists
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            file_config = yaml.safe_load(f) or {}
            config = _deep_update(config, file_config)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults + env vars.")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Convert to Settings
    settings = _dict_to_settings(config)

    logger.info(f"Settings loaded: mode={settings.trading_mode}, "
                f"data_provider={settings.data_provider.name}")

    return settings
