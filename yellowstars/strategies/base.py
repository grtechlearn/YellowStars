"""
Abstract base class for all trading strategies.

Every strategy must implement the `generate_signals` method
which takes historical data and returns a DataFrame of signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies operate on a DataFrame of OHLCV data
    and produce a signals DataFrame with position sizing.

    Convention:
        - Signal column 'position_pct' ranges from -1.0 to 1.0
          where 1.0 = 100% long, -1.0 = 100% short, 0 = flat
        - Signal column 'action' is a SignalAction enum
        - All indicators are added as columns to the data DataFrame
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        self.settings = settings or StrategySettings()
        self.name = self.settings.name
        self._params = kwargs

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  and a DatetimeIndex.

        Returns:
            DataFrame with all original columns PLUS:
              - 'signal': SignalAction value (buy/sell/hold)
              - 'position_pct': Target position as fraction (-1.0 to 1.0)
              - Any additional indicator columns the strategy uses
        """
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed by the strategy.

        Override this to add strategy-specific indicators.
        Called internally by generate_signals().
        """
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data meets strategy requirements.

        Override for strategies that need minimum data lengths.
        """
        if data.empty:
            logger.warning(f"Strategy {self.name}: Empty data provided")
            return False

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            logger.error(f"Strategy {self.name}: Missing columns: {missing}")
            return False

        # Check minimum data length
        min_bars = self.min_required_bars
        if len(data) < min_bars:
            logger.warning(
                f"Strategy {self.name}: Need at least {min_bars} bars, "
                f"got {len(data)}"
            )
            return False

        return True

    @property
    def min_required_bars(self) -> int:
        """Minimum number of bars needed before strategy can generate signals.

        Override in subclass if strategy needs a warmup period.
        """
        return max(self.settings.slow_ma_period, self.settings.fast_ma_period) + 10

    def get_params(self) -> dict:
        """Return all strategy parameters for logging/reporting."""
        return {
            "name": self.name,
            "fast_ma_period": self.settings.fast_ma_period,
            "slow_ma_period": self.settings.slow_ma_period,
            "roc_period": self.settings.roc_period,
            "roc_threshold": self.settings.roc_threshold,
            "max_position_pct": self.settings.max_position_pct,
            "long_asset": self.settings.long_asset,
            "short_asset": self.settings.short_asset,
            **self._params,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================
# Common Indicator Functions (used by multiple strategies)
# ============================================================

def simple_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return series.rolling(window=period, min_periods=period).mean()


def exponential_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def rate_of_change(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change (ROC) - percentage change over N periods."""
    return series.pct_change(periods=period) * 100


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (ATR) - volatility indicator."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Calculate drawdown percentage from equity curve."""
    running_max = equity.expanding().max()
    return (equity - running_max) / running_max * 100
