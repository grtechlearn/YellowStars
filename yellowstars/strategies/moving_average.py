"""
Moving Average Crossover Strategy.

A simpler, more generic strategy that can be applied to any market/asset.
Useful as a baseline comparison and for international/crypto markets.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.base import (
    BaseStrategy,
    simple_moving_average,
    exponential_moving_average,
    rate_of_change,
)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Dual Moving Average Crossover Strategy.

    Rules:
    - Fast MA crosses above Slow MA → BUY (go long)
    - Fast MA crosses below Slow MA → SELL (go short/flat)
    - Position sizing based on distance from MAs

    Works with any asset class (equities, crypto, forex).
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.fast_period = self.settings.fast_ma_period
        self.slow_period = self.settings.slow_ma_period
        self.use_ema = kwargs.get("use_ema", False)  # EMA vs SMA

    @property
    def min_required_bars(self) -> int:
        return self.slow_period + 5

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        df = data.copy()

        ma_func = exponential_moving_average if self.use_ema else simple_moving_average

        df["ma_fast"] = ma_func(df["close"], self.fast_period)
        df["ma_slow"] = ma_func(df["close"], self.slow_period)

        # Crossover detection
        df["fast_above_slow"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
        df["crossover"] = df["fast_above_slow"].diff()  # +1 = golden cross, -1 = death cross

        # Trend strength (distance between MAs as % of price)
        df["ma_spread"] = (df["ma_fast"] - df["ma_slow"]) / df["ma_slow"] * 100

        # ROC for velocity
        df["roc"] = rate_of_change(df["close"], self.settings.roc_period)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals."""
        if not self.validate_data(data):
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            return df

        df = self.calculate_indicators(data)

        # Initialize signals
        df["signal"] = SignalAction.HOLD.value
        df["position_pct"] = 0.0
        df["instrument"] = self.settings.long_asset

        valid_start = self.slow_period
        in_position = False

        for i in range(valid_start, len(df)):
            row = df.iloc[i]

            if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]):
                continue

            # Golden cross → Buy
            if row["crossover"] == 1:
                df.iloc[i, df.columns.get_loc("signal")] = SignalAction.BUY.value
                df.iloc[i, df.columns.get_loc("position_pct")] = self.settings.max_position_pct
                df.iloc[i, df.columns.get_loc("instrument")] = self.settings.long_asset
                in_position = True

            # Death cross → Sell
            elif row["crossover"] == -1:
                # If using short asset (like SQQQ), switch to it
                if self.settings.short_asset:
                    df.iloc[i, df.columns.get_loc("signal")] = SignalAction.FLIP.value
                    df.iloc[i, df.columns.get_loc("position_pct")] = self.settings.max_position_pct * 0.5
                    df.iloc[i, df.columns.get_loc("instrument")] = self.settings.short_asset
                else:
                    df.iloc[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                    df.iloc[i, df.columns.get_loc("position_pct")] = 0.0
                in_position = False

            # Hold existing position
            elif in_position:
                # Adjust size based on trend strength
                spread = abs(row["ma_spread"])
                if spread > 5:
                    size = self.settings.max_position_pct
                elif spread > 2:
                    size = self.settings.max_position_pct * 0.7
                else:
                    size = self.settings.max_position_pct * 0.5

                df.iloc[i, df.columns.get_loc("position_pct")] = size
                df.iloc[i, df.columns.get_loc("signal")] = SignalAction.HOLD.value

        logger.info(
            f"MA Crossover signals generated: {len(df) - valid_start} bars, "
            f"final: {df.iloc[-1]['signal']} at {df.iloc[-1]['position_pct']:.1%}"
        )

        return df
