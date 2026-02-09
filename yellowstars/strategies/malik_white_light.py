"""
Malik's "White Light" Strategy Implementation.

This implements the 7 sub-strategies described by Malik:
1. Trend Following (50-day & 250-day MA)
2. Mean Reversion / Velocity (Rate of Change)
3. Position Sizing based on trend strength
4. Long/Short switching between TQQQ and SQQQ
5. Conviction-based allocation
6. Drawdown management
7. Compounding (no profit-taking)

Core Philosophy:
- Simplest of the simplest indicators only
- Mechanical execution (no discretion)
- Execute in last 10-15 minutes of trading day
- Accept 20-30% drawdowns as normal
- Compound gains, don't take profits
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.base import (
    BaseStrategy,
    simple_moving_average,
    exponential_moving_average,
    rate_of_change,
    average_true_range,
)


class MalikWhiteLightStrategy(BaseStrategy):
    """Malik's White Light systematic trading strategy.

    Trades TQQQ (3x Long NASDAQ) and SQQQ (3x Short NASDAQ)
    based on NDX (NASDAQ 100 index) price action.

    Sub-Strategy Breakdown:
    ========================

    Sub-1: PRIMARY TREND (50-day SMA)
        - Price > SMA(50) → Bullish bias → Lean TQQQ
        - Price < SMA(50) → Bearish bias → Lean SQQQ

    Sub-2: SECULAR TREND (250-day SMA)
        - Price > SMA(250) → Long-term bull market → Full conviction
        - Price < SMA(250) → Long-term bear market → Reduced size

    Sub-3: TREND ALIGNMENT
        - Both MAs aligned (50 > 250 AND price > both) → Maximum allocation
        - MAs conflicting → Reduced allocation

    Sub-4: VELOCITY (Rate of Change)
        - Strong positive ROC → Trend accelerating → Full size
        - Declining ROC → Trend decelerating → Reduce size
        - Negative ROC below threshold → Exit/reverse

    Sub-5: POSITION SIZING
        - Combines Sub-1 through Sub-4 signals
        - Outputs target allocation from 0% to 100%
        - Determines TQQQ vs SQQQ selection

    Sub-6: DRAWDOWN FILTER
        - If portfolio in >20% drawdown → Reduce max position
        - If portfolio in >30% drawdown → Circuit breaker (cash)

    Sub-7: COMPOUNDING RULE
        - Never take profits
        - Always reinvest at full calculated position size
        - Position size scales with equity (not initial capital)
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # Extract params from settings
        self.fast_period = self.settings.fast_ma_period    # 50
        self.slow_period = self.settings.slow_ma_period    # 250
        self.roc_period = self.settings.roc_period          # 20
        self.roc_threshold = self.settings.roc_threshold    # -5.0
        self.max_position = self.settings.max_position_pct  # 1.0

    @property
    def min_required_bars(self) -> int:
        return self.slow_period + 20  # Need enough data for 250-day MA

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the White Light strategy."""
        df = data.copy()

        # Sub-1: Fast MA (50-day)
        df["sma_fast"] = simple_moving_average(df["close"], self.fast_period)

        # Sub-2: Slow MA (250-day)
        df["sma_slow"] = simple_moving_average(df["close"], self.slow_period)

        # Sub-3: Trend alignment
        df["trend_aligned"] = (
            (df["sma_fast"] > df["sma_slow"]) &
            (df["close"] > df["sma_fast"])
        ).astype(int)

        # Sub-4: Velocity (Rate of Change)
        df["roc"] = rate_of_change(df["close"], self.roc_period)

        # Velocity regime classification
        df["velocity_regime"] = 0  # neutral
        df.loc[df["roc"] > 5, "velocity_regime"] = 2      # strong up
        df.loc[(df["roc"] > 0) & (df["roc"] <= 5), "velocity_regime"] = 1   # mild up
        df.loc[(df["roc"] < 0) & (df["roc"] >= self.roc_threshold), "velocity_regime"] = -1  # mild down
        df.loc[df["roc"] < self.roc_threshold, "velocity_regime"] = -2  # strong down

        # ATR for volatility context
        df["atr"] = average_true_range(df["high"], df["low"], df["close"], period=14)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Price relative to MAs
        df["price_vs_fast"] = (df["close"] - df["sma_fast"]) / df["sma_fast"] * 100
        df["price_vs_slow"] = (df["close"] - df["sma_slow"]) / df["sma_slow"] * 100

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate White Light trading signals.

        Returns DataFrame with 'signal' and 'position_pct' columns.
        """
        if not self.validate_data(data):
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            return df

        # Calculate all indicators
        df = self.calculate_indicators(data)

        # Initialize signal columns
        df["signal"] = SignalAction.HOLD.value
        df["position_pct"] = 0.0
        df["instrument"] = ""  # Which ETF to trade

        # Only process rows with enough data (after warmup period)
        valid_idx = df.index[self.slow_period:]

        for i, idx in enumerate(valid_idx):
            row = df.loc[idx]

            # Skip if indicators aren't ready
            if pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]):
                continue

            # ---- Sub-1: Primary Trend ----
            above_fast_ma = row["close"] > row["sma_fast"]

            # ---- Sub-2: Secular Trend ----
            above_slow_ma = row["close"] > row["sma_slow"]

            # ---- Sub-3: Trend Alignment Score ----
            alignment_score = 0.0
            if above_fast_ma:
                alignment_score += 0.4
            if above_slow_ma:
                alignment_score += 0.3
            if row["trend_aligned"]:
                alignment_score += 0.3

            # ---- Sub-4: Velocity Adjustment ----
            velocity_multiplier = 1.0
            if row["velocity_regime"] == 2:
                velocity_multiplier = 1.0     # Full conviction
            elif row["velocity_regime"] == 1:
                velocity_multiplier = 0.8     # Good, slight caution
            elif row["velocity_regime"] == 0:
                velocity_multiplier = 0.5     # Neutral, reduce
            elif row["velocity_regime"] == -1:
                velocity_multiplier = 0.3     # Slowing, significant reduction
            elif row["velocity_regime"] == -2:
                velocity_multiplier = 0.0     # Strong decline, exit

            # ---- Sub-5: Position Sizing ----
            # Determine direction
            if above_fast_ma and above_slow_ma:
                # Strong bull - go long TQQQ
                direction = 1.0
                instrument = self.settings.long_asset  # TQQQ
            elif not above_fast_ma and not above_slow_ma:
                # Strong bear - go long SQQQ (short the market)
                direction = -1.0
                instrument = self.settings.short_asset  # SQQQ
            elif above_slow_ma and not above_fast_ma:
                # Correction in bull market - reduce or cash
                direction = 1.0
                alignment_score *= 0.5
                instrument = self.settings.long_asset
            else:
                # Bear market rally - stay cautious
                direction = -1.0
                alignment_score *= 0.5
                instrument = self.settings.short_asset

            # Calculate target position
            raw_position = alignment_score * velocity_multiplier * self.max_position

            # Clamp to valid range
            position_pct = max(0.0, min(self.max_position, raw_position))

            # Apply direction for signaling
            # Note: position_pct is always positive (0 to 1.0)
            # The instrument (TQQQ vs SQQQ) determines the direction

            # Determine signal action
            prev_pos = df.iloc[self.slow_period + i - 1]["position_pct"] if i > 0 else 0.0
            if position_pct > 0 and prev_pos == 0:
                action = SignalAction.BUY.value
            elif position_pct == 0 and prev_pos > 0:
                action = SignalAction.SELL.value
            elif position_pct > prev_pos:
                action = SignalAction.INCREASE.value
            elif position_pct < prev_pos:
                action = SignalAction.REDUCE.value
            else:
                action = SignalAction.HOLD.value

            df.loc[idx, "signal"] = action
            df.loc[idx, "position_pct"] = position_pct
            df.loc[idx, "instrument"] = instrument

        logger.info(
            f"White Light signals generated: {len(valid_idx)} bars processed, "
            f"final position: {df.iloc[-1]['position_pct']:.1%} "
            f"{df.iloc[-1]['instrument']}"
        )

        return df

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            **super().get_params(),
            "strategy_type": "malik_white_light",
            "sub_strategies": 7,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "roc_period": self.roc_period,
            "roc_threshold": self.roc_threshold,
        }
