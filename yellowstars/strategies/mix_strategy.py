"""
Mix Strategy — Multi-Signal Consensus Trading System.

Combines 5 Malik-inspired sub-strategies and trades only when enough agree:

  S1: MA Crossover Trend     — MA(fast) vs MA(slow) for trend direction
  S2: Bollinger Mean Revert  — Buy lower band dips, sell upper band peaks
  S3: RSI Momentum           — Oversold = buy, overbought = sell
  S4: MACD Signal            — MACD line vs signal line crossovers
  S5: Rate-of-Change Filter  — Momentum acceleration/deceleration

Consensus: Trade only when >= min_votes sub-strategies agree.
Position sizing scales with conviction (more agreeing = larger position).

All parameters are fully configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
)


@dataclass
class MixStrategyParams:
    """All tunable parameters for the Mix Strategy."""
    # S1: MA Crossover
    s1_enabled: bool = True
    s1_fast_period: int = 20
    s1_slow_period: int = 250
    s1_use_ema: bool = False

    # S2: Bollinger Bands
    s2_enabled: bool = True
    s2_period: int = 20
    s2_std: float = 2.0
    s2_buy_pctb: float = 0.2    # Buy below this %B
    s2_sell_pctb: float = 0.8   # Sell above this %B

    # S3: RSI
    s3_enabled: bool = True
    s3_period: int = 14
    s3_oversold: float = 30.0
    s3_overbought: float = 70.0

    # S4: MACD
    s4_enabled: bool = True
    s4_fast: int = 12
    s4_slow: int = 26
    s4_signal: int = 9

    # S5: Rate of Change
    s5_enabled: bool = True
    s5_period: int = 20
    s5_buy_thresh: float = 0.0   # Buy when ROC > this
    s5_sell_thresh: float = 0.0  # Sell when ROC < this

    # Consensus
    min_votes: int = 3           # Need at least N sub-strategies to agree
    scale_by_conviction: bool = True  # Size position by agreement count

    # Risk
    stop_loss_pct: float = 0.0   # 0 = disabled
    take_profit_pct: float = 0.0 # 0 = disabled


class MixStrategy(BaseStrategy):
    """Multi-signal consensus strategy.

    Combines 5 sub-strategies, trades on consensus, sizes by conviction.
    """

    def __init__(
        self,
        settings: Optional[StrategySettings] = None,
        params: Optional[MixStrategyParams] = None,
        **kwargs,
    ):
        settings = settings or StrategySettings(name="mix_strategy")
        if settings.name != "mix_strategy":
            settings.name = "mix_strategy"
        super().__init__(settings, **kwargs)
        self.name = "mix_strategy"
        self.p = params or MixStrategyParams()

    @property
    def min_required_bars(self) -> int:
        periods = [self.p.s1_slow_period, self.p.s2_period, self.p.s3_period,
                   self.p.s4_slow, self.p.s5_period]
        return max(periods) + 10

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # S1: Moving Averages
        ma_func = exponential_moving_average if self.p.s1_use_ema else simple_moving_average
        df["s1_fast"] = ma_func(df["close"], self.p.s1_fast_period)
        df["s1_slow"] = ma_func(df["close"], self.p.s1_slow_period)

        # S2: Bollinger Bands
        df["s2_mid"] = simple_moving_average(df["close"], self.p.s2_period)
        s2_std = df["close"].rolling(window=self.p.s2_period, min_periods=self.p.s2_period).std()
        df["s2_upper"] = df["s2_mid"] + s2_std * self.p.s2_std
        df["s2_lower"] = df["s2_mid"] - s2_std * self.p.s2_std
        band_width = df["s2_upper"] - df["s2_lower"]
        df["s2_pctb"] = np.where(band_width > 0, (df["close"] - df["s2_lower"]) / band_width, 0.5)

        # S3: RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.rolling(window=self.p.s3_period, min_periods=self.p.s3_period).mean()
        avg_loss = loss.rolling(window=self.p.s3_period, min_periods=self.p.s3_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["s3_rsi"] = 100 - (100 / (1 + rs))

        # S4: MACD
        ema_fast = exponential_moving_average(df["close"], self.p.s4_fast)
        ema_slow = exponential_moving_average(df["close"], self.p.s4_slow)
        df["s4_macd"] = ema_fast - ema_slow
        df["s4_signal"] = exponential_moving_average(df["s4_macd"], self.p.s4_signal)
        df["s4_hist"] = df["s4_macd"] - df["s4_signal"]

        # S5: Rate of Change
        df["s5_roc"] = rate_of_change(df["close"], self.p.s5_period)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            df["instrument"] = ""
            return df

        df = self.calculate_indicators(data)

        # Initialize
        df["signal"] = SignalAction.HOLD.value
        df["position_pct"] = 0.0
        df["instrument"] = self.settings.underlying_index
        df["buy_votes"] = 0
        df["sell_votes"] = 0
        df["vote_s1"] = 0
        df["vote_s2"] = 0
        df["vote_s3"] = 0
        df["vote_s4"] = 0
        df["vote_s5"] = 0

        enabled_count = sum([
            self.p.s1_enabled, self.p.s2_enabled, self.p.s3_enabled,
            self.p.s4_enabled, self.p.s5_enabled,
        ])
        if enabled_count == 0:
            logger.warning("MixStrategy: No sub-strategies enabled!")
            return df

        valid_start = self.min_required_bars
        in_position = False
        entry_price = 0.0

        for i in range(valid_start, len(df)):
            row = df.iloc[i]

            # --- Votes ---
            v1, v2, v3, v4, v5 = 0, 0, 0, 0, 0

            # S1: MA Crossover
            if self.p.s1_enabled and not pd.isna(row["s1_fast"]) and not pd.isna(row["s1_slow"]):
                if row["s1_fast"] > row["s1_slow"]:
                    v1 = 1
                elif row["s1_fast"] < row["s1_slow"]:
                    v1 = -1

            # S2: Bollinger %B
            if self.p.s2_enabled and not pd.isna(row["s2_pctb"]):
                if row["s2_pctb"] < self.p.s2_buy_pctb:
                    v2 = 1
                elif row["s2_pctb"] > self.p.s2_sell_pctb:
                    v2 = -1

            # S3: RSI
            if self.p.s3_enabled and not pd.isna(row["s3_rsi"]):
                if row["s3_rsi"] < self.p.s3_oversold:
                    v3 = 1
                elif row["s3_rsi"] > self.p.s3_overbought:
                    v3 = -1

            # S4: MACD
            if self.p.s4_enabled and not pd.isna(row["s4_macd"]) and not pd.isna(row["s4_signal"]):
                if row["s4_macd"] > row["s4_signal"]:
                    v4 = 1
                elif row["s4_macd"] < row["s4_signal"]:
                    v4 = -1

            # S5: ROC
            if self.p.s5_enabled and not pd.isna(row["s5_roc"]):
                if row["s5_roc"] > self.p.s5_buy_thresh:
                    v5 = 1
                elif row["s5_roc"] < self.p.s5_sell_thresh:
                    v5 = -1

            buy_votes = sum(1 for v in [v1, v2, v3, v4, v5] if v == 1)
            sell_votes = sum(1 for v in [v1, v2, v3, v4, v5] if v == -1)

            # Store votes
            df.iat[i, df.columns.get_loc("vote_s1")] = v1
            df.iat[i, df.columns.get_loc("vote_s2")] = v2
            df.iat[i, df.columns.get_loc("vote_s3")] = v3
            df.iat[i, df.columns.get_loc("vote_s4")] = v4
            df.iat[i, df.columns.get_loc("vote_s5")] = v5
            df.iat[i, df.columns.get_loc("buy_votes")] = buy_votes
            df.iat[i, df.columns.get_loc("sell_votes")] = sell_votes

            # --- Consensus ---
            buy_signal = buy_votes >= self.p.min_votes
            sell_signal = sell_votes >= self.p.min_votes

            # Position sizing by conviction
            if self.p.scale_by_conviction and enabled_count > 0:
                conviction = max(buy_votes, sell_votes) / enabled_count
            else:
                conviction = 1.0

            # --- Execution ---
            if buy_signal and not in_position:
                df.iat[i, df.columns.get_loc("signal")] = SignalAction.BUY.value
                df.iat[i, df.columns.get_loc("position_pct")] = conviction
                in_position = True
                entry_price = row["close"]

            elif sell_signal and in_position:
                df.iat[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                in_position = False
                entry_price = 0.0

            elif in_position:
                # Stay in position, adjust size by conviction
                df.iat[i, df.columns.get_loc("signal")] = SignalAction.HOLD.value
                df.iat[i, df.columns.get_loc("position_pct")] = conviction

                # Stop loss check
                if self.p.stop_loss_pct > 0 and entry_price > 0:
                    pnl_pct = (row["close"] - entry_price) / entry_price * 100
                    if pnl_pct < -self.p.stop_loss_pct:
                        df.iat[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                        df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                        in_position = False
                        entry_price = 0.0

                # Take profit check
                if self.p.take_profit_pct > 0 and entry_price > 0 and in_position:
                    pnl_pct = (row["close"] - entry_price) / entry_price * 100
                    if pnl_pct >= self.p.take_profit_pct:
                        df.iat[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                        df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                        in_position = False
                        entry_price = 0.0

        # Count signals
        buys = (df["signal"] == SignalAction.BUY.value).sum()
        sells = (df["signal"] == SignalAction.SELL.value).sum()
        logger.info(
            f"MixStrategy signals: {buys} buys, {sells} sells, "
            f"min_votes={self.p.min_votes}/{enabled_count}, "
            f"period={df.index[0].date()} to {df.index[-1].date()}"
        )

        return df

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "type": "mix_strategy",
            "s1_enabled": self.p.s1_enabled,
            "s1_fast": self.p.s1_fast_period,
            "s1_slow": self.p.s1_slow_period,
            "s2_enabled": self.p.s2_enabled,
            "s2_period": self.p.s2_period,
            "s3_enabled": self.p.s3_enabled,
            "s3_period": self.p.s3_period,
            "s4_enabled": self.p.s4_enabled,
            "s4_fast": self.p.s4_fast,
            "s4_slow": self.p.s4_slow,
            "s5_enabled": self.p.s5_enabled,
            "s5_period": self.p.s5_period,
            "min_votes": self.p.min_votes,
            "scale_by_conviction": self.p.scale_by_conviction,
            "stop_loss_pct": self.p.stop_loss_pct,
            "take_profit_pct": self.p.take_profit_pct,
        }
