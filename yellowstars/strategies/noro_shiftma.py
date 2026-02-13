"""
Noro's WhiteBox ShiftMA Strategy.

Original: Noro (2019), Pine Script v4.
Python implementation for backtesting.

Strategy Logic:
  1. Calculate a configurable MA (SMA/EMA/WMA) on a chosen price source
  2. Shift MA UP by shortlevel% to create a Short entry line
  3. Shift MA DOWN by longlevel% to create a Long entry line
  4. LONG entry: when price (low) touches the Long line from above (buy the dip)
  5. SHORT entry: when price (high) touches the Short line from below (sell the rip)
  6. Optional close-shift take-profit exits near the MA

The strategy buys mean-reversion dips and sells mean-reversion rips
around a shifted moving average envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
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
)


def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (WMA)."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


@dataclass
class NoroShiftMAParams:
    """All tunable parameters for the Noro ShiftMA strategy."""
    # Direction
    enable_long: bool = True
    enable_short: bool = False   # Default long-only for index backtesting

    # MA Settings
    ma_length: int = 3
    ma_offset: int = 0
    ma_type: str = "SMA"         # SMA, EMA, WMA
    ma_source: str = "OHLC4"     # Open, High, Low, Close, HL2, HLC3, OHLC4, OC2, PCMA

    # Shift Levels
    short_level: float = 10.0    # MA shifted UP by this % for short entry
    long_level: float = -5.0     # MA shifted DOWN by this % for long entry
    close_shift: float = 0.0     # Take-profit offset from MA (0 = disabled)


class NoroShiftMAStrategy(BaseStrategy):
    """Noro's WhiteBox ShiftMA — shifted MA envelope mean-reversion.

    Buy when price dips to longLine (MA shifted down).
    Sell when price rips to shortLine (MA shifted up).
    Optional TP at close-shift lines near MA.
    """

    def __init__(
        self,
        settings: Optional[StrategySettings] = None,
        params: Optional[NoroShiftMAParams] = None,
        **kwargs,
    ):
        settings = settings or StrategySettings(name="noro_shiftma")
        if settings.name != "noro_shiftma":
            settings.name = "noro_shiftma"
        super().__init__(settings, **kwargs)
        self.name = "noro_shiftma"
        self.p = params or NoroShiftMAParams()

    @property
    def min_required_bars(self) -> int:
        return self.p.ma_length + self.p.ma_offset + 5

    def _compute_source(self, df: pd.DataFrame) -> pd.Series:
        """Compute the price source series."""
        s = self.p.ma_source
        if s == "Open":
            return df["open"]
        elif s == "High":
            return df["high"]
        elif s == "Low":
            return df["low"]
        elif s == "Close":
            return df["close"]
        elif s == "HL2":
            return (df["high"] + df["low"]) / 2.0
        elif s == "HLC3":
            return (df["high"] + df["low"] + df["close"]) / 3.0
        elif s == "OC2":
            return (df["open"] + df["close"]) / 2.0
        elif s == "PCMA":
            # Price Channel MA: midpoint of highest high / lowest low
            hh = df["high"].rolling(window=self.p.ma_length, min_periods=self.p.ma_length).max()
            ll = df["low"].rolling(window=self.p.ma_length, min_periods=self.p.ma_length).min()
            return (hh + ll) / 2.0
        else:  # OHLC4 default
            return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    def _compute_ma(self, src: pd.Series) -> pd.Series:
        """Compute the MA on the source."""
        t = self.p.ma_type
        p = self.p.ma_length

        if self.p.ma_source == "PCMA":
            # PCMA already computed in source; return it directly
            return src

        if t == "EMA":
            ma = exponential_moving_average(src, p)
        elif t == "WMA":
            ma = weighted_moving_average(src, p)
        else:  # SMA default
            ma = simple_moving_average(src, p)

        return ma

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Source and MA
        src = self._compute_source(df)
        ma2 = self._compute_ma(src)

        # Apply offset (lookback shift)
        if self.p.ma_offset > 0:
            df["ma"] = ma2.shift(self.p.ma_offset)
        else:
            df["ma"] = ma2

        # Shift lines
        df["short_line"] = df["ma"] * (100.0 + self.p.short_level) / 100.0
        df["long_line"] = df["ma"] * (100.0 + self.p.long_level) / 100.0

        # Close-shift (take-profit) lines
        df["cs_long"] = df["ma"] - (df["ma"] / 100.0 * self.p.close_shift)
        df["cs_short"] = df["ma"] + (df["ma"] / 100.0 * self.p.close_shift)

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

        valid_start = self.min_required_bars
        in_position = False
        position_side = ""  # "long" or "short"
        entry_price_val = 0.0

        for i in range(valid_start, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if pd.isna(row["ma"]) or pd.isna(prev.get("long_line", None)):
                continue

            prev_long_line = prev["long_line"]
            prev_short_line = prev["short_line"]

            # --- Entry signals (when flat) ---
            if not in_position:
                # LONG: low touches or dips below previous long_line
                if self.p.enable_long and row["low"] <= prev_long_line:
                    df.iat[i, df.columns.get_loc("signal")] = SignalAction.BUY.value
                    df.iat[i, df.columns.get_loc("position_pct")] = 1.0
                    in_position = True
                    position_side = "long"
                    # Entry at the long_line (limit order fill)
                    entry_price_val = prev_long_line

                # SHORT: high touches or breaks above previous short_line
                elif self.p.enable_short and row["high"] >= prev_short_line:
                    df.iat[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                    df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                    in_position = True
                    position_side = "short"
                    entry_price_val = prev_short_line

            # --- Exit signals (when in position) ---
            elif in_position:
                if position_side == "long":
                    # TP exit: price reaches close-shift long line (or MA if cs=0)
                    exit_target = row["cs_long"] if self.p.close_shift != 0 else row["ma"]
                    # Exit when high reaches the exit target (take profit)
                    if row["high"] >= exit_target:
                        df.iat[i, df.columns.get_loc("signal")] = SignalAction.SELL.value
                        df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                        in_position = False
                        position_side = ""

                elif position_side == "short":
                    # TP exit for short: price drops to close-shift short line
                    exit_target = row["cs_short"] if self.p.close_shift != 0 else row["ma"]
                    if row["low"] <= exit_target:
                        # Close short = buy back
                        df.iat[i, df.columns.get_loc("signal")] = SignalAction.BUY.value
                        df.iat[i, df.columns.get_loc("position_pct")] = 0.0
                        in_position = False
                        position_side = ""

        # Count signals
        buys = (df["signal"] == SignalAction.BUY.value).sum()
        sells = (df["signal"] == SignalAction.SELL.value).sum()
        logger.info(
            f"NoroShiftMA signals: {buys} buys, {sells} sells, "
            f"MA={self.p.ma_type}({self.p.ma_length}), "
            f"long={self.p.long_level}%, short={self.p.short_level}%, "
            f"period={df.index[0].date()} to {df.index[-1].date()}"
        )

        return df

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "type": "noro_shiftma",
            "ma_type": self.p.ma_type,
            "ma_length": self.p.ma_length,
            "ma_offset": self.p.ma_offset,
            "ma_source": self.p.ma_source,
            "short_level": self.p.short_level,
            "long_level": self.p.long_level,
            "close_shift": self.p.close_shift,
            "enable_long": self.p.enable_long,
            "enable_short": self.p.enable_short,
        }
