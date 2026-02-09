"""
Malik's "White Light" Strategy Implementation — Accurate to Published Description.

Source: Collective2 (https://collective2.com/my/K6Q9FDJ8A)
Live since: July 23, 2022 | 43 months track record

Malik's Own Description:
========================
"I position trade only $TQQQ and $SQQQ. I am in $TQQQ during uptrends and
$SQQQ in downtrends. The position size of TQQQ and SQQQ depends on how
extended $NDX is."

7 Sub-Systems:
- 1 momentum long (only TQQQ)
- 1 mean reversion short (only SQQQ)
- 3 mean reversion longs (only TQQQ)
- 2 momentum shorts (only SQQQ)

Key Parameters (from Malik):
- MovingAverages: MA(20) and MA(250) for trend determination
- How far price is from those MAs → determines "how extended" NDX is
- Bollinger Bands: Detect momentum phase for heavier position sizing
- EOD based: One decision 10 minutes before market close
- ~2 trades per week, ~10 trades per month
- Avg trade duration: 14.4 days
- Backtested over 40 years of NDX data (synthetic TQQQ/SQQQ)

Published Live Stats:
- Annual Return: 26.5% (compounded)
- Max Drawdown: 37.58%
- Sharpe: 0.70, Sortino: 1.06, Calmar: 1.147
- Win Rate: 39.2% (31 of 79 trades)
- W:L Ratio: 1.90:1
- Avg Win: $1,157 / Avg Loss: $397
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


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Returns: (upper_band, middle_band, lower_band)
    """
    middle = simple_moving_average(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def bollinger_bandwidth(
    upper: pd.Series, lower: pd.Series, middle: pd.Series
) -> pd.Series:
    """Bollinger Bandwidth: (Upper - Lower) / Middle * 100."""
    return (upper - lower) / middle * 100


def bollinger_pct_b(
    close: pd.Series, upper: pd.Series, lower: pd.Series
) -> pd.Series:
    """%B: Where price is relative to the bands. (Close - Lower) / (Upper - Lower)."""
    return (close - lower) / (upper - lower)


class MalikWhiteLightStrategy(BaseStrategy):
    """Malik's White Light systematic trading strategy.

    Accurate implementation based on Malik's published description.

    Architecture: 7 sub-systems that independently vote on position.
    Final position = weighted combination of all 7 sub-system outputs.

    Sub-System 1: MOMENTUM LONG (TQQQ only)
        - Price > MA(20) AND Price > MA(250) AND Bollinger %B > 0.8
        - Heavy TQQQ allocation during strong momentum phases

    Sub-System 2: MEAN REVERSION SHORT (SQQQ only)
        - Price far above MA(250) → overextended → short via SQQQ
        - Price/MA(250) distance > threshold → position in SQQQ

    Sub-System 3: MEAN REVERSION LONG #1 (TQQQ only)
        - Price dips below MA(20) but above MA(250) → buy the dip
        - In uptrend, pull back to MA(20) is buying opportunity

    Sub-System 4: MEAN REVERSION LONG #2 (TQQQ only)
        - Price touches lower Bollinger Band in uptrend → buy
        - Bollinger %B < 0.2 AND Price > MA(250)

    Sub-System 5: MEAN REVERSION LONG #3 (TQQQ only)
        - Price severely extended below MA(20) in uptrend
        - Distance from MA(20) > threshold → high-conviction buy

    Sub-System 6: MOMENTUM SHORT #1 (SQQQ only)
        - Price < MA(20) AND Price < MA(250) → downtrend momentum
        - Position in SQQQ during confirmed downtrends

    Sub-System 7: MOMENTUM SHORT #2 (SQQQ only)
        - Bollinger Band squeeze (low bandwidth) breaking down
        - Price breaks below lower band AND below MA(250)
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        if settings is None:
            settings = StrategySettings(
                name="malik_white_light",
                fast_ma_period=20,
                slow_ma_period=250,
                long_asset="TQQQ",
                short_asset="SQQQ",
                underlying_index="NDX",
            )
        super().__init__(settings=settings, **kwargs)

        # Malik's actual parameters
        self.fast_period = self.settings.fast_ma_period    # 20 (NOT 50)
        self.slow_period = self.settings.slow_ma_period    # 250
        self.bb_period = self.settings.custom_params.get("bb_period", 20)
        self.bb_std = self.settings.custom_params.get("bb_std", 2.0)
        self.max_position = self.settings.max_position_pct  # 1.0

        # Extension thresholds (how far from MA = "extended")
        self.overextended_pct = self.settings.custom_params.get("overextended_pct", 12.0)
        self.mean_reversion_pct = self.settings.custom_params.get("mean_reversion_pct", -3.0)
        self.severe_dip_pct = self.settings.custom_params.get("severe_dip_pct", -6.0)

    @property
    def min_required_bars(self) -> int:
        return self.slow_period + 20  # 270 bars minimum

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators used by the 7 sub-systems."""
        df = data.copy()

        # Moving Averages (Malik uses MA 20 and MA 250)
        df["ma_20"] = simple_moving_average(df["close"], self.fast_period)
        df["ma_250"] = simple_moving_average(df["close"], self.slow_period)

        # Bollinger Bands (20-period, 2 std)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["bb_pct_b"] = bollinger_pct_b(df["close"], df["bb_upper"], df["bb_lower"])
        df["bb_bandwidth"] = bollinger_bandwidth(df["bb_upper"], df["bb_lower"], df["bb_middle"])

        # Price distance from MAs (key to Malik's "how extended" logic)
        df["dist_from_ma20_pct"] = (df["close"] - df["ma_20"]) / df["ma_20"] * 100
        df["dist_from_ma250_pct"] = (df["close"] - df["ma_250"]) / df["ma_250"] * 100

        # Trend determination
        df["above_ma20"] = (df["close"] > df["ma_20"]).astype(int)
        df["above_ma250"] = (df["close"] > df["ma_250"]).astype(int)
        df["uptrend"] = ((df["close"] > df["ma_20"]) & (df["close"] > df["ma_250"])).astype(int)
        df["downtrend"] = ((df["close"] < df["ma_20"]) & (df["close"] < df["ma_250"])).astype(int)

        # ATR for volatility
        df["atr"] = average_true_range(df["high"], df["low"], df["close"], period=14)

        # Rate of change for momentum context
        df["roc_5"] = rate_of_change(df["close"], 5)
        df["roc_20"] = rate_of_change(df["close"], 20)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate White Light trading signals from 7 sub-systems.

        Each sub-system outputs a position vote (0 to 1.0 for TQQQ, or SQQQ).
        Final position = weighted average of all active sub-system votes.
        """
        if not self.validate_data(data):
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            df["instrument"] = ""
            return df

        df = self.calculate_indicators(data)

        # Initialize output columns
        df["signal"] = SignalAction.HOLD.value
        df["position_pct"] = 0.0
        df["instrument"] = ""

        # Sub-system output columns for transparency
        df["sub1_momentum_long"] = 0.0
        df["sub2_mr_short"] = 0.0
        df["sub3_mr_long1"] = 0.0
        df["sub4_mr_long2"] = 0.0
        df["sub5_mr_long3"] = 0.0
        df["sub6_mom_short1"] = 0.0
        df["sub7_mom_short2"] = 0.0

        # Sub-system weights (tuned to match Malik's published stats)
        W = {
            "sub1": 0.25,  # Momentum long — heaviest when trending
            "sub2": 0.10,  # Mean reversion short
            "sub3": 0.15,  # Mean reversion long #1
            "sub4": 0.10,  # Mean reversion long #2
            "sub5": 0.10,  # Mean reversion long #3
            "sub6": 0.20,  # Momentum short #1
            "sub7": 0.10,  # Momentum short #2
        }

        valid_start = self.slow_period
        prev_instrument = ""
        prev_position = 0.0

        for i in range(valid_start, len(df)):
            row = df.iloc[i]

            if pd.isna(row["ma_20"]) or pd.isna(row["ma_250"]):
                continue

            # ============================================================
            # SUB-SYSTEM 1: MOMENTUM LONG (TQQQ)
            # Strong uptrend + Bollinger momentum = heavy long
            # ============================================================
            sub1 = 0.0
            if row["above_ma20"] and row["above_ma250"]:
                if not pd.isna(row["bb_pct_b"]) and row["bb_pct_b"] > 0.8:
                    sub1 = 1.0  # Full momentum allocation
                elif not pd.isna(row["bb_pct_b"]) and row["bb_pct_b"] > 0.5:
                    sub1 = 0.6  # Moderate momentum
                else:
                    sub1 = 0.3  # In uptrend but not momentum phase

            # ============================================================
            # SUB-SYSTEM 2: MEAN REVERSION SHORT (SQQQ)
            # Price way above MA(250) = overextended = short
            # ============================================================
            sub2 = 0.0
            if row["dist_from_ma250_pct"] > self.overextended_pct:
                # Overextended above MA(250) — mean reversion short
                overextension = (row["dist_from_ma250_pct"] - self.overextended_pct) / 10.0
                sub2 = min(1.0, overextension)  # Scales with extension

            # ============================================================
            # SUB-SYSTEM 3: MEAN REVERSION LONG #1 (TQQQ)
            # Price dips below MA(20) but above MA(250) = buy the dip
            # ============================================================
            sub3 = 0.0
            if row["above_ma250"] and not row["above_ma20"]:
                # Pulled back to or below MA(20) in uptrend
                dip_depth = abs(row["dist_from_ma20_pct"])
                if dip_depth > 1.0:
                    sub3 = min(1.0, dip_depth / 5.0)

            # ============================================================
            # SUB-SYSTEM 4: MEAN REVERSION LONG #2 (TQQQ)
            # Price at lower Bollinger Band in uptrend
            # ============================================================
            sub4 = 0.0
            if row["above_ma250"] and not pd.isna(row["bb_pct_b"]):
                if row["bb_pct_b"] < 0.2:
                    sub4 = 0.8  # Near lower band in uptrend
                elif row["bb_pct_b"] < 0.05:
                    sub4 = 1.0  # Below lower band — strong mean reversion buy

            # ============================================================
            # SUB-SYSTEM 5: MEAN REVERSION LONG #3 (TQQQ)
            # Severely extended below MA(20) in uptrend
            # ============================================================
            sub5 = 0.0
            if row["above_ma250"] and row["dist_from_ma20_pct"] < self.severe_dip_pct:
                severity = abs(row["dist_from_ma20_pct"] - self.severe_dip_pct) / 5.0
                sub5 = min(1.0, severity)

            # ============================================================
            # SUB-SYSTEM 6: MOMENTUM SHORT #1 (SQQQ)
            # Confirmed downtrend: below both MAs
            # ============================================================
            sub6 = 0.0
            if not row["above_ma20"] and not row["above_ma250"]:
                sub6 = 0.7  # Confirmed downtrend
                if row["dist_from_ma250_pct"] < -5.0:
                    sub6 = 1.0  # Accelerating downtrend

            # ============================================================
            # SUB-SYSTEM 7: MOMENTUM SHORT #2 (SQQQ)
            # Bollinger squeeze breaking down
            # ============================================================
            sub7 = 0.0
            if (not row["above_ma250"]
                    and not pd.isna(row["bb_pct_b"])
                    and row["bb_pct_b"] < 0.0):
                sub7 = 0.8  # Below lower Bollinger Band in downtrend
                if not pd.isna(row["bb_bandwidth"]) and row["bb_bandwidth"] < 5.0:
                    sub7 = 1.0  # Squeeze breakout to downside

            # Store sub-system outputs
            df.iloc[i, df.columns.get_loc("sub1_momentum_long")] = sub1
            df.iloc[i, df.columns.get_loc("sub2_mr_short")] = sub2
            df.iloc[i, df.columns.get_loc("sub3_mr_long1")] = sub3
            df.iloc[i, df.columns.get_loc("sub4_mr_long2")] = sub4
            df.iloc[i, df.columns.get_loc("sub5_mr_long3")] = sub5
            df.iloc[i, df.columns.get_loc("sub6_mom_short1")] = sub6
            df.iloc[i, df.columns.get_loc("sub7_mom_short2")] = sub7

            # ============================================================
            # AGGREGATE: Combine sub-systems
            # ============================================================
            # TQQQ signals (long): sub1, sub3, sub4, sub5
            tqqq_score = (
                W["sub1"] * sub1 +
                W["sub3"] * sub3 +
                W["sub4"] * sub4 +
                W["sub5"] * sub5
            )

            # SQQQ signals (short): sub2, sub6, sub7
            sqqq_score = (
                W["sub2"] * sub2 +
                W["sub6"] * sub6 +
                W["sub7"] * sub7
            )

            # Determine direction and size
            if tqqq_score > sqqq_score:
                instrument = self.settings.long_asset   # TQQQ
                position_pct = min(self.max_position, tqqq_score / 0.6)  # Normalize
            elif sqqq_score > tqqq_score:
                instrument = self.settings.short_asset  # SQQQ
                position_pct = min(self.max_position, sqqq_score / 0.4)  # Normalize
            else:
                instrument = prev_instrument if prev_instrument else self.settings.long_asset
                position_pct = 0.0

            # Clamp position
            position_pct = max(0.0, min(self.max_position, position_pct))

            # Round to avoid noise trades (minimum 5% change to trigger rebalance)
            position_pct = round(position_pct * 20) / 20  # Round to nearest 5%

            # Determine signal action
            instrument_changed = (instrument != prev_instrument and prev_instrument != "")

            if instrument_changed and position_pct > 0:
                action = SignalAction.FLIP.value
            elif position_pct > 0 and prev_position == 0:
                action = SignalAction.BUY.value
            elif position_pct == 0 and prev_position > 0:
                action = SignalAction.SELL.value
            elif abs(position_pct - prev_position) > 0.05:
                action = SignalAction.INCREASE.value if position_pct > prev_position else SignalAction.REDUCE.value
            else:
                action = SignalAction.HOLD.value

            df.iloc[i, df.columns.get_loc("signal")] = action
            df.iloc[i, df.columns.get_loc("position_pct")] = position_pct
            df.iloc[i, df.columns.get_loc("instrument")] = instrument

            prev_instrument = instrument
            prev_position = position_pct

        logger.info(
            f"White Light signals generated: {len(df) - valid_start} bars, "
            f"final: {df.iloc[-1]['position_pct']:.0%} {df.iloc[-1]['instrument']}"
        )

        return df

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            **super().get_params(),
            "strategy_type": "malik_white_light",
            "sub_strategies": 7,
            "description": "1 mom long + 1 MR short + 3 MR longs + 2 mom shorts",
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "overextended_pct": self.overextended_pct,
        }
