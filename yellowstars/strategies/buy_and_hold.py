"""
Buy-and-Hold Strategy.

The simplest possible strategy: buy at the open on the first bar,
hold forever, and sell only at the very last bar (for accounting).

position_pct = 1.0 for every bar.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Buy on day 1, hold through the entire period.

    This serves as the baseline benchmark for all other strategies.
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        settings = settings or StrategySettings(name="buy_and_hold")
        if settings.name == "malik_white_light":
            settings.name = "buy_and_hold"
        super().__init__(settings, **kwargs)
        self.name = "buy_and_hold"

    @property
    def min_required_bars(self) -> int:
        return 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals: 100% long from bar 0 to the end.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex.

        Returns:
            DataFrame with signal, position_pct, and instrument columns added.
        """
        if not self.validate_data(data):
            return data

        df = data.copy()
        n = len(df)

        # Buy on the first bar, hold everything
        df["signal"] = SignalAction.HOLD.value
        df["position_pct"] = 1.0
        df["instrument"] = self.settings.underlying_index

        # Mark the first bar as BUY
        df.iloc[0, df.columns.get_loc("signal")] = SignalAction.BUY.value

        logger.info(
            f"BuyAndHold: generated signals for {n} bars, "
            f"100% long {self.settings.underlying_index} throughout"
        )
        return df

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "type": "buy_and_hold",
            "description": "Buy on day 1, hold forever",
            "underlying": self.settings.underlying_index,
        }
