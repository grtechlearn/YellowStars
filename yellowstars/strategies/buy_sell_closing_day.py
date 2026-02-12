"""
Buy-and-Sell on Closing Day Strategy.

Buys at the open of each trading day and sells at the close of the same day.
This is effectively a "day-trade every day" strategy that captures intraday
returns (close - open) each day.

Since we only have daily OHLCV data, we simulate this by:
  - Each day: buy at open, sell at close
  - Position is flat overnight (no overnight risk)
  - Daily P&L = (close - open) / open * capital
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.base import BaseStrategy


class BuySellClosingDayStrategy(BaseStrategy):
    """Buy at open, sell at close each trading day.

    This captures only intraday returns (open-to-close) and avoids
    overnight gap risk. Acts as a comparison to buy-and-hold which
    captures both intraday and overnight returns.
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        settings = settings or StrategySettings(name="buy_sell_closing_day")
        if settings.name == "malik_white_light":
            settings.name = "buy_sell_closing_day"
        super().__init__(settings, **kwargs)
        self.name = "buy_sell_closing_day"

    @property
    def min_required_bars(self) -> int:
        return 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals: buy-at-open, sell-at-close every day.

        For the multi-strategy backtesting engine, we mark each bar with:
          - signal = BUY (engine interprets as intraday buy-sell)
          - position_pct = 1.0 (fully invested intraday)
          - intraday = True (flag for the engine to use open->close returns)

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex.

        Returns:
            DataFrame with signal, position_pct, instrument, intraday columns.
        """
        if not self.validate_data(data):
            return data

        df = data.copy()
        n = len(df)

        df["signal"] = SignalAction.BUY.value
        df["position_pct"] = 1.0
        df["instrument"] = self.settings.underlying_index
        df["intraday"] = True  # Flag: use open-to-close returns

        logger.info(
            f"BuySellClosingDay: generated signals for {n} bars, "
            f"intraday buy-at-open/sell-at-close on {self.settings.underlying_index}"
        )
        return df

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "type": "buy_sell_closing_day",
            "description": "Buy at open, sell at close each day (intraday only)",
            "underlying": self.settings.underlying_index,
        }
