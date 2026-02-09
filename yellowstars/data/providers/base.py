"""
Abstract base class for all data providers.
Every data provider (Polygon, Yahoo, CCXT, etc.) must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

import pandas as pd

from yellowstars.core.models import Asset, MarketType, TimeFrame


class BaseDataProvider(ABC):
    """Abstract data provider interface.

    All data providers return pandas DataFrames with columns:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    indexed by timestamp (DatetimeIndex).
    """

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self._connected = False

    @property
    def name(self) -> str:
        """Provider name for logging and identification."""
        return self.__class__.__name__

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data provider.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD').
            start_date: Start date for historical data.
            end_date: End date for historical data.
            timeframe: Bar timeframe (daily, hourly, etc.).

        Returns:
            DataFrame with columns: open, high, low, close, volume
            DatetimeIndex named 'timestamp'.
        """
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Latest price or None if unavailable.
        """
        pass

    @abstractmethod
    def search_symbols(self, query: str, market_type: Optional[MarketType] = None) -> list[Asset]:
        """Search for tradeable symbols.

        Args:
            query: Search query string.
            market_type: Optional filter by market type.

        Returns:
            List of matching Asset objects.
        """
        pass

    def get_supported_markets(self) -> list[MarketType]:
        """Return list of market types this provider supports."""
        return [MarketType.US_EQUITY]

    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        return self._connected

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a price DataFrame.

        Ensures:
        - Required columns exist
        - No NaN in critical columns
        - Data is sorted by timestamp ascending
        - Index is DatetimeIndex

        Args:
            df: Raw DataFrame from provider.

        Returns:
            Validated and normalized DataFrame.
        """
        required_cols = {"open", "high", "low", "close", "volume"}

        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        # Check required columns
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.index.name = "timestamp"
            else:
                df.index = pd.to_datetime(df.index)
                df.index.name = "timestamp"

        # Sort ascending
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        # Drop rows where all OHLC are NaN
        df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

        # Forward-fill minor gaps (1-2 missing bars)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=2)

        # Fill remaining volume NaN with 0
        df["volume"] = df["volume"].fillna(0)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
