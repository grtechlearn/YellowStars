"""
Polygon.io data provider implementation.

Supports:
- US Equities (stocks, ETFs including TQQQ/SQQQ)
- Historical data back to 2003+ (depends on ticker)
- Real-time and delayed quotes
- Options and crypto data
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.exceptions import (
    DataProviderError,
    MissingAPIKeyError,
    SymbolNotFoundError,
)
from yellowstars.core.models import Asset, MarketType, TimeFrame
from yellowstars.data.providers.base import BaseDataProvider


class PolygonDataProvider(BaseDataProvider):
    """Polygon.io data provider for US equities, options, and crypto."""

    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: ("minute", 1),
        TimeFrame.MINUTE_5: ("minute", 5),
        TimeFrame.MINUTE_15: ("minute", 15),
        TimeFrame.MINUTE_30: ("minute", 30),
        TimeFrame.HOUR_1: ("hour", 1),
        TimeFrame.HOUR_4: ("hour", 4),
        TimeFrame.DAILY: ("day", 1),
        TimeFrame.WEEKLY: ("week", 1),
        TimeFrame.MONTHLY: ("month", 1),
    }

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client = None
        self._rate_limit_per_min = kwargs.get("rate_limit_per_minute", 5)
        self._last_request_time = 0.0

    def connect(self) -> bool:
        """Connect to Polygon.io API."""
        if not self.api_key:
            raise MissingAPIKeyError(
                "Polygon.io API key required. Set via YS_DATA_PROVIDER__API_KEY "
                "environment variable or in config.yaml"
            )

        try:
            from polygon import RESTClient
            self._client = RESTClient(api_key=self.api_key)
            # Test connection with a simple request
            self._client.get_market_status()
            self._connected = True
            logger.info("Connected to Polygon.io")
            return True
        except ImportError:
            raise DataProviderError(
                "polygon-api-client not installed. Run: pip install polygon-api-client"
            )
        except Exception as e:
            self._connected = False
            raise DataProviderError(f"Failed to connect to Polygon.io: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        if self._rate_limit_per_min <= 0:
            return
        min_interval = 60.0 / self._rate_limit_per_min
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Polygon.io.

        Handles pagination for large date ranges automatically.
        """
        if not self._connected or not self._client:
            raise DataProviderError("Not connected to Polygon.io. Call connect() first.")

        timespan, multiplier = self.TIMEFRAME_MAP.get(timeframe, ("day", 1))

        logger.info(
            f"Fetching {symbol} data from Polygon.io: "
            f"{start_date} to {end_date} ({timeframe.value})"
        )

        all_bars = []
        current_start = start_date
        max_retries = 3

        while current_start <= end_date:
            for attempt in range(max_retries):
                try:
                    self._rate_limit()

                    aggs = self._client.get_aggs(
                        ticker=symbol.upper(),
                        multiplier=multiplier,
                        timespan=timespan,
                        from_=current_start.isoformat(),
                        to=end_date.isoformat(),
                        adjusted=True,
                        sort="asc",
                        limit=50000,
                    )

                    if not aggs:
                        logger.warning(f"No data returned for {symbol} from {current_start}")
                        current_start = end_date + timedelta(days=1)
                        break

                    for bar in aggs:
                        all_bars.append({
                            "timestamp": pd.Timestamp(bar.timestamp, unit="ms"),
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume or 0,
                        })

                    # Move past the last bar we received
                    last_ts = pd.Timestamp(aggs[-1].timestamp, unit="ms").date()
                    if last_ts >= end_date:
                        current_start = end_date + timedelta(days=1)
                    else:
                        current_start = last_ts + timedelta(days=1)

                    break  # Success

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"Polygon API error (attempt {attempt + 1}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise DataProviderError(
                            f"Failed to fetch {symbol} after {max_retries} attempts: {e}"
                        )

        if not all_bars:
            logger.warning(f"No data found for {symbol} in range {start_date} - {end_date}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_bars)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df = self.validate_dataframe(df)

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        if not self._connected or not self._client:
            raise DataProviderError("Not connected to Polygon.io.")

        try:
            self._rate_limit()
            snapshot = self._client.get_snapshot_ticker("stocks", symbol.upper())
            if snapshot and snapshot.day:
                return snapshot.day.close
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search for symbols on Polygon.io."""
        if not self._connected or not self._client:
            raise DataProviderError("Not connected to Polygon.io.")

        try:
            self._rate_limit()
            results = self._client.list_tickers(
                search=query,
                market="stocks" if market_type in (None, MarketType.US_EQUITY) else "crypto",
                active=True,
                limit=20,
            )

            assets = []
            for ticker in results:
                assets.append(Asset(
                    symbol=ticker.ticker,
                    name=getattr(ticker, "name", ticker.ticker),
                    market_type=market_type or MarketType.US_EQUITY,
                    exchange=getattr(ticker, "primary_exchange", ""),
                    currency=getattr(ticker, "currency_name", "USD"),
                ))

            return assets
        except Exception as e:
            logger.error(f"Symbol search error: {e}")
            return []

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.US_EQUITY, MarketType.CRYPTO, MarketType.OPTIONS]
