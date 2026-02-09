"""
Polygon.io Data Provider — Optimized for Free Tier.

CRITICAL DESIGN PRINCIPLE:
- Free tier = 5 API calls per minute
- Download historical data ONCE per day (full history up to yesterday's close)
- Cache everything locally in LocalData/ as Parquet files
- NEVER make repeated API calls after initial successful download
- Fall back to Yahoo Finance for data that Polygon can't provide on free tier

Data Flow:
1. Check LocalData/ for cached data
2. If cache is fresh (downloaded today), use it — NO API calls
3. If cache is stale or missing, download via REST API with rate limiting
4. Save to LocalData/ for reuse
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
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


class PolygonFreeProvider(BaseDataProvider):
    """Polygon.io REST API provider optimized for free tier limits.

    Key behaviors:
    - Tracks daily download state to prevent repeated API calls
    - Aggressive local caching in LocalData/
    - Rate limiting: max 5 calls/min with 12-second intervals
    - Downloads full history in date-range chunks to stay within limits
    """

    RATE_LIMIT_INTERVAL = 13.0  # seconds between calls (5 calls/min = 12s each, +1s buffer)

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client = None
        self._last_request_time = 0.0
        self._local_data_dir = Path(kwargs.get("local_data_dir", "LocalData"))
        self._local_data_dir.mkdir(parents=True, exist_ok=True)
        self._download_log_path = self._local_data_dir / ".download_log.json"
        self._download_log = self._load_download_log()

    def _load_download_log(self) -> dict:
        """Load the download log tracking what was downloaded and when."""
        if self._download_log_path.exists():
            try:
                with open(self._download_log_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_download_log(self) -> None:
        """Save the download log."""
        with open(self._download_log_path, "w") as f:
            json.dump(self._download_log, f, indent=2, default=str)

    def _was_downloaded_today(self, symbol: str) -> bool:
        """Check if symbol data was already downloaded today."""
        today = date.today().isoformat()
        entry = self._download_log.get(symbol, {})
        return entry.get("last_download_date") == today and entry.get("success", False)

    def _mark_downloaded(self, symbol: str, num_bars: int, data_range: str) -> None:
        """Mark a symbol as successfully downloaded today."""
        self._download_log[symbol] = {
            "last_download_date": date.today().isoformat(),
            "last_download_time": datetime.now().isoformat(),
            "success": True,
            "num_bars": num_bars,
            "data_range": data_range,
        }
        self._save_download_log()

    def connect(self) -> bool:
        """Connect to Polygon.io REST API."""
        if not self.api_key:
            raise MissingAPIKeyError(
                "Polygon.io API key required. Set via YS_DATA_PROVIDER__API_KEY "
                "environment variable or in config.yaml"
            )

        try:
            from polygon import RESTClient
            self._client = RESTClient(api_key=self.api_key)
            self._connected = True
            logger.info("Connected to Polygon.io (Free Tier — 5 calls/min)")
            return True
        except ImportError:
            raise DataProviderError(
                "polygon-api-client not installed. Run: pip install polygon-api-client"
            )
        except Exception as e:
            self._connected = False
            raise DataProviderError(f"Polygon connection failed: {e}")

    def _rate_limit(self):
        """Enforce strict rate limiting for free tier (5 calls/min)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_INTERVAL:
            sleep_time = self.RATE_LIMIT_INTERVAL - elapsed
            logger.info(f"Rate limiting: waiting {sleep_time:.0f}s (free tier: 5 calls/min)")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_local_cache_path(self, symbol: str) -> Path:
        """Get the local Parquet cache path for a symbol."""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
        return self._local_data_dir / f"{safe_symbol}_daily.parquet"

    def _load_local_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data from LocalData/."""
        cache_path = self._get_local_cache_path(symbol)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                logger.info(
                    f"Loaded cached data for {symbol}: {len(df)} bars "
                    f"({df.index[0].date()} to {df.index[-1].date()})"
                )
                return df
            except Exception as e:
                logger.warning(f"Cache read error for {symbol}: {e}")
        return None

    def _save_local_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to LocalData/ as Parquet."""
        cache_path = self._get_local_cache_path(symbol)
        data.to_parquet(cache_path, engine="pyarrow")
        logger.info(f"Saved {len(data)} bars for {symbol} to {cache_path}")

    def _save_local_csv(self, symbol: str, data: pd.DataFrame) -> None:
        """Also save as CSV for human readability."""
        csv_path = self._local_data_dir / f"{symbol.upper()}_daily.csv"
        data.to_csv(csv_path)
        logger.info(f"Saved CSV: {csv_path}")

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Fetch historical data with single-download-per-day policy.

        Strategy:
        1. If downloaded today → return cached data (NO API calls)
        2. If cache exists but not fresh → fetch only missing days
        3. If no cache → full download with aggressive rate limiting
        """
        if not self._connected or not self._client:
            raise DataProviderError("Not connected. Call connect() first.")

        # POLICY: If already downloaded today, return cache only
        if self._was_downloaded_today(symbol):
            cached = self._load_local_cache(symbol)
            if cached is not None and not cached.empty:
                # Filter to requested range
                mask = (cached.index >= pd.Timestamp(start_date)) & \
                       (cached.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
                filtered = cached[mask]
                if not filtered.empty:
                    logger.info(
                        f"Using today's cached data for {symbol}: {len(filtered)} bars "
                        f"(already downloaded today — no API calls)"
                    )
                    return filtered

        # Check existing cache
        cached = self._load_local_cache(symbol)
        fetch_start = start_date

        if cached is not None and not cached.empty:
            cache_end = cached.index[-1].date()
            yesterday = date.today() - timedelta(days=1)

            if cache_end >= yesterday:
                # Cache is up to date
                mask = (cached.index >= pd.Timestamp(start_date)) & \
                       (cached.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
                filtered = cached[mask]
                self._mark_downloaded(symbol, len(filtered),
                                      f"{filtered.index[0].date()} to {filtered.index[-1].date()}")
                return filtered
            else:
                # Fetch only missing data
                fetch_start = cache_end + timedelta(days=1)
                logger.info(f"Updating {symbol} cache from {fetch_start} to {end_date}")

        # Fetch from Polygon API with rate limiting
        logger.info(f"Downloading {symbol} from Polygon.io: {fetch_start} to {end_date}")
        new_data = self._fetch_with_rate_limit(symbol, fetch_start, end_date)

        # Merge with existing cache
        if cached is not None and not cached.empty and not new_data.empty:
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        elif not new_data.empty:
            combined = new_data
        else:
            combined = cached if cached is not None else pd.DataFrame()

        if not combined.empty:
            # Save to local cache
            self._save_local_cache(symbol, combined)
            self._save_local_csv(symbol, combined)
            self._mark_downloaded(symbol, len(combined),
                                  f"{combined.index[0].date()} to {combined.index[-1].date()}")

            # Filter to requested range
            mask = (combined.index >= pd.Timestamp(start_date)) & \
                   (combined.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
            return combined[mask]

        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _fetch_with_rate_limit(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch data from Polygon REST API with strict rate limiting.

        Fetches in 2-year chunks to handle large date ranges.
        """
        all_bars = []
        current_start = start_date
        chunk_days = 730  # ~2 years per request to minimize API calls

        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)

            try:
                self._rate_limit()

                logger.info(f"  API call: {symbol} {current_start} to {chunk_end}")

                aggs = self._client.get_aggs(
                    ticker=symbol.upper(),
                    multiplier=1,
                    timespan="day",
                    from_=current_start.isoformat(),
                    to=chunk_end.isoformat(),
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                )

                if aggs:
                    for bar in aggs:
                        all_bars.append({
                            "timestamp": pd.Timestamp(bar.timestamp, unit="ms"),
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume or 0,
                        })
                    logger.info(f"  Got {len(aggs)} bars for chunk {current_start} to {chunk_end}")
                else:
                    logger.warning(f"  No data for {symbol} {current_start} to {chunk_end}")

            except Exception as e:
                logger.error(f"  Polygon API error for {symbol}: {e}")
                # Don't retry — respect rate limits, move on
                break

            current_start = chunk_end + timedelta(days=1)

        if not all_bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_bars)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df = self.validate_dataframe(df)
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price — uses cache if downloaded today."""
        cached = self._load_local_cache(symbol)
        if cached is not None and not cached.empty:
            return float(cached.iloc[-1]["close"])
        return None

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search — uses one API call."""
        if not self._connected or not self._client:
            return []

        try:
            self._rate_limit()
            results = self._client.list_tickers(search=query, active=True, limit=10)
            return [
                Asset(
                    symbol=t.ticker,
                    name=getattr(t, "name", t.ticker),
                    market_type=market_type or MarketType.US_EQUITY,
                    exchange=getattr(t, "primary_exchange", ""),
                )
                for t in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.US_EQUITY, MarketType.CRYPTO]
