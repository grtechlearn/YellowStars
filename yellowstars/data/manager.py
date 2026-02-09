"""
DataManager - Central orchestrator for all data operations.

Handles:
- Provider selection and failover
- Cache-first data loading with automatic refresh
- Data validation and quality checks
- Multi-asset, multi-timeframe data retrieval
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.exceptions import (
    DataProviderError,
    InsufficientDataError,
    SymbolNotFoundError,
)
from yellowstars.core.models import Asset, DataProviderConfig, MarketType, TimeFrame
from yellowstars.data.cache.local_cache import LocalCache
from yellowstars.data.providers.base import BaseDataProvider
from yellowstars.data.providers.polygon_provider import PolygonDataProvider
from yellowstars.data.providers.yahoo_provider import YahooDataProvider
from yellowstars.data.providers.crypto_provider import CryptoDataProvider


class DataManager:
    """Central data management layer.

    Provides a unified interface to fetch data from multiple providers
    with automatic caching, failover, and validation.

    Usage:
        dm = DataManager(config)
        dm.initialize()
        df = dm.get_data("AAPL", start_date, end_date)
    """

    PROVIDER_MAP = {
        "polygon": PolygonDataProvider,
        "yahoo": YahooDataProvider,
        "crypto": CryptoDataProvider,
        "ccxt": CryptoDataProvider,
    }

    def __init__(self, config: DataProviderConfig, cache_enabled: bool = True):
        self.config = config
        self.cache_enabled = cache_enabled
        self._primary_provider: Optional[BaseDataProvider] = None
        self._fallback_providers: list[BaseDataProvider] = []
        self._cache = LocalCache(config.cache_directory) if cache_enabled else None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the primary data provider and fallbacks."""
        # Create primary provider
        provider_class = self.PROVIDER_MAP.get(self.config.name.lower())
        if provider_class is None:
            raise DataProviderError(
                f"Unknown data provider: {self.config.name}. "
                f"Supported: {list(self.PROVIDER_MAP.keys())}"
            )

        self._primary_provider = provider_class(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            rate_limit_per_minute=self.config.rate_limit_per_minute,
        )

        try:
            self._primary_provider.connect()
            logger.info(f"Primary data provider connected: {self.config.name}")
        except Exception as e:
            logger.warning(f"Primary provider ({self.config.name}) failed to connect: {e}")

        # Set up Yahoo Finance as fallback for equity data
        if self.config.name.lower() != "yahoo":
            try:
                yahoo = YahooDataProvider()
                yahoo.connect()
                self._fallback_providers.append(yahoo)
                logger.info("Yahoo Finance fallback provider ready")
            except Exception as e:
                logger.warning(f"Yahoo fallback not available: {e}")

        self._initialized = True

    def get_data(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with caching and failover.

        Args:
            symbol: Ticker symbol.
            start_date: Start date.
            end_date: End date (defaults to today).
            timeframe: Data timeframe.
            force_refresh: If True, bypass cache and fetch from API.

        Returns:
            DataFrame with OHLCV data.
        """
        if not self._initialized:
            self.initialize()

        if end_date is None:
            end_date = date.today()

        # 1. Try cache first (unless force refresh)
        if self._cache and self.cache_enabled and not force_refresh:
            cached = self._cache.get(
                symbol, timeframe.value, start_date, end_date,
                provider=self.config.name,
            )
            if cached is not None and not cached.empty:
                # Check if cache covers most of the requested range
                cache_end = cached.index[-1].date()
                # If cache is within 2 days of end_date, use it
                if (end_date - cache_end).days <= 2:
                    logger.info(f"Using cached data for {symbol}: {len(cached)} bars")
                    return cached
                else:
                    # Fetch only missing data
                    gap_start = cache_end + timedelta(days=1)
                    logger.info(
                        f"Cache partial for {symbol}. "
                        f"Fetching gap: {gap_start} to {end_date}"
                    )
                    new_data = self._fetch_from_providers(
                        symbol, gap_start, end_date, timeframe
                    )
                    if not new_data.empty:
                        combined = pd.concat([cached, new_data])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        # Update cache
                        self._cache.put(
                            symbol, timeframe.value, combined,
                            provider=self.config.name,
                        )
                        return combined
                    return cached

        # 2. Fetch from providers
        data = self._fetch_from_providers(symbol, start_date, end_date, timeframe)

        # 3. Cache the result
        if self._cache and self.cache_enabled and not data.empty:
            self._cache.put(
                symbol, timeframe.value, data, provider=self.config.name
            )

        return data

    def _fetch_from_providers(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        """Try fetching data from primary provider, then fallbacks."""
        providers = []
        if self._primary_provider and self._primary_provider.is_connected():
            providers.append(self._primary_provider)
        providers.extend(self._fallback_providers)

        last_error = None
        for provider in providers:
            try:
                data = provider.get_historical_data(
                    symbol, start_date, end_date, timeframe
                )
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Provider {provider.name} failed for {symbol}: {e}. "
                    f"Trying next provider..."
                )

        # All providers failed - try cache as last resort
        if self._cache:
            cached = self._cache.get(
                symbol, timeframe.value, start_date, end_date
            )
            if cached is not None and not cached.empty:
                logger.warning(
                    f"All providers failed for {symbol}. Using stale cache data."
                )
                return cached

        if last_error:
            raise DataProviderError(
                f"All data providers failed for {symbol}: {last_error}"
            )

        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from any available provider."""
        if self._primary_provider and self._primary_provider.is_connected():
            price = self._primary_provider.get_latest_price(symbol)
            if price is not None:
                return price

        for provider in self._fallback_providers:
            price = provider.get_latest_price(symbol)
            if price is not None:
                return price

        return None

    def get_multi_asset_data(
        self,
        symbols: list[str],
        start_date: date,
        end_date: Optional[date] = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Returns:
            Dictionary mapping symbol -> DataFrame.
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_data(
                    symbol, start_date, end_date, timeframe
                )
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

        return results

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search for symbols across all connected providers."""
        results = []

        if self._primary_provider and self._primary_provider.is_connected():
            results.extend(self._primary_provider.search_symbols(query, market_type))

        for provider in self._fallback_providers:
            results.extend(provider.search_symbols(query, market_type))

        # Deduplicate by symbol
        seen = set()
        unique = []
        for asset in results:
            if asset.symbol not in seen:
                seen.add(asset.symbol)
                unique.append(asset)

        return unique

    def get_cache_info(self) -> Optional[dict]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_cache_info()
        return None
