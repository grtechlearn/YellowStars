"""
Local file-based cache for market data.

Uses Parquet format for efficient storage and fast reads.
Cache structure:
    data/cache/{provider}/{symbol}/{timeframe}_{start}_{end}.parquet
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class LocalCache:
    """Local file system cache for OHLCV data using Parquet format."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, dict] = {}

    def _get_cache_path(
        self,
        symbol: str,
        timeframe: str,
        provider: str = "default",
    ) -> Path:
        """Generate cache file path for a symbol/timeframe combo."""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
        dir_path = self.cache_dir / provider / safe_symbol
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{timeframe}.parquet"

    def get(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        provider: str = "default",
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data for a symbol.

        Returns the cached DataFrame filtered to the requested date range,
        or None if no cache exists.
        """
        cache_path = self._get_cache_path(symbol, timeframe, provider)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)

            if df.empty:
                return None

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Filter to requested range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            mask = (df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))
            filtered = df[mask]

            if filtered.empty:
                return None

            logger.debug(
                f"Cache hit: {symbol} ({timeframe}) - "
                f"{len(filtered)} bars from {filtered.index[0]} to {filtered.index[-1]}"
            )
            return filtered

        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")
            return None

    def put(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        provider: str = "default",
    ) -> bool:
        """Store data in cache, merging with existing cached data."""
        if data.empty:
            return False

        cache_path = self._get_cache_path(symbol, timeframe, provider)

        try:
            # Load existing cache if present
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index)

                # Merge: new data takes priority over existing
                combined = pd.concat([existing, data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
            else:
                combined = data.copy()

            # Save as Parquet
            combined.to_parquet(cache_path, engine="pyarrow")

            logger.debug(
                f"Cached {symbol} ({timeframe}): {len(combined)} total bars, "
                f"{combined.index[0]} to {combined.index[-1]}"
            )
            return True

        except Exception as e:
            logger.error(f"Cache write error for {symbol}: {e}")
            return False

    def has_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        provider: str = "default",
        completeness_threshold: float = 0.9,
    ) -> bool:
        """Check if cache has sufficient data for the requested range.

        Args:
            completeness_threshold: Minimum fraction of expected bars required
                to consider the cache "complete" (0.0 to 1.0).
        """
        df = self.get(symbol, timeframe, start_date, end_date, provider)
        if df is None or df.empty:
            return False

        # Estimate expected number of bars (rough: ~252 trading days/year for daily)
        if timeframe in ("1d", "daily"):
            days = (end_date - start_date).days
            expected_bars = int(days * 252 / 365)  # Approximate trading days
        else:
            expected_bars = len(df)  # Can't easily estimate for intraday

        actual_bars = len(df)
        completeness = actual_bars / max(expected_bars, 1)

        return completeness >= completeness_threshold

    def invalidate(
        self,
        symbol: str,
        timeframe: str,
        provider: str = "default",
    ) -> bool:
        """Delete cached data for a symbol."""
        cache_path = self._get_cache_path(symbol, timeframe, provider)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cache invalidated: {symbol} ({timeframe})")
            return True
        return False

    def get_cache_info(self) -> dict:
        """Get summary of all cached data."""
        info = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "symbols": {},
        }

        for parquet_file in self.cache_dir.rglob("*.parquet"):
            info["total_files"] += 1
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            info["total_size_mb"] += size_mb

            # Parse path to get provider/symbol
            parts = parquet_file.relative_to(self.cache_dir).parts
            if len(parts) >= 2:
                provider = parts[0]
                symbol = parts[1]
                key = f"{provider}/{symbol}"
                if key not in info["symbols"]:
                    info["symbols"][key] = {
                        "files": [],
                        "total_size_mb": 0.0,
                    }
                info["symbols"][key]["files"].append(parquet_file.name)
                info["symbols"][key]["total_size_mb"] += size_mb

        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info

    def clear_all(self) -> int:
        """Delete all cached data. Returns number of files deleted."""
        count = 0
        for parquet_file in self.cache_dir.rglob("*.parquet"):
            parquet_file.unlink()
            count += 1
        logger.info(f"Cache cleared: {count} files deleted")
        return count
