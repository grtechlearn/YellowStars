"""
LocalDataStore - Pre-downloaded local data store for offline backtesting.

All historical market data is stored in LocalData/ as Parquet files.
The backtest engine reads ONLY from LocalData/ — no on-the-fly fetching.

File Layout:
    LocalData/
        .manifest.json         # Tracks download state per symbol
        NDX_daily.parquet      # OHLCV data
        QQQ_daily.parquet
        TQQQ_daily.parquet
        SQQQ_daily.parquet

Usage:
    store = LocalDataStore("LocalData")
    store.ensure_symbols(["NDX", "QQQ"], start, end, provider)
    df = store.load("NDX", start, end)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.data.providers.base import BaseDataProvider


class LocalDataStore:
    """Pre-downloaded local data store for offline backtesting.

    Maintains a manifest file tracking per-symbol metadata:
    last_updated date, row count, date range covered, source provider.
    """

    def __init__(self, local_data_dir: str = "LocalData"):
        self.data_dir = Path(local_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.data_dir / ".manifest.json"
        self.manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        """Load manifest from disk, or return empty dict."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read manifest: {e}")
        return {}

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _update_manifest(self, symbol: str, df: pd.DataFrame, source: str = "yahoo") -> None:
        """Update manifest entry for a symbol after saving data."""
        self.manifest[symbol.upper()] = {
            "last_updated": str(date.today()),
            "first_date": str(df.index[0].date()) if len(df) > 0 else "",
            "last_date": str(df.index[-1].date()) if len(df) > 0 else "",
            "bar_count": len(df),
            "source": source,
            "file": f"{symbol.upper()}_daily.parquet",
        }
        self._save_manifest()

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    def _parquet_path(self, symbol: str) -> Path:
        """Return parquet file path for a symbol, e.g., LocalData/NDX_daily.parquet."""
        return self.data_dir / f"{symbol.upper()}_daily.parquet"

    @staticmethod
    def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
        """Strip timezone info from DataFrame index to avoid parquet read issues."""
        if df is not None and hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def load(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Read-only load from parquet. Raises if data not pre-downloaded.

        Args:
            symbol: Ticker symbol (e.g., "NDX", "QQQ").
            start_date: Start date for filtering.
            end_date: End date for filtering.

        Returns:
            DataFrame with OHLCV data filtered to the requested range.

        Raises:
            FileNotFoundError: If no pre-downloaded data exists for the symbol.
        """
        path = self._parquet_path(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"No pre-downloaded data for {symbol} in {self.data_dir}/. "
                f"Run `python preload_data.py` first."
            )

        df = pd.read_parquet(path)

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index)

        # Filter to requested range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Handle timezone-aware index
        if df.index.tz is not None:
            start_ts = start_ts.tz_localize(df.index.tz)
            end_ts = end_ts.tz_localize(df.index.tz)

        filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]

        logger.info(
            f"Loaded {len(filtered)} bars for {symbol} from LocalData "
            f"({filtered.index[0].date()} to {filtered.index[-1].date()})"
            if not filtered.empty
            else f"No data in range for {symbol}"
        )

        return filtered

    def _load_existing(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load existing parquet data, or return None if not found."""
        path = self._parquet_path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception as e:
            logger.warning(f"Error reading existing data for {symbol}: {e}")
            return None

    def ensure_symbols(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        provider: BaseDataProvider,
    ) -> dict[str, int]:
        """Download or update all requested symbols to LocalData/.

        For each symbol:
        - If data exists and is current (within 1 day of end_date), skip.
        - If data exists but has a gap, fetch only the missing portion.
        - If no data exists, fetch the full range.

        Args:
            symbols: List of ticker symbols to ensure.
            start_date: Earliest date needed.
            end_date: Latest date needed.
            provider: Data provider to fetch from (e.g., YahooDataProvider).

        Returns:
            Dictionary mapping symbol -> bar count.
        """
        results = {}

        for symbol in symbols:
            sym_upper = symbol.upper()
            existing = self._load_existing(sym_upper)

            if existing is not None and not existing.empty:
                cache_end = existing.index[-1].date()

                # If cache is within 1 day of requested end, no fetch needed
                if (end_date - cache_end).days <= 1:
                    logger.info(
                        f"{sym_upper}: data current ({len(existing)} bars, "
                        f"through {cache_end}). Skipping."
                    )
                    results[sym_upper] = len(existing)
                    continue

                # Fetch only the gap
                gap_start = cache_end + timedelta(days=1)
                logger.info(
                    f"{sym_upper}: fetching gap {gap_start} to {end_date}..."
                )
                try:
                    new_data = provider.get_historical_data(
                        sym_upper, gap_start, end_date
                    )
                    if new_data is not None and not new_data.empty:
                        combined = pd.concat([existing, new_data])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                    else:
                        combined = existing
                except Exception as e:
                    logger.warning(f"Gap fetch failed for {sym_upper}: {e}. Using existing data.")
                    combined = existing
            else:
                # Full download
                logger.info(
                    f"{sym_upper}: downloading full history {start_date} to {end_date}..."
                )
                try:
                    combined = provider.get_historical_data(
                        sym_upper, start_date, end_date
                    )
                    if combined is None or combined.empty:
                        logger.warning(f"No data returned for {sym_upper}")
                        results[sym_upper] = 0
                        continue
                except Exception as e:
                    logger.error(f"Download failed for {sym_upper}: {e}")
                    results[sym_upper] = 0
                    continue

            # Strip timezone and save to parquet
            combined = self._strip_tz(combined)
            combined.to_parquet(self._parquet_path(sym_upper))
            self._update_manifest(sym_upper, combined, source="yahoo")
            results[sym_upper] = len(combined)

            logger.info(
                f"{sym_upper}: saved {len(combined)} bars to "
                f"{self._parquet_path(sym_upper)}"
            )

        return results

    def update_all(self, provider: BaseDataProvider, target_end: Optional[date] = None) -> dict:
        """Daily update: fetch only missing days for all symbols in manifest.

        Args:
            provider: Data provider to fetch from.
            target_end: Target end date (defaults to today).

        Returns:
            Dictionary mapping symbol -> update info dict.
        """
        target_end = target_end or date.today()
        results = {}

        if not self.manifest:
            logger.warning("No symbols in manifest. Run `python preload_data.py` first.")
            return results

        for symbol, meta in self.manifest.items():
            last_date_str = meta.get("last_date", "")
            if not last_date_str:
                continue

            last_date = date.fromisoformat(last_date_str)

            # Already current
            if last_date >= target_end - timedelta(days=1):
                results[symbol] = {
                    "new_bars": 0,
                    "total_bars": meta.get("bar_count", 0),
                    "last_date": last_date_str,
                    "status": "current",
                }
                continue

            # Fetch gap
            gap_start = last_date + timedelta(days=1)
            existing = self._load_existing(symbol)

            try:
                new_data = provider.get_historical_data(
                    symbol, gap_start, target_end
                )

                if new_data is not None and not new_data.empty and existing is not None:
                    combined = pd.concat([existing, new_data])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)

                    combined = self._strip_tz(combined)
                    combined.to_parquet(self._parquet_path(symbol))
                    self._update_manifest(symbol, combined)

                    results[symbol] = {
                        "new_bars": len(new_data),
                        "total_bars": len(combined),
                        "last_date": str(combined.index[-1].date()),
                        "status": "updated",
                    }
                    logger.info(
                        f"{symbol}: appended {len(new_data)} new bars "
                        f"(total: {len(combined)})"
                    )
                else:
                    results[symbol] = {
                        "new_bars": 0,
                        "total_bars": meta.get("bar_count", 0),
                        "last_date": last_date_str,
                        "status": "no_new_data",
                    }
            except Exception as e:
                logger.error(f"Update failed for {symbol}: {e}")
                results[symbol] = {
                    "new_bars": 0,
                    "total_bars": meta.get("bar_count", 0),
                    "last_date": last_date_str,
                    "status": f"error: {e}",
                }

        return results

    def get_status(self) -> dict:
        """Return manifest summary for display."""
        return dict(self.manifest)

    def has_symbol(self, symbol: str) -> bool:
        """Check if a symbol has pre-downloaded data."""
        return self._parquet_path(symbol.upper()).exists()
