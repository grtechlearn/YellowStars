"""Tests for the local data cache."""

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yellowstars.data.cache.local_cache import LocalCache


def _make_sample_data(num_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start="2023-01-01", periods=num_bars, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(num_bars) * 0.5)
    close = np.maximum(close, 10)

    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(1e6, 1e7, num_bars).astype(float),
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestLocalCache:
    """Test suite for the local file cache."""

    def test_cache_creation(self, tmp_path):
        """Test cache directory creation."""
        cache = LocalCache(str(tmp_path / "cache"))
        assert cache.cache_dir.exists()

    def test_put_and_get(self, tmp_path):
        """Test storing and retrieving data."""
        cache = LocalCache(str(tmp_path / "cache"))
        data = _make_sample_data(100)

        # Store
        success = cache.put("AAPL", "1d", data, provider="test")
        assert success

        # Retrieve
        result = cache.get(
            "AAPL", "1d",
            date(2023, 1, 1), date(2023, 12, 31),
            provider="test",
        )
        assert result is not None
        assert len(result) == len(data)

    def test_get_missing_symbol(self, tmp_path):
        """Test retrieving non-existent symbol."""
        cache = LocalCache(str(tmp_path / "cache"))
        result = cache.get("MISSING", "1d", date(2023, 1, 1), date(2023, 12, 31))
        assert result is None

    def test_cache_merge(self, tmp_path):
        """Test merging new data with existing cache."""
        cache = LocalCache(str(tmp_path / "cache"))

        # Store first batch
        data1 = _make_sample_data(50)
        cache.put("AAPL", "1d", data1, provider="test")

        # Store second batch (overlapping dates)
        data2 = _make_sample_data(100)
        cache.put("AAPL", "1d", data2, provider="test")

        # Should have merged without duplicates
        result = cache.get(
            "AAPL", "1d",
            date(2023, 1, 1), date(2024, 12, 31),
            provider="test",
        )
        assert result is not None
        assert len(result) == len(data2)  # Should be max of both

    def test_invalidate(self, tmp_path):
        """Test cache invalidation."""
        cache = LocalCache(str(tmp_path / "cache"))
        data = _make_sample_data(50)

        cache.put("AAPL", "1d", data, provider="test")
        assert cache.invalidate("AAPL", "1d", provider="test")

        result = cache.get(
            "AAPL", "1d",
            date(2023, 1, 1), date(2023, 12, 31),
            provider="test",
        )
        assert result is None

    def test_cache_info(self, tmp_path):
        """Test cache info reporting."""
        cache = LocalCache(str(tmp_path / "cache"))

        data = _make_sample_data(50)
        cache.put("AAPL", "1d", data, provider="test")
        cache.put("GOOG", "1d", data, provider="test")

        info = cache.get_cache_info()
        assert info["total_files"] == 2
        assert info["total_size_mb"] >= 0
        assert len(info["symbols"]) == 2

    def test_clear_all(self, tmp_path):
        """Test clearing all cache."""
        cache = LocalCache(str(tmp_path / "cache"))

        data = _make_sample_data(50)
        cache.put("AAPL", "1d", data, provider="test")
        cache.put("GOOG", "1d", data, provider="test")

        count = cache.clear_all()
        assert count == 2

        info = cache.get_cache_info()
        assert info["total_files"] == 0

    def test_special_characters_in_symbol(self, tmp_path):
        """Test handling symbols with special characters (crypto pairs)."""
        cache = LocalCache(str(tmp_path / "cache"))
        data = _make_sample_data(50)

        # Crypto symbol with /
        success = cache.put("BTC/USDT", "1d", data, provider="test")
        assert success

        result = cache.get(
            "BTC/USDT", "1d",
            date(2023, 1, 1), date(2023, 12, 31),
            provider="test",
        )
        assert result is not None
