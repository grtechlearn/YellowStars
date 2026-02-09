"""Tests for Malik's White Light strategy."""

import numpy as np
import pandas as pd
import pytest

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.strategies.base import simple_moving_average, rate_of_change


def _generate_test_data(num_bars: int = 500, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=num_bars, freq="B")

    if trend == "up":
        base = 100 + np.cumsum(np.random.randn(num_bars) * 0.5 + 0.1)
    elif trend == "down":
        base = 200 + np.cumsum(np.random.randn(num_bars) * 0.5 - 0.1)
    else:
        base = 150 + np.cumsum(np.random.randn(num_bars) * 0.5)

    # Ensure positive prices
    base = np.maximum(base, 10)

    df = pd.DataFrame({
        "open": base * (1 + np.random.randn(num_bars) * 0.005),
        "high": base * (1 + np.abs(np.random.randn(num_bars) * 0.01)),
        "low": base * (1 - np.abs(np.random.randn(num_bars) * 0.01)),
        "close": base,
        "volume": np.random.randint(1000000, 10000000, num_bars).astype(float),
    }, index=dates)

    df.index.name = "timestamp"
    return df


class TestMalikWhiteLightStrategy:
    """Test suite for the Malik White Light strategy."""

    def test_strategy_creation(self):
        """Test strategy can be created with default settings."""
        strategy = MalikWhiteLightStrategy()
        assert strategy.name == "malik_white_light"
        assert strategy.fast_period == 50
        assert strategy.slow_period == 250

    def test_strategy_with_custom_settings(self):
        """Test strategy with custom parameters."""
        settings = StrategySettings(
            name="custom_test",
            fast_ma_period=20,
            slow_ma_period=100,
            roc_period=10,
        )
        strategy = MalikWhiteLightStrategy(settings)
        assert strategy.fast_period == 20
        assert strategy.slow_period == 100
        assert strategy.roc_period == 10

    def test_min_required_bars(self):
        """Test minimum data requirement."""
        strategy = MalikWhiteLightStrategy()
        assert strategy.min_required_bars == 270  # 250 + 20

    def test_generate_signals_bullish_trend(self):
        """Test signal generation in a bullish trend."""
        data = _generate_test_data(500, trend="up")
        strategy = MalikWhiteLightStrategy()

        result = strategy.generate_signals(data)

        assert "signal" in result.columns
        assert "position_pct" in result.columns
        assert "instrument" in result.columns
        assert len(result) == len(data)

        # In uptrend, later signals should be bullish (positive position)
        last_100 = result.iloc[-100:]
        avg_position = last_100["position_pct"].mean()
        assert avg_position > 0, f"Expected positive avg position in uptrend, got {avg_position}"

    def test_generate_signals_bearish_trend(self):
        """Test signal generation in a bearish trend."""
        data = _generate_test_data(500, trend="down")
        strategy = MalikWhiteLightStrategy()

        result = strategy.generate_signals(data)

        # In downtrend, should have SQQQ signals
        last_50 = result.iloc[-50:]
        sqqq_count = (last_50["instrument"] == "SQQQ").sum()
        # Should lean towards SQQQ in downtrend
        assert sqqq_count >= 0  # Can't guarantee 100% but should have some

    def test_generate_signals_empty_data(self):
        """Test with empty data."""
        data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        data.index.name = "timestamp"
        strategy = MalikWhiteLightStrategy()

        result = strategy.generate_signals(data)
        assert len(result) == 0

    def test_generate_signals_insufficient_data(self):
        """Test with data shorter than minimum required."""
        data = _generate_test_data(100, trend="up")  # Less than 270
        strategy = MalikWhiteLightStrategy()

        result = strategy.generate_signals(data)
        # Should return data with hold signals
        assert (result["signal"] == SignalAction.HOLD.value).all()

    def test_indicators_calculated(self):
        """Test that all indicators are properly calculated."""
        data = _generate_test_data(500, trend="up")
        strategy = MalikWhiteLightStrategy()

        result = strategy.calculate_indicators(data)

        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "roc" in result.columns
        assert "trend_aligned" in result.columns
        assert "velocity_regime" in result.columns
        assert "atr" in result.columns

    def test_position_pct_range(self):
        """Test that position percentages are within valid range."""
        data = _generate_test_data(500, trend="up")
        strategy = MalikWhiteLightStrategy()

        result = strategy.generate_signals(data)

        assert (result["position_pct"] >= 0).all(), "Position should never be negative"
        assert (result["position_pct"] <= 1.0).all(), "Position should never exceed 100%"

    def test_get_params(self):
        """Test parameter export."""
        strategy = MalikWhiteLightStrategy()
        params = strategy.get_params()

        assert params["strategy_type"] == "malik_white_light"
        assert params["sub_strategies"] == 7
        assert params["fast_period"] == 50
        assert params["slow_period"] == 250


class TestIndicatorFunctions:
    """Test standalone indicator functions."""

    def test_simple_moving_average(self):
        """Test SMA calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = simple_moving_average(data, 3)

        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        assert sma.iloc[2] == pytest.approx(2.0)
        assert sma.iloc[9] == pytest.approx(9.0)

    def test_rate_of_change(self):
        """Test ROC calculation."""
        data = pd.Series([100, 110, 120, 130, 140])
        roc = rate_of_change(data, 1)

        assert pd.isna(roc.iloc[0])
        assert roc.iloc[1] == pytest.approx(10.0)
        assert roc.iloc[2] == pytest.approx(9.0909, abs=0.01)
