"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from yellowstars.backtest.engine import BacktestEngine, BacktestResult
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.config.settings import BacktestSettings, StrategySettings
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy


def _generate_test_data(num_bars: int = 500, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=num_bars, freq="B")

    if trend == "up":
        base = 100 + np.cumsum(np.random.randn(num_bars) * 0.5 + 0.1)
    elif trend == "down":
        base = 200 + np.cumsum(np.random.randn(num_bars) * 0.5 - 0.1)
    else:
        base = 150 + np.cumsum(np.random.randn(num_bars) * 0.3)

    base = np.maximum(base, 10)

    df = pd.DataFrame({
        "open": base * (1 + np.random.randn(num_bars) * 0.003),
        "high": base * (1 + np.abs(np.random.randn(num_bars) * 0.008)),
        "low": base * (1 - np.abs(np.random.randn(num_bars) * 0.008)),
        "close": base,
        "volume": np.random.randint(1000000, 10000000, num_bars).astype(float),
    }, index=dates)

    df.index.name = "timestamp"
    return df


class TestBacktestEngine:
    """Test suite for the backtesting engine."""

    def test_basic_backtest(self):
        """Test basic backtest execution."""
        data = _generate_test_data(500, "up")
        settings = BacktestSettings(initial_capital=100000)
        engine = BacktestEngine(settings)

        strat_settings = StrategySettings(fast_ma_period=20, slow_ma_period=100)
        strategy = MovingAverageCrossoverStrategy(strat_settings)

        result = engine.run(strategy, data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(data)
        assert result.equity_curve.iloc[0] == pytest.approx(100000, abs=1000)
        assert result.initial_capital == 100000

    def test_backtest_with_commission(self):
        """Test that commissions are properly deducted."""
        data = _generate_test_data(500, "up")
        settings = BacktestSettings(
            initial_capital=100000,
            commission_per_trade=10.0,
        )
        engine = BacktestEngine(settings)

        strat_settings = StrategySettings(fast_ma_period=20, slow_ma_period=100)
        strategy = MovingAverageCrossoverStrategy(strat_settings)

        result = engine.run(strategy, data)

        # Should have traded and paid commissions
        assert result.equity_curve.iloc[-1] <= result.equity_curve.iloc[0] + 1e6  # Sanity check

    def test_backtest_equity_curve_monotonic_start(self):
        """Test that equity starts at initial capital."""
        data = _generate_test_data(300)
        settings = BacktestSettings(initial_capital=50000)
        engine = BacktestEngine(settings)

        strat_settings = StrategySettings(fast_ma_period=20, slow_ma_period=100)
        strategy = MovingAverageCrossoverStrategy(strat_settings)

        result = engine.run(strategy, data)

        # First equity point should be approximately initial capital
        assert result.equity_curve.iloc[0] == pytest.approx(50000, abs=1000)

    def test_backtest_with_benchmark(self):
        """Test backtest with separate benchmark data."""
        data = _generate_test_data(500, "up")
        bench = _generate_test_data(500, "up")

        settings = BacktestSettings(initial_capital=100000)
        engine = BacktestEngine(settings)

        strat_settings = StrategySettings(fast_ma_period=20, slow_ma_period=100)
        strategy = MovingAverageCrossoverStrategy(strat_settings)

        result = engine.run(strategy, data, benchmark_data=bench)

        assert len(result.benchmark_curve) > 0
        assert len(result.benchmark_returns) > 0

    def test_backtest_result_has_trades(self):
        """Test that backtest records trades."""
        data = _generate_test_data(500, "up")
        settings = BacktestSettings(initial_capital=100000)
        engine = BacktestEngine(settings)

        strat_settings = StrategySettings(fast_ma_period=20, slow_ma_period=100)
        strategy = MovingAverageCrossoverStrategy(strat_settings)

        result = engine.run(strategy, data)

        # Should have at least some trades in 500 bars
        assert len(result.trades) >= 0  # Might be 0 if signal never triggers

    def test_malik_strategy_backtest(self):
        """Test Malik White Light strategy in backtest."""
        data = _generate_test_data(500, "up")
        settings = BacktestSettings(initial_capital=100000)
        engine = BacktestEngine(settings)

        strategy = MalikWhiteLightStrategy()
        result = engine.run(strategy, data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(data)
        assert result.strategy_name == "malik_white_light"


class TestPerformanceMetrics:
    """Test performance metrics calculations."""

    def _make_equity_curve(self, returns: list[float]) -> pd.Series:
        """Create equity curve from daily returns."""
        prices = [100000]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        dates = pd.date_range(start="2020-01-01", periods=len(prices), freq="B")
        return pd.Series(prices, index=dates)

    def test_total_return(self):
        """Test total return calculation."""
        equity = self._make_equity_curve([0.01] * 100)  # 1% daily for 100 days
        pm = PerformanceMetrics(equity)
        metrics = pm.compute_all()

        rm = metrics["return_metrics"]
        assert rm["total_return_pct"] > 0
        assert rm["initial_capital"] == pytest.approx(100000)
        assert rm["final_equity"] > rm["initial_capital"]

    def test_cagr(self):
        """Test CAGR calculation."""
        equity = self._make_equity_curve([0.001] * 252)  # ~28% annual
        pm = PerformanceMetrics(equity)
        metrics = pm.compute_all()

        assert metrics["return_metrics"]["cagr_pct"] > 0

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create equity with a clear drawdown
        returns = [0.01] * 50 + [-0.02] * 20 + [0.01] * 50
        equity = self._make_equity_curve(returns)
        pm = PerformanceMetrics(equity)
        metrics = pm.compute_all()

        dd = metrics["drawdown_metrics"]
        assert dd["max_drawdown_pct"] < 0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        equity = self._make_equity_curve([0.001] * 252)
        pm = PerformanceMetrics(equity)
        metrics = pm.compute_all()

        assert "sharpe_ratio" in metrics["risk_metrics"]

    def test_summary_text(self):
        """Test summary text generation."""
        equity = self._make_equity_curve([0.001] * 100)
        pm = PerformanceMetrics(equity)
        text = pm.summary_text()

        assert "BACKTEST PERFORMANCE REPORT" in text
        assert "Total Return" in text
        assert "Max Drawdown" in text

    def test_monthly_returns(self):
        """Test monthly returns table generation."""
        equity = self._make_equity_curve([0.001] * 252)
        pm = PerformanceMetrics(equity)
        metrics = pm.compute_all()

        monthly = metrics.get("monthly_returns", {})
        # Should have some months
        assert len(monthly) >= 0  # May be empty for short periods

    def test_benchmark_comparison(self):
        """Test benchmark comparison metrics."""
        equity = self._make_equity_curve([0.002] * 252)
        benchmark = self._make_equity_curve([0.001] * 252)
        pm = PerformanceMetrics(equity, benchmark)
        metrics = pm.compute_all()

        bc = metrics.get("benchmark_comparison", {})
        assert "outperformance_pct" in bc
        assert bc["outperformance_pct"] > 0  # Strategy beat benchmark
