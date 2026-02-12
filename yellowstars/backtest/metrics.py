"""
Performance Metrics Calculator.

Computes comprehensive performance metrics from backtest results:
- Return metrics (CAGR, total return, monthly/yearly returns)
- Risk metrics (max drawdown, Sharpe, Sortino, volatility)
- Trade statistics (win rate, avg win/loss, profit factor)
- Benchmarking comparisons
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceMetrics:
    """Calculate and store performance metrics from backtest results."""

    def __init__(
        self,
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        trades: Optional[list] = None,
        risk_free_rate: float = 0.04,  # ~4% annual risk-free rate
        trading_days_per_year: int = 252,
        sell_hold_curve: Optional[pd.Series] = None,
    ):
        self.equity = equity_curve
        self.benchmark = benchmark_curve
        self.sell_hold = sell_hold_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year

        # Calculate daily returns
        self.daily_returns = equity_curve.pct_change().fillna(0)
        self.benchmark_returns = (
            benchmark_curve.pct_change().fillna(0)
            if benchmark_curve is not None
            else pd.Series(dtype=float)
        )
        self.sell_hold_returns = (
            sell_hold_curve.pct_change().fillna(0)
            if sell_hold_curve is not None and not sell_hold_curve.empty
            else pd.Series(dtype=float)
        )

    def compute_all(self) -> dict:
        """Compute all performance metrics.

        Returns:
            Dictionary with all metrics grouped by category.
        """
        metrics = {}
        metrics["return_metrics"] = self._return_metrics()
        metrics["risk_metrics"] = self._risk_metrics()
        metrics["drawdown_metrics"] = self._drawdown_metrics()
        metrics["trade_metrics"] = self._trade_metrics()
        metrics["monthly_returns"] = self._monthly_returns_table()
        metrics["yearly_returns"] = self._yearly_returns_table()

        if self.benchmark is not None and not self.benchmark.empty:
            metrics["benchmark_comparison"] = self._benchmark_comparison()

        return metrics

    def _return_metrics(self) -> dict:
        """Calculate return-based metrics."""
        if len(self.equity) < 2:
            return {}

        initial = self.equity.iloc[0]
        final = self.equity.iloc[-1]
        total_return = (final / initial - 1) * 100

        # CAGR
        years = len(self.equity) / self.trading_days
        if years > 0 and initial > 0:
            cagr = ((final / initial) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "initial_capital": round(initial, 2),
            "final_equity": round(final, 2),
            "profit_loss": round(final - initial, 2),
            "years": round(years, 1),
            "best_day_pct": round(self.daily_returns.max() * 100, 2),
            "worst_day_pct": round(self.daily_returns.min() * 100, 2),
            "avg_daily_return_pct": round(self.daily_returns.mean() * 100, 4),
            "positive_days": int((self.daily_returns > 0).sum()),
            "negative_days": int((self.daily_returns < 0).sum()),
            "positive_day_ratio": round(
                (self.daily_returns > 0).mean() * 100, 1
            ),
        }

    def _risk_metrics(self) -> dict:
        """Calculate risk-based metrics."""
        if len(self.daily_returns) < 2:
            return {}

        # Annualized volatility
        vol = self.daily_returns.std() * np.sqrt(self.trading_days)

        # Sharpe Ratio
        excess_return = self.daily_returns.mean() * self.trading_days - self.risk_free_rate
        sharpe = excess_return / (vol if vol > 0 else 1)

        # Sortino Ratio (only downside volatility)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0
        sortino = excess_return / (downside_vol if downside_vol > 0 else 1)

        # Calmar Ratio (CAGR / Max Drawdown)
        max_dd = self._max_drawdown_pct()
        cagr = self._return_metrics().get("cagr_pct", 0)
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0

        return {
            "annualized_volatility_pct": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "risk_free_rate": self.risk_free_rate,
            "skewness": round(self.daily_returns.skew(), 3),
            "kurtosis": round(self.daily_returns.kurtosis(), 3),
        }

    def _drawdown_metrics(self) -> dict:
        """Calculate drawdown metrics."""
        if len(self.equity) < 2:
            return {}

        # Running maximum
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max * 100

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find drawdown duration
        peak_idx = self.equity[:max_dd_idx].idxmax() if max_dd_idx is not None else None

        # Recovery (if applicable)
        if max_dd_idx is not None:
            post_dd = self.equity[max_dd_idx:]
            recovery_mask = post_dd >= running_max[max_dd_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                recovery_days = (recovery_idx - max_dd_idx).days
            else:
                recovery_days = None
        else:
            recovery_days = None

        # All drawdowns > 5%
        significant_drawdowns = []
        dd_start = None
        dd_trough = None
        dd_min = 0.0

        for idx, dd_val in drawdown.items():
            if dd_val < -5 and dd_start is None:
                dd_start = idx
                dd_min = dd_val
                dd_trough = idx
            elif dd_start is not None:
                if dd_val < dd_min:
                    dd_min = dd_val
                    dd_trough = idx
                if dd_val >= 0:
                    significant_drawdowns.append({
                        "start": dd_start,
                        "trough": dd_trough,
                        "end": idx,
                        "depth_pct": round(dd_min, 2),
                    })
                    dd_start = None
                    dd_min = 0.0

        return {
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_date": str(max_dd_idx.date()) if max_dd_idx is not None else None,
            "peak_date": str(peak_idx.date()) if peak_idx is not None else None,
            "recovery_days": recovery_days,
            "current_drawdown_pct": round(drawdown.iloc[-1], 2),
            "avg_drawdown_pct": round(drawdown[drawdown < 0].mean(), 2) if (drawdown < 0).any() else 0,
            "significant_drawdowns_count": len(significant_drawdowns),
            "significant_drawdowns": significant_drawdowns[:10],  # Top 10
        }

    def _max_drawdown_pct(self) -> float:
        """Helper: get max drawdown percentage."""
        if len(self.equity) < 2:
            return 0.0
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max * 100
        return drawdown.min()

    def _trade_metrics(self) -> dict:
        """Calculate trade statistics."""
        if not self.trades:
            return {"total_trades": 0}

        pnls = [t.net_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        total_trades = len(self.trades)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0
        profit_factor = sum(winners) / abs(sum(losers)) if losers else float("inf")

        holding_days = [t.holding_period_days for t in self.trades]

        return {
            "total_trades": total_trades,
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate_pct": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(sum(pnls), 2),
            "total_commission": round(sum(t.commission for t in self.trades), 2),
            "avg_holding_days": round(np.mean(holding_days), 1) if holding_days else 0,
            "max_holding_days": max(holding_days) if holding_days else 0,
            "min_holding_days": min(holding_days) if holding_days else 0,
        }

    def _monthly_returns_table(self) -> dict:
        """Generate monthly returns table."""
        if len(self.equity) < 20:
            return {}

        monthly = self.equity.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100

        result = {}
        for dt, ret in monthly_returns.items():
            year = dt.year
            month = dt.month
            if year not in result:
                result[year] = {}
            result[year][month] = round(ret, 2)

        return result

    def _yearly_returns_table(self) -> dict:
        """Generate yearly returns summary."""
        if len(self.equity) < 252:
            return {}

        yearly = self.equity.resample("YE").last()
        yearly_returns = yearly.pct_change().dropna() * 100

        result = {}
        for dt, ret in yearly_returns.items():
            result[dt.year] = round(ret, 2)

        return result

    def _benchmark_comparison(self) -> dict:
        """Compare strategy vs benchmark."""
        if self.benchmark is None or self.benchmark.empty:
            return {}

        bench_initial = self.benchmark.iloc[0]
        bench_final = self.benchmark.iloc[-1]
        bench_total_return = (bench_final / bench_initial - 1) * 100

        years = len(self.benchmark) / self.trading_days
        bench_cagr = (
            ((bench_final / bench_initial) ** (1 / years) - 1) * 100
            if years > 0 and bench_initial > 0
            else 0
        )

        # Benchmark drawdown
        bench_running_max = self.benchmark.expanding().max()
        bench_dd = (self.benchmark - bench_running_max) / bench_running_max * 100
        bench_max_dd = bench_dd.min()

        # Strategy metrics
        strat_metrics = self._return_metrics()
        strat_cagr = strat_metrics.get("cagr_pct", 0)
        strat_total = strat_metrics.get("total_return_pct", 0)

        # Alpha & Beta
        if len(self.daily_returns) == len(self.benchmark_returns):
            covariance = np.cov(self.daily_returns, self.benchmark_returns)
            if covariance.shape == (2, 2) and covariance[1, 1] > 0:
                beta = covariance[0, 1] / covariance[1, 1]
                alpha = (
                    self.daily_returns.mean() * self.trading_days
                    - beta * self.benchmark_returns.mean() * self.trading_days
                ) * 100
            else:
                beta = 0.0
                alpha = 0.0
        else:
            beta = 0.0
            alpha = 0.0

        result = {
            "strategy_total_return_pct": round(strat_total, 2),
            "benchmark_total_return_pct": round(bench_total_return, 2),
            "strategy_cagr_pct": round(strat_cagr, 2),
            "benchmark_cagr_pct": round(bench_cagr, 2),
            "outperformance_pct": round(strat_total - bench_total_return, 2),
            "cagr_difference_pct": round(strat_cagr - bench_cagr, 2),
            "strategy_max_drawdown_pct": round(self._max_drawdown_pct(), 2),
            "benchmark_max_drawdown_pct": round(bench_max_dd, 2),
            "alpha_annualized": round(alpha, 3),
            "beta": round(beta, 3),
        }

        # Sell & Hold metrics (SQQQ buy-and-hold)
        if self.sell_hold is not None and not self.sell_hold.empty:
            sh_initial = self.sell_hold.iloc[0]
            sh_final = self.sell_hold.iloc[-1]
            sh_total = (sh_final / sh_initial - 1) * 100
            sh_years = len(self.sell_hold) / self.trading_days
            sh_cagr = (
                ((sh_final / sh_initial) ** (1 / sh_years) - 1) * 100
                if sh_years > 0 and sh_initial > 0
                else 0
            )

            sh_running_max = self.sell_hold.expanding().max()
            sh_dd = ((self.sell_hold - sh_running_max) / sh_running_max * 100).min()

            sh_vol = self.sell_hold_returns.std() * np.sqrt(self.trading_days)
            sh_sharpe = (
                (self.sell_hold_returns.mean() * self.trading_days - self.risk_free_rate)
                / (sh_vol if sh_vol > 0 else 1)
            )

            result["sell_hold_total_return_pct"] = round(sh_total, 2)
            result["sell_hold_cagr_pct"] = round(sh_cagr, 2)
            result["sell_hold_max_drawdown_pct"] = round(sh_dd, 2)
            result["sell_hold_sharpe"] = round(sh_sharpe, 3)
            result["sell_hold_final_equity"] = round(sh_final, 2)

        return result

    def summary_text(self) -> str:
        """Generate a human-readable performance summary."""
        metrics = self.compute_all()
        rm = metrics.get("return_metrics", {})
        rk = metrics.get("risk_metrics", {})
        dd = metrics.get("drawdown_metrics", {})
        tm = metrics.get("trade_metrics", {})
        bc = metrics.get("benchmark_comparison", {})

        lines = [
            "=" * 60,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            "--- Return Metrics ---",
            f"  Total Return:       {rm.get('total_return_pct', 0):>10.2f}%",
            f"  CAGR:               {rm.get('cagr_pct', 0):>10.2f}%",
            f"  Initial Capital:    ${rm.get('initial_capital', 0):>12,.2f}",
            f"  Final Equity:       ${rm.get('final_equity', 0):>12,.2f}",
            f"  Profit/Loss:        ${rm.get('profit_loss', 0):>12,.2f}",
            f"  Period:             {rm.get('years', 0):>10.1f} years",
            f"  Best Day:           {rm.get('best_day_pct', 0):>10.2f}%",
            f"  Worst Day:          {rm.get('worst_day_pct', 0):>10.2f}%",
            f"  Positive Days:      {rm.get('positive_day_ratio', 0):>10.1f}%",
            "",
            "--- Risk Metrics ---",
            f"  Ann. Volatility:    {rk.get('annualized_volatility_pct', 0):>10.2f}%",
            f"  Sharpe Ratio:       {rk.get('sharpe_ratio', 0):>10.3f}",
            f"  Sortino Ratio:      {rk.get('sortino_ratio', 0):>10.3f}",
            f"  Calmar Ratio:       {rk.get('calmar_ratio', 0):>10.3f}",
            "",
            "--- Drawdown ---",
            f"  Max Drawdown:       {dd.get('max_drawdown_pct', 0):>10.2f}%",
            f"  Max DD Date:        {dd.get('max_drawdown_date', 'N/A'):>10}",
            f"  Recovery Days:      {dd.get('recovery_days', 'N/A'):>10}",
            f"  Current Drawdown:   {dd.get('current_drawdown_pct', 0):>10.2f}%",
            "",
            "--- Trade Statistics ---",
            f"  Total Trades:       {tm.get('total_trades', 0):>10}",
            f"  Win Rate:           {tm.get('win_rate_pct', 0):>10.1f}%",
            f"  Profit Factor:      {tm.get('profit_factor', 0):>10.2f}",
            f"  Avg Win:            ${tm.get('avg_win', 0):>12,.2f}",
            f"  Avg Loss:           ${tm.get('avg_loss', 0):>12,.2f}",
            f"  Total Commission:   ${tm.get('total_commission', 0):>12,.2f}",
        ]

        if bc:
            lines.extend([
                "",
                "--- vs Benchmark ---",
                f"  Strategy CAGR:      {bc.get('strategy_cagr_pct', 0):>10.2f}%",
                f"  Benchmark CAGR:     {bc.get('benchmark_cagr_pct', 0):>10.2f}%",
                f"  Outperformance:     {bc.get('outperformance_pct', 0):>10.2f}%",
                f"  Alpha (ann.):       {bc.get('alpha_annualized', 0):>10.3f}",
                f"  Beta:               {bc.get('beta', 0):>10.3f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
