"""
Report Generator - Creates comprehensive trading reports.

Generates:
- Daily P&L reports
- Monthly/Quarterly/Annual performance summaries
- Equity curve charts
- Drawdown analysis
- Trade logs
- HTML and console output
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.backtest.engine import BacktestResult
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.core.models import Portfolio, Trade


class ReportGenerator:
    """Generates formatted trading reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_backtest_report(
        self,
        result: BacktestResult,
        save_html: bool = True,
        save_json: bool = True,
    ) -> dict:
        """Generate comprehensive backtest performance report.

        Args:
            result: BacktestResult from the backtesting engine.
            save_html: Save as HTML file.
            save_json: Save as JSON file.

        Returns:
            Dictionary with all metrics.
        """
        # Calculate metrics
        metrics_calc = PerformanceMetrics(
            equity_curve=result.equity_curve,
            benchmark_curve=result.benchmark_curve,
            trades=result.trades,
        )
        metrics = metrics_calc.compute_all()
        result.metrics = metrics

        # Print console summary
        summary = metrics_calc.summary_text()
        logger.info(f"\n{summary}")

        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"backtest_{result.strategy_name}_{timestamp}"

        if save_json:
            self._save_json_report(metrics, base_name, result)

        if save_html:
            self._save_html_report(metrics, base_name, result)

        # Save equity curve as CSV
        equity_path = self.output_dir / f"{base_name}_equity.csv"
        result.equity_curve.to_csv(equity_path, header=True)

        # Save trade log
        if result.trades:
            trades_path = self.output_dir / f"{base_name}_trades.csv"
            trades_df = self._trades_to_dataframe(result.trades)
            trades_df.to_csv(trades_path, index=False)

        logger.info(f"Reports saved to {self.output_dir}/{base_name}_*")
        return metrics

    def generate_daily_report(
        self,
        portfolio: Portfolio,
        trades_today: list[Trade],
        date_str: str = "",
    ) -> dict:
        """Generate end-of-day P&L report.

        Args:
            portfolio: Current portfolio state.
            trades_today: Trades executed today.
            date_str: Date string for the report.

        Returns:
            Daily report dictionary.
        """
        if not date_str:
            date_str = date.today().isoformat()

        report = {
            "date": date_str,
            "portfolio": {
                "total_equity": round(portfolio.total_value, 2),
                "cash": round(portfolio.cash, 2),
                "positions_value": round(portfolio.positions_value, 2),
                "total_return_pct": round(portfolio.total_return_pct, 2),
                "positions": [
                    {
                        "symbol": p.asset.symbol if p.asset else "?",
                        "quantity": p.quantity,
                        "avg_entry": round(p.avg_entry_price, 2),
                        "current_price": round(p.current_price, 2),
                        "unrealized_pnl": round(p.unrealized_pnl, 2),
                        "pnl_pct": round(p.pnl_pct, 2),
                    }
                    for p in portfolio.positions
                ],
            },
            "trades_today": [
                {
                    "symbol": t.asset.symbol if t.asset else "?",
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "entry_price": round(t.entry_price, 2),
                    "exit_price": round(t.exit_price, 2),
                    "pnl": round(t.net_pnl, 2),
                    "commission": round(t.commission, 2),
                }
                for t in trades_today
            ],
            "daily_pnl": round(sum(t.net_pnl for t in trades_today), 2),
            "total_commission_today": round(sum(t.commission for t in trades_today), 2),
        }

        # Save daily report
        daily_dir = self.output_dir / "daily"
        daily_dir.mkdir(exist_ok=True)
        report_path = daily_dir / f"daily_{date_str}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def generate_periodic_report(
        self,
        period: str,  # "monthly", "quarterly", "annual"
        equity_curve: pd.Series,
        trades: list[Trade],
        start_date: date,
        end_date: date,
    ) -> dict:
        """Generate periodic (monthly/quarterly/annual) summary report."""
        metrics_calc = PerformanceMetrics(equity_curve=equity_curve, trades=trades)
        metrics = metrics_calc.compute_all()

        report = {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": metrics,
        }

        # Save
        report_path = self.output_dir / f"{period}_{start_date}_{end_date}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _save_json_report(self, metrics: dict, base_name: str, result: BacktestResult) -> None:
        """Save metrics as JSON."""
        report_data = {
            "strategy": result.strategy_name,
            "parameters": result.strategy_params,
            "backtest_settings": result.backtest_settings,
            "period": {
                "start": str(result.start_date),
                "end": str(result.end_date),
            },
            "metrics": metrics,
        }

        # Clean non-serializable values
        path = self.output_dir / f"{base_name}.json"
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _save_html_report(self, metrics: dict, base_name: str, result: BacktestResult) -> None:
        """Save an HTML performance report."""
        rm = metrics.get("return_metrics", {})
        rk = metrics.get("risk_metrics", {})
        dd = metrics.get("drawdown_metrics", {})
        tm = metrics.get("trade_metrics", {})
        bc = metrics.get("benchmark_comparison", {})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YellowStars - Backtest Report: {result.strategy_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0f0f23; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #ffd700; margin-bottom: 10px; }}
        h2 {{ color: #00d4ff; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 15px 0; }}
        .card {{ background: #1a1a2e; border-radius: 8px; padding: 15px; border: 1px solid #333; }}
        .card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        .card .value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #222; }}
        th {{ color: #00d4ff; text-align: left; }}
        td:first-child {{ text-align: left; }}
        .footer {{ margin-top: 30px; text-align: center; color: #555; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>YellowStars Backtest Report</h1>
        <p class="subtitle">Strategy: {result.strategy_name} | {result.start_date} to {result.end_date}</p>

        <h2>Performance Summary</h2>
        <div class="grid">
            <div class="card">
                <div class="label">Total Return</div>
                <div class="value {'positive' if rm.get('total_return_pct', 0) >= 0 else 'negative'}">{rm.get('total_return_pct', 0):.2f}%</div>
            </div>
            <div class="card">
                <div class="label">CAGR</div>
                <div class="value {'positive' if rm.get('cagr_pct', 0) >= 0 else 'negative'}">{rm.get('cagr_pct', 0):.2f}%</div>
            </div>
            <div class="card">
                <div class="label">Final Equity</div>
                <div class="value">${rm.get('final_equity', 0):,.2f}</div>
            </div>
            <div class="card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{rk.get('sharpe_ratio', 0):.3f}</div>
            </div>
            <div class="card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{dd.get('max_drawdown_pct', 0):.2f}%</div>
            </div>
            <div class="card">
                <div class="label">Win Rate</div>
                <div class="value">{tm.get('win_rate_pct', 0):.1f}%</div>
            </div>
            <div class="card">
                <div class="label">Total Trades</div>
                <div class="value">{tm.get('total_trades', 0)}</div>
            </div>
            <div class="card">
                <div class="label">Profit Factor</div>
                <div class="value">{tm.get('profit_factor', 0):.2f}</div>
            </div>
        </div>

        <h2>Risk Metrics</h2>
        <table>
            <tr><td>Annualized Volatility</td><td>{rk.get('annualized_volatility_pct', 0):.2f}%</td></tr>
            <tr><td>Sortino Ratio</td><td>{rk.get('sortino_ratio', 0):.3f}</td></tr>
            <tr><td>Calmar Ratio</td><td>{rk.get('calmar_ratio', 0):.3f}</td></tr>
            <tr><td>Best Day</td><td class="positive">{rm.get('best_day_pct', 0):.2f}%</td></tr>
            <tr><td>Worst Day</td><td class="negative">{rm.get('worst_day_pct', 0):.2f}%</td></tr>
            <tr><td>Positive Days Ratio</td><td>{rm.get('positive_day_ratio', 0):.1f}%</td></tr>
        </table>

        {'<h2>vs Benchmark</h2><table>' +
         f'<tr><td>Strategy CAGR</td><td>{bc.get("strategy_cagr_pct", 0):.2f}%</td></tr>' +
         f'<tr><td>Benchmark CAGR</td><td>{bc.get("benchmark_cagr_pct", 0):.2f}%</td></tr>' +
         f'<tr><td>Outperformance</td><td class="positive">{bc.get("outperformance_pct", 0):.2f}%</td></tr>' +
         f'<tr><td>Alpha</td><td>{bc.get("alpha_annualized", 0):.3f}</td></tr>' +
         f'<tr><td>Beta</td><td>{bc.get("beta", 0):.3f}</td></tr>' +
         '</table>' if bc else ''}

        <div class="footer">
            Generated by YellowStars Trading Platform v0.1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

        path = self.output_dir / f"{base_name}.html"
        with open(path, "w") as f:
            f.write(html)

    @staticmethod
    def _trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
        """Convert list of Trade objects to a DataFrame."""
        records = []
        for t in trades:
            records.append({
                "trade_id": t.trade_id,
                "symbol": t.asset.symbol if t.asset else "",
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
                "tax": t.tax,
                "net_pnl": t.net_pnl,
                "holding_days": t.holding_period_days,
                "is_winner": t.is_winner,
            })
        return pd.DataFrame(records)
