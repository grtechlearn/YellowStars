"""
Excel Export System — Comprehensive trading data export to LocalData/.

Generates Excel workbooks with multiple sheets:
1. Backtest Results — Full equity curve and trade log
2. Strategy Performance — Per-strategy metrics and signals
3. Drawdown Analysis — Peak-to-valley drawdown history
4. P&L Monthly/Yearly — Rows=years, Columns=months + annual total
5. Overall P&L — Year-over-year profit summary
6. Portfolio — Current holdings and allocation

All files saved to LocalData/ folder.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.backtest.engine import BacktestResult
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.core.models import Portfolio, Trade


class ExcelExporter:
    """Export all trading data to well-formatted Excel workbooks in LocalData/."""

    MONTH_NAMES = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    def __init__(self, output_dir: str = "LocalData"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_full_backtest(self, result: BacktestResult) -> str:
        """Export complete backtest results to a multi-sheet Excel workbook.

        Sheets:
        1. Summary — Key performance metrics
        2. Equity_Curve — Daily equity values
        3. Trades — All executed trades
        4. PnL_Monthly — Monthly returns grid (years x months)
        5. PnL_Yearly — Annual returns summary
        6. Drawdowns — Drawdown analysis
        7. Strategy_Signals — Raw signals from strategy
        8. Portfolio — Final portfolio state

        Returns:
            Path to the saved Excel file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{result.strategy_name}_{timestamp}.xlsx"
        filepath = self.output_dir / filename

        # Strip timezone info from all datetime indices (Excel doesn't support tz-aware datetimes)
        if result.equity_curve is not None and hasattr(result.equity_curve.index, 'tz') and result.equity_curve.index.tz is not None:
            result.equity_curve.index = result.equity_curve.index.tz_localize(None)
        if result.benchmark_curve is not None and hasattr(result.benchmark_curve.index, 'tz') and result.benchmark_curve.index.tz is not None:
            result.benchmark_curve.index = result.benchmark_curve.index.tz_localize(None)
        if result.daily_returns is not None and hasattr(result.daily_returns.index, 'tz') and result.daily_returns.index.tz is not None:
            result.daily_returns.index = result.daily_returns.index.tz_localize(None)
        if result.signals_df is not None and hasattr(result.signals_df.index, 'tz') and result.signals_df.index.tz is not None:
            result.signals_df.index = result.signals_df.index.tz_localize(None)
        if result.position_history is not None and hasattr(result.position_history.index, 'tz') and result.position_history.index.tz is not None:
            result.position_history.index = result.position_history.index.tz_localize(None)

        # Calculate metrics
        pm = PerformanceMetrics(
            result.equity_curve, result.benchmark_curve, result.trades
        )
        metrics = pm.compute_all()

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book

            # ---- Sheet 1: Summary ----
            self._write_summary_sheet(writer, workbook, result, metrics)

            # ---- Sheet 2: Equity Curve ----
            self._write_equity_curve_sheet(writer, workbook, result)

            # ---- Sheet 3: Trades ----
            self._write_trades_sheet(writer, workbook, result.trades)

            # ---- Sheet 4: P&L Monthly ----
            self._write_pnl_monthly_sheet(writer, workbook, result.equity_curve)

            # ---- Sheet 5: P&L Yearly ----
            self._write_pnl_yearly_sheet(writer, workbook, result.equity_curve)

            # ---- Sheet 6: Drawdowns ----
            self._write_drawdown_sheet(writer, workbook, result.equity_curve)

            # ---- Sheet 7: Strategy Signals ----
            self._write_signals_sheet(writer, workbook, result.signals_df)

            # ---- Sheet 8: Portfolio (Overall P&L) ----
            self._write_portfolio_sheet(writer, workbook, result, metrics)

        logger.info(f"Excel report exported: {filepath}")
        return str(filepath)

    def export_history_data(self, data: pd.DataFrame, symbol: str) -> str:
        """Export raw market data to Excel in LocalData/."""
        filename = f"history_{symbol.upper()}.xlsx"
        filepath = self.output_dir / filename

        # Strip timezone info (Excel doesn't support tz-aware datetimes)
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            data.to_excel(writer, sheet_name="OHLCV_Data", index=True)

            workbook = writer.book
            worksheet = writer.sheets["OHLCV_Data"]
            worksheet.set_column("A:A", 18)  # Date column
            worksheet.set_column("B:F", 14)  # OHLCV columns

            # Add header format
            header_fmt = workbook.add_format({
                "bold": True, "bg_color": "#FFD700", "border": 1
            })

        logger.info(f"History data exported: {filepath} ({len(data)} rows)")
        return str(filepath)

    def export_order_data(self, trades: list[Trade]) -> str:
        """Export order/trade data to Excel."""
        filename = f"orders_{date.today().isoformat()}.xlsx"
        filepath = self.output_dir / filename

        records = []
        for t in trades:
            records.append({
                "Trade ID": t.trade_id,
                "Date": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "",
                "Symbol": t.asset.symbol if t.asset else "",
                "Side": t.side.value,
                "Quantity": t.quantity,
                "Entry Price": round(t.entry_price, 2),
                "Exit Price": round(t.exit_price, 2),
                "P&L": round(t.net_pnl, 2),
                "P&L %": round(t.pnl_pct, 2),
                "Commission": round(t.commission, 2),
                "Holding Days": t.holding_period_days,
                "Winner": "Yes" if t.is_winner else "No",
            })

        df = pd.DataFrame(records)
        df.to_excel(filepath, sheet_name="Orders", index=False)
        logger.info(f"Order data exported: {filepath}")
        return str(filepath)

    # ============================================================
    # Private Sheet Writers
    # ============================================================

    def _write_summary_sheet(
        self, writer, workbook, result: BacktestResult, metrics: dict
    ):
        """Write the summary metrics sheet."""
        rm = metrics.get("return_metrics", {})
        rk = metrics.get("risk_metrics", {})
        dd = metrics.get("drawdown_metrics", {})
        tm = metrics.get("trade_metrics", {})
        bc = metrics.get("benchmark_comparison", {})

        summary_data = {
            "Metric": [
                "Strategy Name",
                "Period Start", "Period End", "Duration (Years)",
                "",
                "--- RETURNS ---",
                "Initial Capital", "Final Equity", "Total Profit/Loss",
                "Total Return %", "CAGR %",
                "Best Day %", "Worst Day %", "Positive Days %",
                "",
                "--- RISK ---",
                "Ann. Volatility %", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                "",
                "--- DRAWDOWN ---",
                "Max Drawdown %", "Max DD Date", "Current Drawdown %",
                "",
                "--- TRADES ---",
                "Total Trades", "Win Rate %", "Profit Factor",
                "Avg Win $", "Avg Loss $", "W:L Ratio",
                "Avg Holding Days", "Total Commission $",
            ],
            "Value": [
                result.strategy_name,
                str(result.start_date), str(result.end_date), rm.get("years", 0),
                "",
                "",
                f"${rm.get('initial_capital', 0):,.2f}",
                f"${rm.get('final_equity', 0):,.2f}",
                f"${rm.get('profit_loss', 0):,.2f}",
                f"{rm.get('total_return_pct', 0):.2f}%",
                f"{rm.get('cagr_pct', 0):.2f}%",
                f"{rm.get('best_day_pct', 0):.2f}%",
                f"{rm.get('worst_day_pct', 0):.2f}%",
                f"{rm.get('positive_day_ratio', 0):.1f}%",
                "",
                "",
                f"{rk.get('annualized_volatility_pct', 0):.2f}%",
                f"{rk.get('sharpe_ratio', 0):.3f}",
                f"{rk.get('sortino_ratio', 0):.3f}",
                f"{rk.get('calmar_ratio', 0):.3f}",
                "",
                "",
                f"{dd.get('max_drawdown_pct', 0):.2f}%",
                dd.get("max_drawdown_date", "N/A"),
                f"{dd.get('current_drawdown_pct', 0):.2f}%",
                "",
                "",
                tm.get("total_trades", 0),
                f"{tm.get('win_rate_pct', 0):.1f}%",
                f"{tm.get('profit_factor', 0):.2f}",
                f"${tm.get('avg_win', 0):,.2f}",
                f"${tm.get('avg_loss', 0):,.2f}",
                f"{tm.get('avg_win', 0) / max(tm.get('avg_loss', 1), 1):.2f}:1",
                tm.get("avg_holding_days", 0),
                f"${tm.get('total_commission', 0):,.2f}",
            ],
        }

        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name="Summary", index=False)

        # Format
        ws = writer.sheets["Summary"]
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#FFD700", "border": 1})
        section_fmt = workbook.add_format({"bold": True, "bg_color": "#333333", "font_color": "#FFD700"})
        ws.set_column("A:A", 25)
        ws.set_column("B:B", 25)

    def _write_equity_curve_sheet(self, writer, workbook, result: BacktestResult):
        """Write daily equity curve with benchmark comparison."""
        eq_df = pd.DataFrame({
            "Strategy_Equity": result.equity_curve,
        })

        if result.benchmark_curve is not None and not result.benchmark_curve.empty:
            # Align benchmark to equity curve index (benchmark may cover shorter period)
            eq_df["Benchmark_Equity"] = result.benchmark_curve.reindex(eq_df.index)

        eq_df["Daily_Return_%"] = result.daily_returns.values[:len(eq_df)] * 100

        # Running drawdown
        running_max = result.equity_curve.expanding().max()
        eq_df["Drawdown_%"] = ((result.equity_curve - running_max) / running_max * 100).values

        eq_df.index.name = "Date"
        eq_df.to_excel(writer, sheet_name="Equity_Curve")

        ws = writer.sheets["Equity_Curve"]
        ws.set_column("A:A", 18)
        ws.set_column("B:F", 16)

    def _write_trades_sheet(self, writer, workbook, trades: list[Trade]):
        """Write complete trade log."""
        if not trades:
            pd.DataFrame({"Note": ["No trades executed"]}).to_excel(
                writer, sheet_name="Trades", index=False
            )
            return

        records = []
        for i, t in enumerate(trades, 1):
            records.append({
                "#": i,
                "Entry Date": t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "",
                "Exit Date": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "",
                "Symbol": t.asset.symbol if t.asset else "",
                "Side": t.side.value,
                "Quantity": round(t.quantity, 2),
                "Entry Price": round(t.entry_price, 2),
                "Exit Price": round(t.exit_price, 2),
                "Gross P&L": round(t.pnl, 2),
                "Commission": round(t.commission, 2),
                "Net P&L": round(t.net_pnl, 2),
                "P&L %": round(t.pnl_pct, 2),
                "Holding Days": t.holding_period_days,
                "Winner": "Win" if t.is_winner else "Loss",
                "Cumulative P&L": 0,  # Fill below
            })

        df = pd.DataFrame(records)
        df["Cumulative P&L"] = df["Net P&L"].cumsum().round(2)
        df.to_excel(writer, sheet_name="Trades", index=False)

        ws = writer.sheets["Trades"]
        ws.set_column("A:A", 5)
        ws.set_column("B:C", 14)
        ws.set_column("D:D", 10)
        ws.set_column("E:O", 12)

    def _write_pnl_monthly_sheet(self, writer, workbook, equity_curve: pd.Series):
        """Write Monthly P&L grid: Rows=Years, Columns=Months.

        This is the key sheet showing month-by-month returns.
        """
        if len(equity_curve) < 20:
            pd.DataFrame({"Note": ["Insufficient data"]}).to_excel(
                writer, sheet_name="PnL_Monthly", index=False
            )
            return

        # Calculate monthly returns
        monthly = equity_curve.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100

        # Build grid: rows = years, columns = months
        years = sorted(set(monthly_returns.index.year))
        grid = {}

        for year in years:
            year_data = monthly_returns[monthly_returns.index.year == year]
            row = {}
            for dt, ret in year_data.items():
                month_name = self.MONTH_NAMES[dt.month - 1]
                row[month_name] = round(ret, 2)
            # Annual total
            row["Annual %"] = round(sum(row.values()), 2)
            grid[year] = row

        df = pd.DataFrame(grid).T
        df.index.name = "Year"

        # Ensure all 12 months + Annual column exist
        for m in self.MONTH_NAMES:
            if m not in df.columns:
                df[m] = ""
        if "Annual %" not in df.columns:
            df["Annual %"] = ""

        # Reorder columns
        df = df[self.MONTH_NAMES + ["Annual %"]]

        df.to_excel(writer, sheet_name="PnL_Monthly")

        ws = writer.sheets["PnL_Monthly"]
        ws.set_column("A:A", 8)
        ws.set_column("B:N", 10)

        # Conditional formatting for positive/negative
        green_fmt = workbook.add_format({"font_color": "#00AA00", "num_format": "0.00"})
        red_fmt = workbook.add_format({"font_color": "#CC0000", "num_format": "0.00"})

    def _write_pnl_yearly_sheet(self, writer, workbook, equity_curve: pd.Series):
        """Write annual P&L summary with cumulative totals."""
        if len(equity_curve) < 252:
            pd.DataFrame({"Note": ["Insufficient data"]}).to_excel(
                writer, sheet_name="PnL_Yearly", index=False
            )
            return

        # Calculate yearly returns
        yearly = equity_curve.resample("YE").last()
        yearly_start = equity_curve.resample("YS").first()

        records = []
        cumulative_return = 0.0

        for i in range(len(yearly)):
            year = yearly.index[i].year
            start_val = yearly_start.iloc[i] if i < len(yearly_start) else yearly.iloc[i]
            end_val = yearly.iloc[i]
            annual_return = (end_val / start_val - 1) * 100 if start_val > 0 else 0

            cumulative_return += annual_return

            records.append({
                "Year": year,
                "Start Equity": round(start_val, 2),
                "End Equity": round(end_val, 2),
                "Annual Return %": round(annual_return, 2),
                "Annual P&L $": round(end_val - start_val, 2),
                "Cumulative Return %": round(cumulative_return, 2),
            })

        df = pd.DataFrame(records)
        df.to_excel(writer, sheet_name="PnL_Yearly", index=False)

        ws = writer.sheets["PnL_Yearly"]
        ws.set_column("A:A", 8)
        ws.set_column("B:F", 18)

    def _write_drawdown_sheet(self, writer, workbook, equity_curve: pd.Series):
        """Write drawdown analysis sheet."""
        running_max = equity_curve.expanding().max()
        drawdown_pct = (equity_curve - running_max) / running_max * 100

        # Find significant drawdowns
        dd_periods = []
        in_dd = False
        dd_start = None
        dd_trough = None
        dd_min = 0.0

        for idx, dd_val in drawdown_pct.items():
            if dd_val < -3.0 and not in_dd:
                in_dd = True
                dd_start = idx
                dd_min = dd_val
                dd_trough = idx
            elif in_dd:
                if dd_val < dd_min:
                    dd_min = dd_val
                    dd_trough = idx
                if dd_val >= -0.5:
                    dd_periods.append({
                        "Start Date": dd_start.strftime("%Y-%m-%d"),
                        "Trough Date": dd_trough.strftime("%Y-%m-%d"),
                        "Recovery Date": idx.strftime("%Y-%m-%d"),
                        "Depth %": round(dd_min, 2),
                        "Duration (Days)": (idx - dd_start).days,
                        "Peak Equity": round(running_max[dd_start], 2),
                        "Trough Equity": round(equity_curve[dd_trough], 2),
                    })
                    in_dd = False
                    dd_min = 0.0

        if dd_periods:
            df = pd.DataFrame(dd_periods)
            df = df.sort_values("Depth %").reset_index(drop=True)
        else:
            df = pd.DataFrame({"Note": ["No significant drawdowns found"]})

        df.to_excel(writer, sheet_name="Drawdowns", index=False)

        ws = writer.sheets["Drawdowns"]
        ws.set_column("A:C", 14)
        ws.set_column("D:G", 16)

    def _write_signals_sheet(self, writer, workbook, signals_df: pd.DataFrame):
        """Write strategy signals (last 500 rows to keep file size reasonable)."""
        if signals_df.empty:
            pd.DataFrame({"Note": ["No signals"]}).to_excel(
                writer, sheet_name="Strategy_Signals", index=False
            )
            return

        # Select key columns only
        cols_to_export = ["open", "high", "low", "close", "volume", "signal", "position_pct", "instrument"]
        available_cols = [c for c in cols_to_export if c in signals_df.columns]

        # Add indicator columns if present
        indicator_cols = [c for c in signals_df.columns if c.startswith(("ma_", "bb_", "dist_", "sub"))]
        available_cols.extend(indicator_cols[:10])  # Limit to 10 indicators

        export_df = signals_df[available_cols].tail(500)  # Last 500 bars
        export_df.to_excel(writer, sheet_name="Strategy_Signals")

        ws = writer.sheets["Strategy_Signals"]
        ws.set_column("A:A", 18)

    def _write_portfolio_sheet(
        self, writer, workbook, result: BacktestResult, metrics: dict
    ):
        """Write overall portfolio and P&L summary."""
        rm = metrics.get("return_metrics", {})
        bc = metrics.get("benchmark_comparison", {})

        # Overall P&L by year
        yearly_returns = metrics.get("yearly_returns", {})
        if yearly_returns:
            yearly_data = []
            cumulative = rm.get("initial_capital", 100000)
            for year, ret in sorted(yearly_returns.items()):
                year_pnl = cumulative * (ret / 100)
                cumulative += year_pnl
                yearly_data.append({
                    "Year": year,
                    "Return %": round(ret, 2),
                    "P&L $": round(year_pnl, 2),
                    "Equity End of Year $": round(cumulative, 2),
                    "Cumulative Growth %": round((cumulative / rm.get("initial_capital", 100000) - 1) * 100, 2),
                })

            df = pd.DataFrame(yearly_data)
        else:
            df = pd.DataFrame({
                "Year": ["Total"],
                "Return %": [rm.get("total_return_pct", 0)],
                "P&L $": [rm.get("profit_loss", 0)],
                "Equity End of Year $": [rm.get("final_equity", 0)],
            })

        df.to_excel(writer, sheet_name="Portfolio", index=False)

        ws = writer.sheets["Portfolio"]
        ws.set_column("A:A", 8)
        ws.set_column("B:E", 20)
