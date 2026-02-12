"""
Multi-Strategy Excel Report Generator.

Creates a consolidated Excel workbook with:
1. Per-strategy results sheets (one per strategy)
2. Consolidated order history sheet
3. Charts sheet with:
   - Combined equity curve for all strategies
   - Performance comparison bar chart
   - Drawdown analysis per strategy

Also generates individual CSV files for each strategy's order history.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.backtest.multi_strategy_engine import (
    MultiStrategyResult,
    StrategyResult,
    OrderRecord,
)
from yellowstars.backtest.metrics import PerformanceMetrics


class MultiStrategyReport:
    """Generate CSV and Excel reports for multi-strategy backtests."""

    def __init__(self, output_dir: str = "LocalData"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # CSV Export - Per-Strategy Order History
    # ==================================================================

    def generate_csv_reports(self, result: MultiStrategyResult) -> list[str]:
        """Generate a separate CSV file for each strategy's order history.

        Args:
            result: MultiStrategyResult from the engine.

        Returns:
            List of paths to generated CSV files.
        """
        csv_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for strat_result in result.strategies:
            filename = f"orders_{strat_result.strategy_name}_{timestamp}.csv"
            filepath = self.output_dir / filename

            # Convert orders to DataFrame
            order_rows = []
            for o in strat_result.orders:
                order_rows.append({
                    "Order_ID": o.order_id,
                    "Date": o.date.strftime("%Y-%m-%d") if o.date else "",
                    "Strategy": o.strategy,
                    "Symbol": o.symbol,
                    "Side": o.side,
                    "Quantity": round(o.quantity, 6),
                    "Price": round(o.price, 2),
                    "Value": round(o.value, 2),
                    "Commission": round(o.commission, 2),
                    "Slippage": round(o.slippage, 2),
                    "Net_Value": round(o.net_value, 2),
                    "Portfolio_Value": round(o.portfolio_value, 2),
                    "Cash_After": round(o.cash_after, 2),
                    "Reason": o.reason,
                })

            if order_rows:
                df = pd.DataFrame(order_rows)
            else:
                df = pd.DataFrame(columns=[
                    "Order_ID", "Date", "Strategy", "Symbol", "Side",
                    "Quantity", "Price", "Value", "Commission", "Slippage",
                    "Net_Value", "Portfolio_Value", "Cash_After", "Reason",
                ])

            df.to_csv(filepath, index=False)
            csv_paths.append(str(filepath))
            logger.info(
                f"CSV order history: {filepath} "
                f"({len(order_rows)} orders)"
            )

        return csv_paths

    # ==================================================================
    # Excel Report - Consolidated Workbook
    # ==================================================================

    def generate_excel_report(self, result: MultiStrategyResult) -> str:
        """Generate consolidated Excel workbook.

        Sheets:
        1. Summary         - Side-by-side comparison of all strategies
        2. {StrategyName}  - One sheet per strategy with detailed results
        3. AllOrders        - Combined order history from all strategies
        4. Charts           - Equity curves, drawdowns, performance comparison

        Args:
            result: MultiStrategyResult from the engine.

        Returns:
            Path to generated Excel file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_strategy_report_{result.symbol}_{timestamp}.xlsx"
        filepath = self.output_dir / filename

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            wb = writer.book

            # Sheet 1: Summary comparison
            self._write_summary(writer, wb, result)

            # Sheet 2+: Individual strategy sheets
            for strat_result in result.strategies:
                self._write_strategy_sheet(writer, wb, strat_result)

            # AllOrders sheet
            self._write_all_orders(writer, wb, result)

            # Charts sheet
            self._write_charts(writer, wb, result)

        logger.info(f"Multi-strategy Excel report: {filepath}")
        return str(filepath)

    # ==================================================================
    # Sheet: Summary
    # ==================================================================

    def _write_summary(self, writer, wb, result: MultiStrategyResult):
        """Summary sheet: side-by-side performance comparison."""
        rows = []

        for sr in result.strategies:
            rm = sr.metrics.get("return_metrics", {})
            rk = sr.metrics.get("risk_metrics", {})
            dd = sr.metrics.get("drawdown_metrics", {})
            tm = sr.metrics.get("trade_metrics", {})

            rows.append({
                "Strategy": sr.strategy_name,
                "Initial Capital ($)": rm.get("initial_capital", sr.initial_capital),
                "Final Equity ($)": rm.get("final_equity", 0),
                "Total Return (%)": rm.get("total_return_pct", 0),
                "CAGR (%)": rm.get("cagr_pct", 0),
                "Ann. Volatility (%)": rk.get("annualized_volatility_pct", 0),
                "Sharpe Ratio": rk.get("sharpe_ratio", 0),
                "Sortino Ratio": rk.get("sortino_ratio", 0),
                "Calmar Ratio": rk.get("calmar_ratio", 0),
                "Max Drawdown (%)": dd.get("max_drawdown_pct", 0),
                "Max DD Date": dd.get("max_drawdown_date", ""),
                "Recovery Days": dd.get("recovery_days", "N/A"),
                "Total Trades": tm.get("total_trades", 0),
                "Win Rate (%)": tm.get("win_rate_pct", 0),
                "Profit Factor": tm.get("profit_factor", 0),
                "Best Day (%)": rm.get("best_day_pct", 0),
                "Worst Day (%)": rm.get("worst_day_pct", 0),
                "Positive Days (%)": rm.get("positive_day_ratio", 0),
                "Total Orders": len(sr.orders),
                "Period (Years)": rm.get("years", 0),
            })

        summary_df = pd.DataFrame(rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Format
        ws = writer.sheets["Summary"]
        ws.set_column("A:A", 25)  # Strategy name
        ws.set_column("B:T", 18)  # Metric columns

        # Header format
        header_fmt = wb.add_format({
            "bold": True,
            "bg_color": "#4472C4",
            "font_color": "white",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        })
        for col_num, col_name in enumerate(summary_df.columns):
            ws.write(0, col_num, col_name, header_fmt)

        # Number formats
        pct_fmt = wb.add_format({"num_format": "0.00"})
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        ratio_fmt = wb.add_format({"num_format": "0.000"})

        for row_idx in range(len(rows)):
            r = row_idx + 1
            ws.write(r, 1, rows[row_idx]["Initial Capital ($)"], money_fmt)
            ws.write(r, 2, rows[row_idx]["Final Equity ($)"], money_fmt)
            ws.write(r, 3, rows[row_idx]["Total Return (%)"], pct_fmt)
            ws.write(r, 4, rows[row_idx]["CAGR (%)"], pct_fmt)
            ws.write(r, 5, rows[row_idx]["Ann. Volatility (%)"], pct_fmt)
            ws.write(r, 6, rows[row_idx]["Sharpe Ratio"], ratio_fmt)
            ws.write(r, 7, rows[row_idx]["Sortino Ratio"], ratio_fmt)
            ws.write(r, 8, rows[row_idx]["Calmar Ratio"], ratio_fmt)
            ws.write(r, 9, rows[row_idx]["Max Drawdown (%)"], pct_fmt)

        # Add title row above
        ws.set_row(0, 35)

    # ==================================================================
    # Sheet: Per-Strategy Details
    # ==================================================================

    def _write_strategy_sheet(self, writer, wb, sr: StrategyResult):
        """Write a detailed sheet for one strategy."""
        sheet_name = sr.strategy_name[:31]  # Excel sheet name limit

        # --- Section 1: Metrics Summary (rows 0-14) ---
        rm = sr.metrics.get("return_metrics", {})
        rk = sr.metrics.get("risk_metrics", {})
        dd = sr.metrics.get("drawdown_metrics", {})
        tm = sr.metrics.get("trade_metrics", {})

        summary_rows = [
            {"Metric": "Strategy Name", "Value": sr.strategy_name},
            {"Metric": "Period", "Value": f"{sr.start_date} to {sr.end_date}"},
            {"Metric": "Initial Capital", "Value": f"${sr.initial_capital:,.2f}"},
            {"Metric": "Final Equity", "Value": f"${rm.get('final_equity', 0):,.2f}"},
            {"Metric": "Total Return", "Value": f"{rm.get('total_return_pct', 0):.2f}%"},
            {"Metric": "CAGR", "Value": f"{rm.get('cagr_pct', 0):.2f}%"},
            {"Metric": "Sharpe Ratio", "Value": f"{rk.get('sharpe_ratio', 0):.3f}"},
            {"Metric": "Sortino Ratio", "Value": f"{rk.get('sortino_ratio', 0):.3f}"},
            {"Metric": "Max Drawdown", "Value": f"{dd.get('max_drawdown_pct', 0):.2f}%"},
            {"Metric": "Total Trades", "Value": str(tm.get('total_trades', 0))},
            {"Metric": "Win Rate", "Value": f"{tm.get('win_rate_pct', 0):.1f}%"},
            {"Metric": "Profit Factor", "Value": f"{tm.get('profit_factor', 0):.2f}"},
            {"Metric": "Total Orders", "Value": str(len(sr.orders))},
            {"Metric": "", "Value": ""},
        ]

        metrics_df = pd.DataFrame(summary_rows)
        metrics_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

        # --- Section 2: Order History (starting at row 16) ---
        order_rows = []
        for o in sr.orders:
            order_rows.append({
                "Order_ID": o.order_id,
                "Date": o.date.strftime("%Y-%m-%d") if o.date else "",
                "Side": o.side,
                "Symbol": o.symbol,
                "Quantity": round(o.quantity, 4),
                "Price": round(o.price, 2),
                "Value": round(o.value, 2),
                "Commission": round(o.commission, 2),
                "Slippage": round(o.slippage, 2),
                "Net_Value": round(o.net_value, 2),
                "Cash_After": round(o.cash_after, 2),
                "Reason": o.reason,
            })

        if order_rows:
            orders_df = pd.DataFrame(order_rows)
        else:
            orders_df = pd.DataFrame(columns=[
                "Order_ID", "Date", "Side", "Symbol", "Quantity",
                "Price", "Value", "Commission", "Slippage",
                "Net_Value", "Cash_After", "Reason",
            ])

        start_row = len(summary_rows) + 2
        # Write "Order History" header
        ws = None
        orders_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)

        # Format
        ws = writer.sheets[sheet_name]
        ws.set_column("A:A", 20)  # Metric / Order_ID
        ws.set_column("B:B", 20)  # Value / Date
        ws.set_column("C:L", 15)  # Other columns

        # Bold header for metrics section
        bold_fmt = wb.add_format({"bold": True, "bg_color": "#D9E2F3"})
        ws.write(0, 0, "Metric", bold_fmt)
        ws.write(0, 1, "Value", bold_fmt)

        # Write order history header label
        section_fmt = wb.add_format({
            "bold": True, "font_size": 12,
            "bg_color": "#4472C4", "font_color": "white",
        })
        ws.write(start_row - 1, 0, "ORDER HISTORY", section_fmt)

    # ==================================================================
    # Sheet: AllOrders (combined)
    # ==================================================================

    def _write_all_orders(self, writer, wb, result: MultiStrategyResult):
        """Combined order history from all strategies."""
        all_orders = []
        for sr in result.strategies:
            for o in sr.orders:
                all_orders.append({
                    "Order_ID": o.order_id,
                    "Date": o.date.strftime("%Y-%m-%d") if o.date else "",
                    "Strategy": o.strategy,
                    "Symbol": o.symbol,
                    "Side": o.side,
                    "Quantity": round(o.quantity, 6),
                    "Price": round(o.price, 2),
                    "Value": round(o.value, 2),
                    "Commission": round(o.commission, 2),
                    "Slippage": round(o.slippage, 2),
                    "Net_Value": round(o.net_value, 2),
                    "Portfolio_Value": round(o.portfolio_value, 2),
                    "Cash_After": round(o.cash_after, 2),
                    "Reason": o.reason,
                })

        if all_orders:
            df = pd.DataFrame(all_orders)
        else:
            df = pd.DataFrame(columns=[
                "Order_ID", "Date", "Strategy", "Symbol", "Side",
                "Quantity", "Price", "Value", "Commission", "Slippage",
                "Net_Value", "Portfolio_Value", "Cash_After", "Reason",
            ])

        df.to_excel(writer, sheet_name="AllOrders", index=False)

        # Format
        ws = writer.sheets["AllOrders"]
        ws.set_column("A:A", 10)   # Order_ID
        ws.set_column("B:B", 14)   # Date
        ws.set_column("C:C", 25)   # Strategy
        ws.set_column("D:D", 10)   # Symbol
        ws.set_column("E:E", 8)    # Side
        ws.set_column("F:N", 16)   # Numeric columns

        # Header format
        header_fmt = wb.add_format({
            "bold": True,
            "bg_color": "#4472C4",
            "font_color": "white",
            "border": 1,
        })
        for col_num, col_name in enumerate(df.columns):
            ws.write(0, col_num, col_name, header_fmt)

        # Conditional formatting for Side column (BUY = green, SELL = red)
        num_rows = len(all_orders)
        if num_rows > 0:
            green_fmt = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
            red_fmt = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
            ws.conditional_format(f"E2:E{num_rows + 1}", {
                "type": "text",
                "criteria": "containing",
                "value": "BUY",
                "format": green_fmt,
            })
            ws.conditional_format(f"E2:E{num_rows + 1}", {
                "type": "text",
                "criteria": "containing",
                "value": "SELL",
                "format": red_fmt,
            })

    # ==================================================================
    # Sheet: Charts
    # ==================================================================

    def _write_charts(self, writer, wb, result: MultiStrategyResult):
        """Charts sheet with equity curves, drawdowns, and performance comparison."""
        sheet_name = "Charts"

        # Prepare data for charts
        # Build a combined DataFrame of equity curves
        eq_data = {}
        dd_data = {}

        for sr in result.strategies:
            eq = sr.equity_curve
            if eq is not None and not eq.empty:
                # Strip timezone
                if hasattr(eq.index, "tz") and eq.index.tz is not None:
                    eq.index = eq.index.tz_localize(None)
                eq_data[sr.strategy_name] = eq

                # Drawdown
                running_max = eq.expanding().max()
                drawdown = (eq - running_max) / running_max * 100
                dd_data[sr.strategy_name] = drawdown

        # Add benchmark
        bench = result.benchmark_curve
        if bench is not None and not bench.empty:
            if hasattr(bench.index, "tz") and bench.index.tz is not None:
                bench.index = bench.index.tz_localize(None)
            eq_data[f"{result.symbol} Buy&Hold (Benchmark)"] = bench

            running_max = bench.expanding().max()
            dd_data[f"{result.symbol} Buy&Hold (Benchmark)"] = (
                (bench - running_max) / running_max * 100
            )

        # Write equity curve data
        eq_df = pd.DataFrame(eq_data)
        eq_df.index.name = "Date"
        eq_df.to_excel(writer, sheet_name=sheet_name)

        ws = writer.sheets[sheet_name]
        ws.set_column("A:A", 14)
        ws.set_column("B:Z", 18)

        data_rows = len(eq_df)
        num_series = len(eq_data)

        # ---- Chart 1: Combined Equity Curve ----
        chart1 = wb.add_chart({"type": "line"})
        colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#5B9BD5", "#A5A5A5"]

        for idx, col_name in enumerate(eq_df.columns):
            col_idx = idx + 1  # Column B, C, D, ...
            color = colors[idx % len(colors)]
            chart1.add_series({
                "name": [sheet_name, 0, col_idx],
                "categories": [sheet_name, 1, 0, data_rows, 0],
                "values": [sheet_name, 1, col_idx, data_rows, col_idx],
                "line": {"color": color, "width": 1.5},
            })

        chart1.set_title({"name": f"Equity Curves - {result.symbol}"})
        chart1.set_x_axis({"name": "Date", "date_axis": True})
        chart1.set_y_axis({"name": "Portfolio Value ($)", "log_base": 10})
        chart1.set_size({"width": 1000, "height": 520})
        chart1.set_legend({"position": "bottom"})

        ws.insert_chart("A3", chart1)

        # ---- Drawdown data (write below equity data) ----
        dd_start_row = data_rows + 4
        dd_df = pd.DataFrame(dd_data)
        dd_df.index.name = "Date"
        dd_df.to_excel(writer, sheet_name=sheet_name, startrow=dd_start_row)

        # ---- Chart 2: Drawdown Analysis ----
        chart2 = wb.add_chart({"type": "area"})
        dd_rows = len(dd_df)

        for idx, col_name in enumerate(dd_df.columns):
            col_idx = idx + 1
            color = colors[idx % len(colors)]
            chart2.add_series({
                "name": [sheet_name, dd_start_row, col_idx],
                "categories": [sheet_name, dd_start_row + 1, 0, dd_start_row + dd_rows, 0],
                "values": [sheet_name, dd_start_row + 1, col_idx, dd_start_row + dd_rows, col_idx],
                "fill": {"color": color, "transparency": 50},
                "line": {"color": color, "width": 0.5},
            })

        chart2.set_title({"name": f"Drawdown Analysis - {result.symbol}"})
        chart2.set_x_axis({"name": "Date", "date_axis": True})
        chart2.set_y_axis({"name": "Drawdown (%)"})
        chart2.set_size({"width": 1000, "height": 450})
        chart2.set_legend({"position": "bottom"})

        ws.insert_chart(f"A{data_rows + 6 + dd_rows + 4}", chart2)

        # ---- Chart 3: Performance Comparison Bar Chart ----
        # Write performance comparison table
        perf_start_row = dd_start_row + dd_rows + 4
        perf_rows = []
        for sr in result.strategies:
            rm = sr.metrics.get("return_metrics", {})
            rk = sr.metrics.get("risk_metrics", {})
            dd_m = sr.metrics.get("drawdown_metrics", {})
            perf_rows.append({
                "Strategy": sr.strategy_name,
                "Total Return (%)": rm.get("total_return_pct", 0),
                "CAGR (%)": rm.get("cagr_pct", 0),
                "Sharpe": rk.get("sharpe_ratio", 0),
                "Max DD (%)": dd_m.get("max_drawdown_pct", 0),
                "Ann. Vol (%)": rk.get("annualized_volatility_pct", 0),
            })

        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_excel(writer, sheet_name=sheet_name, startrow=perf_start_row, index=False)

        chart3 = wb.add_chart({"type": "column"})
        perf_count = len(perf_rows)

        # CAGR bars
        chart3.add_series({
            "name": [sheet_name, perf_start_row, 2],
            "categories": [sheet_name, perf_start_row + 1, 0, perf_start_row + perf_count, 0],
            "values": [sheet_name, perf_start_row + 1, 2, perf_start_row + perf_count, 2],
            "fill": {"color": "#4472C4"},
        })
        # Sharpe bars
        chart3.add_series({
            "name": [sheet_name, perf_start_row, 3],
            "categories": [sheet_name, perf_start_row + 1, 0, perf_start_row + perf_count, 0],
            "values": [sheet_name, perf_start_row + 1, 3, perf_start_row + perf_count, 3],
            "fill": {"color": "#70AD47"},
            "y2_axis": True,
        })

        chart3.set_title({"name": "Strategy Performance Comparison"})
        chart3.set_x_axis({"name": "Strategy"})
        chart3.set_y_axis({"name": "CAGR (%)"})
        chart3.set_y2_axis({"name": "Sharpe Ratio"})
        chart3.set_size({"width": 800, "height": 400})
        chart3.set_legend({"position": "bottom"})

        bar_chart_row = perf_start_row + perf_count + 3
        ws.insert_chart(f"A{bar_chart_row}", chart3)
