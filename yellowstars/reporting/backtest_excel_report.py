"""
Backtest Excel Report -- Malik's PositionTradingStrategy format.

Matches the exact tab structure and column format from Malik's
Google Sheet "PositionTradingStrategy":

Tabs:
1. EquityCurve       -- Strategy vs QQQ Buy&Hold vs TQQQ(Simulated) Buy&Hold
2. MonthlyPerf       -- Monthly return heatmap (Year x Jan-Dec + Annual)
3. Drawdowns         -- Drawdown % time series chart data
4. Trades            -- Trade log: Symbol, StartDate, EndDate, Duration, BuyPrice,
                        SellPrice, ShareSize, Profit, Profit Percent, IsProfitable,
                        IsShortSale, BuyInfo, SellInfo
5. PortfolioGrowth   -- Growth of portfolio over time

Summary header on Trades sheet:
  Strategy Name, Total Profit, Winning Percentage, Profit/Loss Ratio, Total trades

Saved to LocalData/ by default.
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.backtest.engine import BacktestResult
from yellowstars.backtest.metrics import PerformanceMetrics


class BacktestExcelReport:
    """Generate a 5-tab Excel workbook matching Malik's PositionTradingStrategy format."""

    def __init__(self, output_dir: str = "LocalData"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, result: BacktestResult) -> str:
        """Generate the 5-tab Excel workbook.

        Args:
            result: BacktestResult from the engine.

        Returns:
            Path to the saved Excel file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_report_{result.strategy_name}_{timestamp}.xlsx"
        filepath = self.output_dir / filename

        # Strip timezone from all datetime indices
        self._strip_timezones(result)

        # Compute metrics
        pm = PerformanceMetrics(
            result.equity_curve,
            result.benchmark_curve,
            result.trades,
            sell_hold_curve=result.sell_hold_curve,
        )
        metrics = pm.compute_all()

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            wb = writer.book

            # Tab 1: EquityCurve
            self._write_equity_curve(writer, wb, result)

            # Tab 2: MonthlyPerf
            self._write_monthly_perf(writer, wb, result)

            # Tab 3: Drawdowns
            self._write_drawdowns(writer, wb, result)

            # Tab 4: Trades (with summary header)
            self._write_trades(writer, wb, result, metrics)

            # Tab 5: PortfolioGrowth
            self._write_portfolio_growth(writer, wb, result)

        logger.info(f"Backtest report saved: {filepath}")
        return str(filepath)

    # ==================================================================
    # Tab 1: EquityCurve
    # ==================================================================

    def _write_equity_curve(self, writer, wb, result: BacktestResult):
        """Tab 1: Strategy vs QQQ Buy&Hold vs TQQQ(Simulated) Buy&Hold.

        Three columns of cumulative equity over time, matching Malik's
        chart with blue (My strategy), red (QQQ B&H), yellow (TQQQ B&H).
        """
        eq_df = pd.DataFrame(index=result.equity_curve.index)
        eq_df.index.name = "Date"
        eq_df["My strategy"] = result.equity_curve.values

        # Buy & Hold benchmark (QQQ)
        if result.benchmark_curve is not None and not result.benchmark_curve.empty:
            eq_df["QQQ Buy and Hold"] = result.benchmark_curve.reindex(
                eq_df.index
            )

        # TQQQ Simulated Buy & Hold (if sell_hold_curve is actually TQQQ B&H)
        # Note: In Malik's chart it shows TQQQ B&H. Our sell_hold is SQQQ.
        # We include both if available.
        if result.sell_hold_curve is not None and not result.sell_hold_curve.empty:
            eq_df["TQQQ(Simulated) Buy and Hold"] = result.sell_hold_curve.reindex(
                eq_df.index
            )

        eq_df.to_excel(writer, sheet_name="EquityCurve")

        # Format
        ws = writer.sheets["EquityCurve"]
        ws.set_column("A:A", 14)   # Date
        ws.set_column("B:D", 22)   # Equity columns

        # Add chart
        chart = wb.add_chart({"type": "line"})
        data_rows = len(eq_df)

        chart.add_series({
            "name": "My strategy",
            "categories": ["EquityCurve", 1, 0, data_rows, 0],
            "values": ["EquityCurve", 1, 1, data_rows, 1],
            "line": {"color": "#4472C4", "width": 1.5},
        })

        if "QQQ Buy and Hold" in eq_df.columns:
            chart.add_series({
                "name": "QQQ Buy and Hold",
                "categories": ["EquityCurve", 1, 0, data_rows, 0],
                "values": ["EquityCurve", 1, 2, data_rows, 2],
                "line": {"color": "#C0504D", "width": 1.5},
            })

        if "TQQQ(Simulated) Buy and Hold" in eq_df.columns:
            chart.add_series({
                "name": "TQQQ(Simulated) Buy and Hold",
                "categories": ["EquityCurve", 1, 0, data_rows, 0],
                "values": ["EquityCurve", 1, 3, data_rows, 3],
                "line": {"color": "#E2A83E", "width": 1.5},
            })

        chart.set_title({"name": "Equity Curve Comparison"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Equity ($)", "log_base": 10})
        chart.set_size({"width": 960, "height": 500})
        chart.set_legend({"position": "top"})

        ws.insert_chart("A3", chart, {"x_offset": 0, "y_offset": 0})

    # ==================================================================
    # Tab 2: MonthlyPerf
    # ==================================================================

    def _write_monthly_perf(self, writer, wb, result: BacktestResult):
        """Tab 2: Monthly return heatmap — Year x Jan-Dec + Annual total.

        Format matches Malik's MonthlyPerf sheet:
        Row per year (1985-present), columns Jan-Dec + Year total.
        Green text for positive, red for negative.
        """
        eq = result.equity_curve
        if eq is None or eq.empty:
            pd.DataFrame({"Note": ["No equity data"]}).to_excel(
                writer, sheet_name="MonthlyPerf", index=False
            )
            return

        # Calculate daily returns
        daily_ret = eq.pct_change().fillna(0)

        # Group by year and month
        monthly_returns = {}
        for dt, ret in daily_ret.items():
            yr = dt.year
            mo = dt.month
            if yr not in monthly_returns:
                monthly_returns[yr] = {}
            if mo not in monthly_returns[yr]:
                monthly_returns[yr][mo] = 1.0
            monthly_returns[yr][mo] *= (1 + ret)

        # Convert compounded returns to percentage
        for yr in monthly_returns:
            for mo in monthly_returns[yr]:
                monthly_returns[yr][mo] = (monthly_returns[yr][mo] - 1) * 100

        # Build DataFrame: rows=years, columns=Jan-Dec + Year
        years = sorted(monthly_returns.keys())
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        rows = []
        for yr in years:
            row = {"": yr}
            annual_compound = 1.0
            for mo_idx, mo_name in enumerate(month_names, 1):
                val = monthly_returns[yr].get(mo_idx)
                if val is not None:
                    row[mo_name] = round(val, 2)
                    annual_compound *= (1 + val / 100)
                else:
                    row[mo_name] = ""
            row["Year"] = round((annual_compound - 1) * 100, 2)
            rows.append(row)

        monthly_df = pd.DataFrame(rows)
        monthly_df.to_excel(writer, sheet_name="MonthlyPerf", index=False)

        # Format with conditional coloring (green positive, red negative)
        ws = writer.sheets["MonthlyPerf"]
        ws.set_column("A:A", 8)    # Year
        ws.set_column("B:M", 10)   # Jan-Dec
        ws.set_column("N:N", 12)   # Year total

        # Define formats
        green_fmt = wb.add_format({
            "num_format": "0.00",
            "font_color": "#008000",
            "bold": False,
        })
        red_fmt = wb.add_format({
            "num_format": "0.00",
            "font_color": "#CC0000",
            "bold": False,
        })
        bold_green_fmt = wb.add_format({
            "num_format": "0.00",
            "font_color": "#008000",
            "bold": True,
        })
        bold_red_fmt = wb.add_format({
            "num_format": "0.00",
            "font_color": "#CC0000",
            "bold": True,
        })

        # Apply conditional formatting to monthly columns (B2:M{last_row})
        last_row = len(monthly_df)
        for col_idx in range(1, 13):  # Columns B-M (Jan-Dec)
            col_letter = chr(ord("A") + col_idx)
            cell_range = f"{col_letter}2:{col_letter}{last_row + 1}"
            ws.conditional_format(cell_range, {
                "type": "cell",
                "criteria": ">",
                "value": 0,
                "format": green_fmt,
            })
            ws.conditional_format(cell_range, {
                "type": "cell",
                "criteria": "<",
                "value": 0,
                "format": red_fmt,
            })

        # Apply conditional formatting to Year column (N)
        year_range = f"N2:N{last_row + 1}"
        ws.conditional_format(year_range, {
            "type": "cell",
            "criteria": ">",
            "value": 0,
            "format": bold_green_fmt,
        })
        ws.conditional_format(year_range, {
            "type": "cell",
            "criteria": "<",
            "value": 0,
            "format": bold_red_fmt,
        })

    # ==================================================================
    # Tab 3: Drawdowns
    # ==================================================================

    def _write_drawdowns(self, writer, wb, result: BacktestResult):
        """Tab 3: Drawdown % time series with chart.

        Shows daily drawdown percentage from peak equity.
        Matches Malik's Drawdowns tab with the red area chart.
        """
        eq = result.equity_curve
        if eq is None or eq.empty:
            pd.DataFrame({"Note": ["No equity data"]}).to_excel(
                writer, sheet_name="Drawdowns", index=False
            )
            return

        running_max = eq.expanding().max()
        drawdown_pct = (eq - running_max) / running_max * 100

        dd_df = pd.DataFrame({
            "Drawdown %": drawdown_pct.values,
        }, index=eq.index)
        dd_df.index.name = "Date"

        dd_df.to_excel(writer, sheet_name="Drawdowns")

        # Format
        ws = writer.sheets["Drawdowns"]
        ws.set_column("A:A", 14)
        ws.set_column("B:B", 14)

        # Add area chart (matching Malik's red drawdown chart)
        chart = wb.add_chart({"type": "area"})
        data_rows = len(dd_df)

        chart.add_series({
            "name": "Drawdown %",
            "categories": ["Drawdowns", 1, 0, data_rows, 0],
            "values": ["Drawdowns", 1, 1, data_rows, 1],
            "fill": {"color": "#E06666", "transparency": 30},
            "line": {"color": "#CC0000", "width": 0.5},
        })

        chart.set_title({"name": "Drawdown Analysis"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Drawdown %"})
        chart.set_size({"width": 960, "height": 400})
        chart.set_legend({"none": True})

        ws.insert_chart("A3", chart, {"x_offset": 0, "y_offset": 0})

    # ==================================================================
    # Tab 4: Trades
    # ==================================================================

    def _write_trades(self, writer, wb, result: BacktestResult, metrics: dict):
        """Tab 4: Trade log matching Malik's exact format.

        Header rows 1-5: Summary stats
        Row 7: Column headers
        Row 8+: Trade data

        Columns: Symbol, StartDate, EndDate, Duration, BuyPrice, SellPrice,
                 ShareSize, Profit, Profit Percent, IsProfitable, IsShortSale,
                 BuyInfo, SellInfo
        """
        sheet_name = "Trades"

        # --- Summary Header (rows 1-5) ---
        trade_metrics = metrics.get("trade_metrics", {})
        return_metrics = metrics.get("return_metrics", {})

        # Calculate stats matching Malik's format
        total_trades = trade_metrics.get("total_trades", len(result.trades))
        win_rate = trade_metrics.get("win_rate_pct", 0)

        # Profit/Loss Ratio = avg_win / avg_loss (as ratio)
        avg_win = trade_metrics.get("avg_win", 0)
        avg_loss = abs(trade_metrics.get("avg_loss", 1)) or 1
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Total profit
        total_profit = sum(t.net_pnl for t in result.trades) if result.trades else 0

        summary_rows = [
            {"A": "Strategy Name :", "B": result.strategy_name or "My strategy"},
            {"A": f"Total Profit : {total_profit}"},
            {"A": f"Winning Percentage : {win_rate}"},
            {"A": f"Proft/Loss Ratio : {pl_ratio}"},
            {"A": f"Total trades : {total_trades}"},
            {"A": ""},  # Empty row 6
        ]

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        # --- Trade Data (starting at row 7) ---
        trade_rows = []
        for t in result.trades:
            symbol = t.asset.symbol if t.asset else ""
            start_dt = t.entry_time.strftime("%Y-%m-%d--00-") if t.entry_time else ""
            end_dt = t.exit_time.strftime("%Y-%m-%d--00-") if t.exit_time else ""
            duration = t.holding_period_days if t.holding_period_days else 0
            buy_price = t.entry_price
            sell_price = t.exit_price
            share_size = t.quantity
            profit = t.net_pnl
            profit_pct = t.pnl_pct
            is_profitable = profit > 0
            is_short_sale = False  # Always FALSE — buys TQQQ/SQQQ, never short-sells

            trade_rows.append({
                "Symbol": symbol,
                "StartDate": start_dt,
                "EndDate": end_dt,
                "Duration": duration,
                "BuyPrice": round(buy_price, 6) if buy_price else 0,
                "SellPrice": round(sell_price, 6) if sell_price else 0,
                "ShareSize": round(share_size, 2) if share_size else 0,
                "Profit": round(profit, 2) if profit else 0,
                "Profit Percent": round(profit_pct, 6) if profit_pct else 0,
                "IsProfitable": is_profitable,
                "IsShortSale": is_short_sale,
                "BuyInfo": "",
                "SellInfo": "",
            })

        if trade_rows:
            trades_df = pd.DataFrame(trade_rows)
        else:
            trades_df = pd.DataFrame(columns=[
                "Symbol", "StartDate", "EndDate", "Duration", "BuyPrice",
                "SellPrice", "ShareSize", "Profit", "Profit Percent",
                "IsProfitable", "IsShortSale", "BuyInfo", "SellInfo",
            ])

        # Write trades starting at row 7 (index 6 in 0-based, header at row 7)
        trades_df.to_excel(
            writer, sheet_name=sheet_name, index=False, startrow=6
        )

        # --- Formatting ---
        ws = writer.sheets[sheet_name]

        # Column widths
        ws.set_column("A:A", 12)   # Symbol
        ws.set_column("B:C", 20)   # StartDate, EndDate
        ws.set_column("D:D", 10)   # Duration
        ws.set_column("E:F", 16)   # BuyPrice, SellPrice
        ws.set_column("G:G", 18)   # ShareSize
        ws.set_column("H:H", 22)   # Profit
        ws.set_column("I:I", 16)   # Profit Percent
        ws.set_column("J:K", 14)   # IsProfitable, IsShortSale
        ws.set_column("L:M", 12)   # BuyInfo, SellInfo

        # Conditional formatting for IsProfitable column (J)
        green_bg = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        red_bg = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

        last_row = 7 + len(trade_rows)
        ws.conditional_format(f"J8:J{last_row}", {
            "type": "text",
            "criteria": "containing",
            "value": "TRUE",
            "format": green_bg,
        })
        ws.conditional_format(f"J8:J{last_row}", {
            "type": "text",
            "criteria": "containing",
            "value": "FALSE",
            "format": red_bg,
        })

    # ==================================================================
    # Tab 5: PortfolioGrowth
    # ==================================================================

    def _write_portfolio_growth(self, writer, wb, result: BacktestResult):
        """Tab 5: Portfolio growth — daily equity with position details.

        Shows daily portfolio value, cash, position, and instrument.
        """
        ph = result.position_history

        if ph is None or ph.empty:
            # Fallback: just write equity curve
            eq_df = pd.DataFrame({
                "Equity ($)": result.equity_curve.values,
            }, index=result.equity_curve.index)
            eq_df.index.name = "Date"
            eq_df.to_excel(writer, sheet_name="PortfolioGrowth")

            ws = writer.sheets["PortfolioGrowth"]
            ws.set_column("A:A", 14)
            ws.set_column("B:B", 18)
            return

        portfolio_df = pd.DataFrame({
            "Date": ph["timestamp"],
            "Cash ($)": ph["cash"].round(2),
            "Position Value ($)": ph["position_value"].round(2),
            "Total Equity ($)": ph["total_equity"].round(2),
            "Position (%)": (ph["position_pct"] * 100).round(1),
            "Instrument": ph["instrument"],
        })

        # Cash percentage
        portfolio_df["Cash (%)"] = (
            (ph["cash"] / ph["total_equity"].replace(0, np.nan)) * 100
        ).round(1).fillna(0)

        # Strip timezone from Date column
        if len(portfolio_df) > 0:
            try:
                if hasattr(portfolio_df["Date"].iloc[0], "tz"):
                    portfolio_df["Date"] = portfolio_df["Date"].dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass

        portfolio_df.to_excel(writer, sheet_name="PortfolioGrowth", index=False)

        # Format
        ws = writer.sheets["PortfolioGrowth"]
        ws.set_column("A:A", 14)   # Date
        ws.set_column("B:D", 18)   # Cash, Position Value, Total Equity
        ws.set_column("E:E", 14)   # Position %
        ws.set_column("F:F", 12)   # Instrument
        ws.set_column("G:G", 12)   # Cash %

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _strip_timezones(result: BacktestResult):
        """Strip timezone info from all datetime indices."""
        series_fields = [
            "equity_curve", "benchmark_curve", "sell_hold_curve",
            "daily_returns", "benchmark_returns", "sell_hold_returns",
        ]
        for field_name in series_fields:
            series = getattr(result, field_name, None)
            if series is not None and hasattr(series, "index"):
                if hasattr(series.index, "tz") and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)

        df_fields = ["signals_df", "position_history"]
        for field_name in df_fields:
            df = getattr(result, field_name, None)
            if df is not None and hasattr(df, "index"):
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
