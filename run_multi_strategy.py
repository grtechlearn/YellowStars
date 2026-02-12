#!/usr/bin/env python3
"""
Multi-Strategy Backtesting Runner.

Runs multiple trading strategies against historical NDX data and produces:
- Per-strategy CSV order histories
- Consolidated Excel report with charts and comparison metrics

Usage:
    python run_multi_strategy.py
    python run_multi_strategy.py --symbol NDX --capital 100000
    python run_multi_strategy.py --start 2000-01-01 --end 2025-12-31

Strategies:
    1. Buy-and-Hold: Buy on day 1, hold forever
    2. Buy-and-Sell on Closing Day: Buy at open, sell at close each day

Data: Loaded from LocalData/ (pre-downloaded parquet files).
"""

import sys
import argparse
from datetime import date

from loguru import logger

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings, BacktestSettings, StrategySettings
from yellowstars.data.local_data_store import LocalDataStore
from yellowstars.strategies.buy_and_hold import BuyAndHoldStrategy
from yellowstars.strategies.buy_sell_closing_day import BuySellClosingDayStrategy
from yellowstars.backtest.multi_strategy_engine import MultiStrategyEngine
from yellowstars.reporting.multi_strategy_report import MultiStrategyReport


def main():
    parser = argparse.ArgumentParser(
        description="YellowStars Multi-Strategy Backtester"
    )
    parser.add_argument("--symbol", default="NDX", help="Symbol to backtest (default: NDX)")
    parser.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", default="LocalData", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Load config
    settings = load_settings(args.config)

    # Override with CLI arguments
    bt_settings = BacktestSettings(
        start_date=args.start or settings.backtest.start_date,
        end_date=args.end or settings.backtest.end_date,
        initial_capital=args.capital,
        commission_per_trade=settings.backtest.commission_per_trade,
        slippage_pct=settings.backtest.slippage_pct,
    )

    symbol = args.symbol.upper()

    print()
    print("=" * 70)
    print("  YellowStars Multi-Strategy Backtester")
    print("=" * 70)
    print(f"  Symbol:          {symbol}")
    print(f"  Initial Capital: ${bt_settings.initial_capital:,.2f}")
    print(f"  Date Range:      {bt_settings.start_date} to {bt_settings.end_date or 'today'}")
    print(f"  Commission:      ${bt_settings.commission_per_trade:.2f} per trade")
    print(f"  Slippage:        {bt_settings.slippage_pct:.2f} bps")
    print(f"  Output:          {args.output}/")
    print("=" * 70)
    print()

    # ---- Step 1: Load data from LocalData ----
    logger.info("Step 1: Loading data from LocalData/...")
    store = LocalDataStore("LocalData")

    start_date = date.fromisoformat(bt_settings.start_date)
    end_date = (
        date.fromisoformat(bt_settings.end_date)
        if bt_settings.end_date
        else date.today()
    )

    if not store.has_symbol(symbol):
        logger.error(
            f"No data for {symbol} in LocalData/. "
            f"Run `python preload_data.py --symbols {symbol}` first."
        )
        sys.exit(1)

    data = store.load(symbol, start_date, end_date)
    if data.empty:
        logger.error(f"No data in range for {symbol}")
        sys.exit(1)

    logger.info(
        f"  Loaded {len(data)} bars: "
        f"{data.index[0].date()} to {data.index[-1].date()}"
    )

    # ---- Step 2: Configure strategies ----
    logger.info("Step 2: Configuring strategies...")

    strat_settings = StrategySettings(
        underlying_index=symbol,
        long_asset=symbol,
    )

    strategies = [
        BuyAndHoldStrategy(strat_settings),
        BuySellClosingDayStrategy(strat_settings),
    ]

    # ---- Step 3: Run multi-strategy backtest ----
    logger.info(f"Step 3: Running {len(strategies)} strategies...")
    engine = MultiStrategyEngine(bt_settings)
    for s in strategies:
        engine.add_strategy(s)

    result = engine.run(data, symbol=symbol)

    # ---- Step 4: Print summary ----
    print()
    print("=" * 70)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print()

    for sr in result.strategies:
        rm = sr.metrics.get("return_metrics", {})
        rk = sr.metrics.get("risk_metrics", {})
        dd = sr.metrics.get("drawdown_metrics", {})
        tm = sr.metrics.get("trade_metrics", {})

        print(f"  Strategy: {sr.strategy_name}")
        print(f"  {'─' * 50}")
        print(f"    Initial Capital:    ${sr.initial_capital:>14,.2f}")
        print(f"    Final Equity:       ${rm.get('final_equity', 0):>14,.2f}")
        print(f"    Total Return:       {rm.get('total_return_pct', 0):>14.2f}%")
        print(f"    CAGR:               {rm.get('cagr_pct', 0):>14.2f}%")
        print(f"    Sharpe Ratio:       {rk.get('sharpe_ratio', 0):>14.3f}")
        print(f"    Sortino Ratio:      {rk.get('sortino_ratio', 0):>14.3f}")
        print(f"    Max Drawdown:       {dd.get('max_drawdown_pct', 0):>14.2f}%")
        print(f"    Ann. Volatility:    {rk.get('annualized_volatility_pct', 0):>14.2f}%")
        print(f"    Total Trades:       {tm.get('total_trades', 0):>14}")
        print(f"    Total Orders:       {len(sr.orders):>14}")
        print(f"    Win Rate:           {tm.get('win_rate_pct', 0):>14.1f}%")
        print(f"    Best Day:           {rm.get('best_day_pct', 0):>14.2f}%")
        print(f"    Worst Day:          {rm.get('worst_day_pct', 0):>14.2f}%")
        print()

    # ---- Step 5: Generate reports ----
    logger.info("Step 5: Generating reports...")
    reporter = MultiStrategyReport(output_dir=args.output)

    # CSV files (one per strategy)
    csv_paths = reporter.generate_csv_reports(result)
    for p in csv_paths:
        logger.info(f"  CSV: {p}")

    # Consolidated Excel report
    excel_path = reporter.generate_excel_report(result)
    logger.info(f"  Excel: {excel_path}")

    print()
    print("=" * 70)
    print("  GENERATED FILES")
    print("=" * 70)
    for p in csv_paths:
        print(f"  CSV:   {p}")
    print(f"  Excel: {excel_path}")
    print("=" * 70)
    print()
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
