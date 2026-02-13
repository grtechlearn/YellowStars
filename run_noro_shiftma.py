#!/usr/bin/env python3
"""
Noro ShiftMA Backtesting Runner.

Runs Noro's WhiteBox ShiftMA strategy: buy dips at shifted-down MA,
sell rips at shifted-up MA. Mean-reversion around an MA envelope.

Usage:
    python run_noro_shiftma.py
    python run_noro_shiftma.py --symbol NDX --ma-type SMA --ma-length 3
    python run_noro_shiftma.py --long-level -5 --short-level 10 --enable-short
"""

import sys
import argparse
from datetime import date

from loguru import logger

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings, BacktestSettings, StrategySettings
from yellowstars.data.local_data_store import LocalDataStore
from yellowstars.strategies.noro_shiftma import NoroShiftMAStrategy, NoroShiftMAParams
from yellowstars.backtest.multi_strategy_engine import MultiStrategyEngine
from yellowstars.reporting.multi_strategy_report import MultiStrategyReport


def main():
    parser = argparse.ArgumentParser(description="YellowStars Noro ShiftMA Backtester")
    parser.add_argument("--symbol", default="NDX", help="Symbol to backtest")
    parser.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA", "WMA"], help="MA type")
    parser.add_argument("--ma-length", type=int, default=3, help="MA period")
    parser.add_argument("--ma-source", default="OHLC4", help="Price source")
    parser.add_argument("--long-level", type=float, default=-5.0, help="Long line shift %%")
    parser.add_argument("--short-level", type=float, default=10.0, help="Short line shift %%")
    parser.add_argument("--close-shift", type=float, default=0.0, help="TP close shift %%")
    parser.add_argument("--enable-short", action="store_true", help="Enable short trades")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output", default="LocalData", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    settings = load_settings(args.config)

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
    print("  YellowStars Noro ShiftMA Backtester")
    print("=" * 70)
    print(f"  Symbol:          {symbol}")
    print(f"  Initial Capital: ${bt_settings.initial_capital:,.2f}")
    print(f"  MA:              {args.ma_type}({args.ma_length}) on {args.ma_source}")
    print(f"  Long Line:       {args.long_level}%")
    print(f"  Short Line:      {args.short_level}%")
    print(f"  Close Shift:     {'OFF' if args.close_shift == 0 else str(args.close_shift) + '%'}")
    print(f"  Short Enabled:   {args.enable_short}")
    print("=" * 70)
    print()

    # Load data
    logger.info("Step 1: Loading data...")
    store = LocalDataStore("LocalData")
    start_date = date.fromisoformat(bt_settings.start_date)
    end_date = date.fromisoformat(bt_settings.end_date) if bt_settings.end_date else date.today()

    if not store.has_symbol(symbol):
        logger.error(f"No data for {symbol}. Run `python preload_data.py --symbols {symbol}` first.")
        sys.exit(1)

    data = store.load(symbol, start_date, end_date)
    if data.empty:
        logger.error(f"No data in range for {symbol}")
        sys.exit(1)

    logger.info(f"  {len(data)} bars: {data.index[0].date()} to {data.index[-1].date()}")

    # Configure strategy
    logger.info("Step 2: Configuring Noro ShiftMA...")
    params = NoroShiftMAParams(
        enable_long=True,
        enable_short=args.enable_short,
        ma_length=args.ma_length,
        ma_type=args.ma_type,
        ma_source=args.ma_source,
        short_level=args.short_level,
        long_level=args.long_level,
        close_shift=args.close_shift,
    )
    strat_settings = StrategySettings(name="noro_shiftma", underlying_index=symbol)
    strategy = NoroShiftMAStrategy(settings=strat_settings, params=params)

    # Run backtest
    logger.info("Step 3: Running backtest...")
    engine = MultiStrategyEngine(bt_settings)
    engine.add_strategy(strategy)
    result = engine.run(data, symbol=symbol)

    # Print results
    sr = result.strategies[0]
    rm = sr.metrics.get("return_metrics", {})
    rk = sr.metrics.get("risk_metrics", {})
    dd = sr.metrics.get("drawdown_metrics", {})
    tm = sr.metrics.get("trade_metrics", {})

    print()
    print("=" * 70)
    print("  NORO SHIFTMA RESULTS")
    print("=" * 70)
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
    print(f"    Profit Factor:      {tm.get('profit_factor', 0):>14.2f}")
    print(f"    Avg Holding Days:   {tm.get('avg_holding_days', 0):>14.1f}")
    print()

    # Generate reports
    logger.info("Step 4: Generating reports...")
    reporter = MultiStrategyReport(output_dir=args.output)
    csv_paths = reporter.generate_csv_reports(result)
    excel_path = reporter.generate_excel_report(result)

    print("=" * 70)
    print("  GENERATED FILES")
    print("=" * 70)
    for p in csv_paths:
        print(f"  CSV:   {p}")
    print(f"  Excel: {excel_path}")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
