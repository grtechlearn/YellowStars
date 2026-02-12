#!/usr/bin/env python3
"""
Mix Strategy Backtesting Runner.

Runs the consensus-based Mix Strategy against historical data.
Combines 5 sub-strategies (MA Crossover, Bollinger, RSI, MACD, ROC)
and trades only when a configurable number agree.

Usage:
    python run_mix_strategy.py
    python run_mix_strategy.py --symbol NDX --min-votes 3
    python run_mix_strategy.py --symbol NDX --min-votes 2 --stop-loss 10
"""

import sys
import argparse
from datetime import date

from loguru import logger

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings, BacktestSettings, StrategySettings
from yellowstars.data.local_data_store import LocalDataStore
from yellowstars.strategies.mix_strategy import MixStrategy, MixStrategyParams
from yellowstars.backtest.multi_strategy_engine import MultiStrategyEngine
from yellowstars.reporting.multi_strategy_report import MultiStrategyReport


def main():
    parser = argparse.ArgumentParser(description="YellowStars Mix Strategy Backtester")
    parser.add_argument("--symbol", default="NDX", help="Symbol to backtest")
    parser.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--min-votes", type=int, default=3, help="Min votes to trade (1-5)")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Stop loss %% (0=disabled)")
    parser.add_argument("--take-profit", type=float, default=0.0, help="Take profit %% (0=disabled)")
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
    print("  YellowStars Mix Strategy Backtester")
    print("=" * 70)
    print(f"  Symbol:          {symbol}")
    print(f"  Initial Capital: ${bt_settings.initial_capital:,.2f}")
    print(f"  Date Range:      {bt_settings.start_date} to {bt_settings.end_date or 'today'}")
    print(f"  Min Votes:       {args.min_votes}/5")
    print(f"  Stop Loss:       {'OFF' if args.stop_loss == 0 else str(args.stop_loss) + '%'}")
    print(f"  Take Profit:     {'OFF' if args.take_profit == 0 else str(args.take_profit) + '%'}")
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
    logger.info("Step 2: Configuring Mix Strategy...")
    mix_params = MixStrategyParams(
        min_votes=args.min_votes,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    strat_settings = StrategySettings(name="mix_strategy", underlying_index=symbol)
    strategy = MixStrategy(settings=strat_settings, params=mix_params)

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
    print("  MIX STRATEGY RESULTS")
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
