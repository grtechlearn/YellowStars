#!/usr/bin/env python3
"""
Quick-start script to run a backtest.

Usage:
    python run_backtest.py
    python run_backtest.py --symbol AAPL --start 2010-01-01

Data Flow:
    1. Pre-download all required symbols to LocalData/ (if not already cached)
    2. Switch DataManager to local-only mode (no network fetches during backtest)
    3. Run Malik White Light strategy on NDX data
    4. Generate 5-sheet Excel report in LocalData/
"""

import sys
from datetime import date

from loguru import logger

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.data.manager import DataManager
from yellowstars.data.local_data_store import LocalDataStore
from yellowstars.data.providers.yahoo_provider import YahooDataProvider
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy
from yellowstars.backtest.engine import BacktestEngine
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.reporting.report_generator import ReportGenerator
from yellowstars.reporting.backtest_excel_report import BacktestExcelReport


def main():
    setup_logging(level="INFO")

    # Load config
    settings = load_settings("config.yaml")

    logger.info("=" * 60)
    logger.info("YellowStars Backtest Runner")
    logger.info("=" * 60)

    # Configure strategy
    strat_settings = settings.strategies[0] if settings.strategies else None
    strategy = MalikWhiteLightStrategy(strat_settings)

    # Determine required symbols and date range
    underlying = strat_settings.underlying_index if strat_settings else "NDX"
    benchmark_symbol = settings.backtest.benchmark_symbol
    long_asset = strat_settings.long_asset if strat_settings else "TQQQ"
    short_asset = strat_settings.short_asset if strat_settings else "SQQQ"

    start_date = date.fromisoformat(settings.backtest.start_date)
    end_date = date.fromisoformat(settings.backtest.end_date) if settings.backtest.end_date else date.today()

    symbols = [underlying, benchmark_symbol, long_asset, short_asset]

    # ---- Step 1: Ensure all data is pre-downloaded to LocalData/ ----
    logger.info("Step 1: Ensuring data in LocalData/...")
    store = LocalDataStore("LocalData")
    provider = YahooDataProvider()
    provider.connect()
    download_results = store.ensure_symbols(symbols, start_date, end_date, provider)

    for sym, count in download_results.items():
        logger.info(f"  {sym}: {count} bars in LocalData/")

    # ---- Step 2: Initialize DataManager in local-only mode ----
    logger.info("Step 2: Switching to local-only mode (no network fetches)...")
    dm = DataManager(settings.data_provider, cache_enabled=True)
    dm.initialize()
    dm.enable_local_only_mode()

    # ---- Step 3: Load data from LocalData/ ----
    logger.info(f"Step 3: Loading data from LocalData/...")
    data = dm.get_data(underlying, start_date, end_date)

    if data.empty:
        logger.error("No data retrieved! Run `python preload_data.py` first.")
        sys.exit(1)

    logger.info(f"  {underlying}: {len(data)} bars ({data.index[0].date()} to {data.index[-1].date()})")

    # Benchmark (Buy & Hold)
    try:
        bench_data = dm.get_data(benchmark_symbol, start_date, end_date)
        logger.info(f"  {benchmark_symbol} (benchmark): {len(bench_data)} bars")
    except Exception as e:
        logger.warning(f"Benchmark data not available: {e}")
        bench_data = None

    # Sell & Hold (SQQQ)
    try:
        sqqq_data = dm.get_data(short_asset, start_date, end_date)
        logger.info(f"  {short_asset} (sell & hold): {len(sqqq_data)} bars")
    except Exception as e:
        logger.warning(f"Sell & Hold data not available: {e}")
        sqqq_data = None

    # ---- Step 4: Run backtest ----
    logger.info("Step 4: Running backtest...")
    engine = BacktestEngine(settings.backtest)
    result = engine.run(strategy, data, bench_data, sell_hold_data=sqqq_data)

    # Calculate and display metrics
    pm = PerformanceMetrics(
        result.equity_curve, result.benchmark_curve, result.trades,
        sell_hold_curve=result.sell_hold_curve,
    )
    print(pm.summary_text())

    # ---- Step 5: Save reports ----
    logger.info("Step 5: Generating reports...")

    # JSON/HTML reports
    reporter = ReportGenerator(settings.reports_directory)
    reporter.generate_backtest_report(result)

    # 5-sheet Excel report to LocalData/
    excel_report = BacktestExcelReport(output_dir="LocalData")
    excel_path = excel_report.generate(result)
    logger.info(f"5-sheet Excel report: {excel_path}")

    logger.info(f"Reports saved to {settings.reports_directory}/ and LocalData/")
    logger.info("Backtest complete!")


if __name__ == "__main__":
    main()
