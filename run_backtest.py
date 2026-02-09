#!/usr/bin/env python3
"""
Quick-start script to run a backtest.

Usage:
    python run_backtest.py
    python run_backtest.py --symbol AAPL --start 2010-01-01
"""

import sys
from datetime import date

from loguru import logger

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.data.manager import DataManager
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy
from yellowstars.backtest.engine import BacktestEngine
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.reporting.report_generator import ReportGenerator
from yellowstars.reporting.excel_exporter import ExcelExporter


def main():
    setup_logging(level="INFO")

    # Load config
    settings = load_settings("config.yaml")

    logger.info("=" * 60)
    logger.info("YellowStars Backtest Runner")
    logger.info("=" * 60)

    # Initialize data manager
    dm = DataManager(settings.data_provider, cache_enabled=True)
    dm.initialize()

    # Configure strategy
    strat_settings = settings.strategies[0] if settings.strategies else None
    strategy = MalikWhiteLightStrategy(strat_settings)

    # Get data
    underlying = strat_settings.underlying_index if strat_settings else "NDX"
    start_date = date.fromisoformat(settings.backtest.start_date)
    end_date = date.fromisoformat(settings.backtest.end_date) if settings.backtest.end_date else date.today()

    logger.info(f"Fetching {underlying} data: {start_date} to {end_date}")
    data = dm.get_data(underlying, start_date, end_date)

    if data.empty:
        logger.error("No data retrieved! Check your data provider configuration.")
        sys.exit(1)

    logger.info(f"Data: {len(data)} bars ({data.index[0].date()} to {data.index[-1].date()})")

    # Fetch benchmark
    benchmark_symbol = settings.backtest.benchmark_symbol
    try:
        bench_data = dm.get_data(benchmark_symbol, start_date, end_date)
        logger.info(f"Benchmark ({benchmark_symbol}): {len(bench_data)} bars")
    except Exception as e:
        logger.warning(f"Benchmark data not available: {e}")
        bench_data = None

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(settings.backtest)
    result = engine.run(strategy, data, bench_data)

    # Calculate and display metrics
    pm = PerformanceMetrics(result.equity_curve, result.benchmark_curve, result.trades)
    print(pm.summary_text())

    # Save reports (JSON/HTML)
    reporter = ReportGenerator(settings.reports_directory)
    reporter.generate_backtest_report(result)

    # Save Excel reports to LocalData/
    excel = ExcelExporter(output_dir=settings.data_directory)
    excel_path = excel.export_full_backtest(result)
    logger.info(f"Excel backtest report: {excel_path}")

    # Save raw history data as Excel
    excel.export_history_data(data, underlying)
    logger.info(f"History data exported for {underlying}")

    logger.info(f"Reports saved to {settings.reports_directory}/ and {settings.data_directory}/")
    logger.info("Backtest complete!")


if __name__ == "__main__":
    main()
