"""
YellowStars CLI - Command-line interface for the trading platform.

Commands:
  backtest    Run a strategy backtest on historical data
  paper       Start paper trading mode
  live        Start live trading mode
  status      Show current portfolio and system status
  data        Data management (fetch, cache, info)
  report      Generate reports
  config      Show/validate configuration
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import click
from loguru import logger

from yellowstars import __version__
from yellowstars.utils.logger_setup import setup_logging


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config, verbose):
    """YellowStars - Systematic Trading Platform

    Autonomous trading system with backtesting, paper trading,
    and live execution capabilities.
    """
    ctx.ensure_object(dict)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Load settings
    from yellowstars.config.settings import load_settings
    settings = load_settings(config)
    ctx.obj["settings"] = settings
    ctx.obj["config_path"] = config


@main.command()
@click.option("--strategy", "-s", default="malik_white_light", help="Strategy name")
@click.option("--symbol", default="", help="Symbol to test (overrides config)")
@click.option("--start", default="1985-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", default="", help="End date (YYYY-MM-DD, default: today)")
@click.option("--capital", default=100000.0, help="Initial capital")
@click.option("--benchmark", default="QQQ", help="Benchmark symbol")
@click.option("--output", "-o", default="reports", help="Output directory")
@click.pass_context
def backtest(ctx, strategy, symbol, start, end, capital, benchmark, output):
    """Run a strategy backtest on historical data.

    Example:
        yellowstars backtest -s malik_white_light --start 1985-01-01 --capital 100000
    """
    settings = ctx.obj["settings"]
    settings.backtest.initial_capital = capital
    settings.backtest.start_date = start
    settings.backtest.end_date = end or ""
    settings.backtest.benchmark_symbol = benchmark
    settings.reports_directory = output

    click.echo(f"\n{'='*60}")
    click.echo(f"  YellowStars Backtest Engine v{__version__}")
    click.echo(f"{'='*60}")
    click.echo(f"  Strategy:  {strategy}")
    click.echo(f"  Period:    {start} to {end or 'today'}")
    click.echo(f"  Capital:   ${capital:,.2f}")
    click.echo(f"  Benchmark: {benchmark}")
    click.echo(f"{'='*60}\n")

    # Initialize data manager
    from yellowstars.data.manager import DataManager
    dm = DataManager(settings.data_provider, cache_enabled=True)
    dm.initialize()

    # Select strategy
    from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
    from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy

    strat_settings = settings.get_strategy(strategy)
    if not strat_settings:
        strat_settings = settings.strategies[0] if settings.strategies else None

    if strategy == "malik_white_light":
        strat = MalikWhiteLightStrategy(strat_settings)
    elif strategy == "ma_crossover":
        strat = MovingAverageCrossoverStrategy(strat_settings)
    else:
        click.echo(f"Unknown strategy: {strategy}")
        click.echo("Available: malik_white_light, ma_crossover")
        sys.exit(1)

    # Fetch data
    underlying = strat_settings.underlying_index if strat_settings else "NDX"
    if symbol:
        underlying = symbol

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    click.echo(f"Fetching data for {underlying}...")
    data = dm.get_data(underlying, start_date, end_date)
    click.echo(f"Data: {len(data)} bars ({data.index[0].date()} to {data.index[-1].date()})")

    # Fetch benchmark
    click.echo(f"Fetching benchmark: {benchmark}...")
    try:
        bench_data = dm.get_data(benchmark, start_date, end_date)
    except Exception:
        bench_data = None
        click.echo("  (Benchmark data not available, using underlying)")

    # Run backtest
    from yellowstars.backtest.engine import BacktestEngine
    engine = BacktestEngine(settings.backtest)

    click.echo("\nRunning backtest...")
    result = engine.run(strat, data, bench_data)

    # Generate report
    from yellowstars.reporting.report_generator import ReportGenerator
    reporter = ReportGenerator(output)
    metrics = reporter.generate_backtest_report(result)

    # Print summary
    from yellowstars.backtest.metrics import PerformanceMetrics
    pm = PerformanceMetrics(result.equity_curve, result.benchmark_curve, result.trades)
    click.echo(pm.summary_text())

    click.echo(f"\nReports saved to: {output}/")


@main.command()
@click.option("--strategy", "-s", default="malik_white_light", help="Strategy name")
@click.pass_context
def paper(ctx, strategy):
    """Start paper trading mode.

    Runs the daily cycle with paper (simulated) money.
    """
    settings = ctx.obj["settings"]
    settings.trading_mode = "paper"

    click.echo(f"\n{'='*60}")
    click.echo(f"  YellowStars PAPER TRADING v{__version__}")
    click.echo(f"{'='*60}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Mode: PAPER (simulated money)")
    click.echo(f"{'='*60}\n")

    _run_trading(settings, strategy)


@main.command()
@click.option("--strategy", "-s", default="malik_white_light", help="Strategy name")
@click.option("--confirm", is_flag=True, help="Confirm live trading")
@click.pass_context
def live(ctx, strategy, confirm):
    """Start live trading mode.

    WARNING: This trades with real money!
    """
    if not confirm:
        click.echo("⚠️  LIVE TRADING uses REAL MONEY!")
        click.echo("Add --confirm flag to proceed.")
        sys.exit(1)

    settings = ctx.obj["settings"]
    settings.trading_mode = "live"

    click.echo(f"\n{'='*60}")
    click.echo(f"  YellowStars LIVE TRADING v{__version__}")
    click.echo(f"  ⚠️  REAL MONEY MODE")
    click.echo(f"{'='*60}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"{'='*60}\n")

    _run_trading(settings, strategy)


@main.command()
@click.pass_context
def status(ctx):
    """Show current system and portfolio status."""
    settings = ctx.obj["settings"]

    click.echo(f"\n{'='*60}")
    click.echo(f"  YellowStars Status v{__version__}")
    click.echo(f"{'='*60}")
    click.echo(f"  Mode:          {settings.trading_mode.value}")
    click.echo(f"  Data Provider: {settings.data_provider.name}")
    click.echo(f"  Broker:        {settings.broker.name}")
    click.echo(f"  Strategies:    {len(settings.strategies)}")

    # Show cache info
    from yellowstars.data.cache.local_cache import LocalCache
    cache = LocalCache(settings.data_provider.cache_directory)
    info = cache.get_cache_info()
    click.echo(f"\n  Cache:")
    click.echo(f"    Files:       {info['total_files']}")
    click.echo(f"    Size:        {info['total_size_mb']:.1f} MB")
    click.echo(f"    Symbols:     {len(info['symbols'])}")

    click.echo(f"{'='*60}")


@main.command()
@click.option("--fetch", "-f", help="Fetch data for symbol")
@click.option("--start", default="2020-01-01", help="Start date")
@click.option("--end", default="", help="End date")
@click.option("--info", is_flag=True, help="Show cache info")
@click.option("--clear", is_flag=True, help="Clear all cached data")
@click.pass_context
def data(ctx, fetch, start, end, info, clear):
    """Data management commands."""
    settings = ctx.obj["settings"]

    if info:
        from yellowstars.data.cache.local_cache import LocalCache
        cache = LocalCache(settings.data_provider.cache_directory)
        cache_info = cache.get_cache_info()
        click.echo(f"Cache: {cache_info['total_files']} files, {cache_info['total_size_mb']:.1f} MB")
        for sym, details in cache_info["symbols"].items():
            click.echo(f"  {sym}: {details['total_size_mb']:.2f} MB")
        return

    if clear:
        from yellowstars.data.cache.local_cache import LocalCache
        cache = LocalCache(settings.data_provider.cache_directory)
        count = cache.clear_all()
        click.echo(f"Cleared {count} cached files")
        return

    if fetch:
        from yellowstars.data.manager import DataManager
        dm = DataManager(settings.data_provider)
        dm.initialize()
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end) if end else date.today()
        df = dm.get_data(fetch, start_date, end_date)
        click.echo(f"Fetched {len(df)} bars for {fetch}")
        if not df.empty:
            click.echo(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")
            click.echo(f"  Latest close: ${df.iloc[-1]['close']:.2f}")
        return

    click.echo("Use --fetch SYMBOL, --info, or --clear")


def _run_trading(settings, strategy_name: str):
    """Common trading loop for paper and live modes."""
    from yellowstars.data.manager import DataManager
    from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
    from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy
    from yellowstars.execution.alpaca_broker import AlpacaBroker
    from yellowstars.alerting.notifier import Notifier
    from yellowstars.orchestrator.daily_runner import DailyRunner

    # Init data manager
    dm = DataManager(settings.data_provider)

    # Init strategy
    strat_settings = settings.get_strategy(strategy_name)
    if strategy_name == "malik_white_light":
        strategy = MalikWhiteLightStrategy(strat_settings)
    else:
        strategy = MovingAverageCrossoverStrategy(strat_settings)

    # Init broker
    broker = AlpacaBroker(settings.broker)

    # Init notifier
    notifier = Notifier(settings.alerts)

    # Create daily runner
    runner = DailyRunner(settings, dm, strategy, broker, notifier)

    # Run single cycle
    click.echo("Running daily trading cycle...")
    results = runner.run_daily_cycle()
    click.echo(f"\nCycle result: {results['status']}")

    if results.get("phases", {}).get("analysis", {}).get("today_signal"):
        signal = results["phases"]["analysis"]["today_signal"]
        click.echo(f"Signal: {signal['action']} | Position: {signal['position_pct']:.1%} | Instrument: {signal['instrument']}")


if __name__ == "__main__":
    main()
