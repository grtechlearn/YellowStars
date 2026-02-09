#!/usr/bin/env python3
"""
Run the daily trading cycle (for scheduled execution).

This script runs one complete daily cycle:
1. Pre-market checks
2. Strategy analysis
3. Trade execution (if paper/live mode)
4. Post-market reporting

Usage:
    python run_daily.py
    python run_daily.py --mode paper
    python run_daily.py --mode backtest  (analysis only, no execution)
"""

import sys
import argparse
import json

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.core.models import TradingMode
from yellowstars.data.manager import DataManager
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.alerting.notifier import Notifier
from yellowstars.orchestrator.daily_runner import DailyRunner


def main():
    parser = argparse.ArgumentParser(description="YellowStars Daily Trading Cycle")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="paper")
    parser.add_argument("--backtest-only", action="store_true", help="Run backtest analysis only")
    args = parser.parse_args()

    setup_logging(level="INFO")

    settings = load_settings(args.config)
    settings.trading_mode = TradingMode(args.mode)

    # Initialize components
    dm = DataManager(settings.data_provider)
    strat_settings = settings.strategies[0] if settings.strategies else None
    strategy = MalikWhiteLightStrategy(strat_settings)
    notifier = Notifier(settings.alerts)

    # Broker (only for paper/live)
    broker = None
    if args.mode in ("paper", "live"):
        try:
            from yellowstars.execution.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(settings.broker)
        except Exception as e:
            print(f"Warning: Broker not configured: {e}")

    # Create runner
    runner = DailyRunner(settings, dm, strategy, broker, notifier)

    if args.backtest_only:
        result = runner.run_backtest_only()
    else:
        result = runner.run_daily_cycle()

    # Print result
    print(json.dumps(result, indent=2, default=str))

    return 0 if result.get("status") in ("success", "hold") else 1


if __name__ == "__main__":
    sys.exit(main())
