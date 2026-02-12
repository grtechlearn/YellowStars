#!/usr/bin/env python3
"""
Daily data updater. Fetches only missing days for all symbols in LocalData/.

Checks the manifest in LocalData/.manifest.json for existing symbols,
determines date gaps, and fetches only the missing data from Yahoo Finance.

Usage:
    python update_data.py              # Update all known symbols
    python update_data.py --force      # Re-download everything from scratch
    python update_data.py --status     # Show current data status only
"""

import sys
import argparse
from datetime import date

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.data.providers.yahoo_provider import YahooDataProvider
from yellowstars.data.local_data_store import LocalDataStore


def main():
    parser = argparse.ArgumentParser(description="YellowStars Daily Data Updater")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--force", action="store_true", help="Re-download everything")
    parser.add_argument("--status", action="store_true", help="Show data status only")
    args = parser.parse_args()

    setup_logging(level="INFO")

    store = LocalDataStore("LocalData")

    # Show status
    if args.status:
        status = store.get_status()
        if not status:
            print("\nNo data in LocalData/. Run `python preload_data.py` first.")
            return 0

        print("\n" + "=" * 60)
        print("  YellowStars LocalData Status")
        print("=" * 60)
        for symbol, meta in status.items():
            print(f"  {symbol:8s} | {meta.get('bar_count', 0):>6,} bars | "
                  f"{meta.get('first_date', '?')} to {meta.get('last_date', '?')} | "
                  f"updated: {meta.get('last_updated', '?')}")
        print("=" * 60)
        return 0

    # Check if we have any data to update
    status = store.get_status()
    if not status and not args.force:
        print("\nNo data in LocalData/. Run `python preload_data.py` first.")
        return 1

    # Create provider
    provider = YahooDataProvider()
    provider.connect()

    today = date.today()

    print("\n" + "=" * 60)
    print("  YellowStars Daily Data Updater")
    print("=" * 60)
    print(f"  Target date: {today}")
    print("=" * 60 + "\n")

    if args.force:
        # Force re-download: use config to determine symbols and range
        settings = load_settings(args.config)
        strat = settings.strategies[0] if settings.strategies else None
        symbols = []
        if strat:
            symbols = [
                strat.underlying_index,
                strat.benchmark_asset,
                strat.long_asset,
                strat.short_asset,
            ]
        else:
            symbols = ["NDX", "QQQ", "TQQQ", "SQQQ"]

        start_date = date.fromisoformat(settings.backtest.start_date)
        results = store.ensure_symbols(symbols, start_date, today, provider)

        print("\n" + "=" * 60)
        print("  Force Download Summary")
        print("=" * 60)
        for symbol, count in results.items():
            print(f"  {symbol:8s} | {count:>6,} bars")
        print("=" * 60)
    else:
        # Incremental update
        results = store.update_all(provider, target_end=today)

        print("\n" + "=" * 60)
        print("  Update Summary")
        print("=" * 60)
        for symbol, info in results.items():
            new_bars = info.get("new_bars", 0)
            total_bars = info.get("total_bars", 0)
            status_str = info.get("status", "")
            last_date = info.get("last_date", "?")
            print(f"  {symbol:8s} | +{new_bars:>4} new bars | "
                  f"total: {total_bars:>6,} | "
                  f"through: {last_date} | {status_str}")
        print("=" * 60)

    print(f"\nAll data stored in LocalData/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
