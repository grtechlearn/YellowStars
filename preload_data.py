#!/usr/bin/env python3
"""
Pre-download all required market data to LocalData/ for offline backtesting.

Downloads NDX, QQQ, TQQQ, and SQQQ historical data using Yahoo Finance (free).
Data is stored as Parquet files in LocalData/ with a manifest tracking downloads.

Usage:
    python preload_data.py                     # Download all configured symbols
    python preload_data.py --symbols NDX QQQ   # Specific symbols only
    python preload_data.py --update            # Append missing days only
    python preload_data.py --status            # Show current data status
"""

import sys
import argparse
from datetime import date

from yellowstars.utils.logger_setup import setup_logging
from yellowstars.config.settings import load_settings
from yellowstars.data.providers.yahoo_provider import YahooDataProvider
from yellowstars.data.local_data_store import LocalDataStore


def main():
    parser = argparse.ArgumentParser(description="YellowStars Data Pre-loader")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--update", action="store_true", help="Update existing data only")
    parser.add_argument("--status", action="store_true", help="Show data status")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Load settings
    settings = load_settings(args.config)

    # Initialize local data store
    store = LocalDataStore("LocalData")

    # Show status and exit
    if args.status:
        status = store.get_status()
        if not status:
            print("\nNo data in LocalData/. Run `python preload_data.py` to download.")
            return 0

        print("\n" + "=" * 60)
        print("  YellowStars LocalData Status")
        print("=" * 60)
        for symbol, meta in status.items():
            print(f"  {symbol:8s} | {meta['bar_count']:>6,} bars | "
                  f"{meta['first_date']} to {meta['last_date']} | "
                  f"updated: {meta['last_updated']}")
        print("=" * 60)
        return 0

    # Determine symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        # Get all required symbols from strategy config
        strat = settings.strategies[0] if settings.strategies else None
        symbols = []
        if strat:
            symbols = [
                strat.underlying_index,  # NDX
                strat.benchmark_asset,   # QQQ
                strat.long_asset,        # TQQQ
                strat.short_asset,       # SQQQ
            ]
        else:
            symbols = ["NDX", "QQQ", "TQQQ", "SQQQ"]

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s.upper() not in seen:
            seen.add(s.upper())
            unique_symbols.append(s.upper())
    symbols = unique_symbols

    # Create Yahoo provider (free, no API key)
    provider = YahooDataProvider()
    provider.connect()

    # Date range
    start_date = date.fromisoformat(settings.backtest.start_date)
    end_date = (
        date.fromisoformat(settings.backtest.end_date)
        if settings.backtest.end_date
        else date.today()
    )

    print("\n" + "=" * 60)
    print("  YellowStars Data Pre-loader")
    print("=" * 60)
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Output:     LocalData/")
    print("=" * 60 + "\n")

    if args.update:
        # Update existing data only
        results = store.update_all(provider, target_end=end_date)
    else:
        # Full ensure (download or update as needed)
        results = store.ensure_symbols(symbols, start_date, end_date, provider)

    # Print summary
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for symbol, count_or_info in results.items():
        if isinstance(count_or_info, int):
            print(f"  {symbol:8s} | {count_or_info:>6,} bars")
        else:
            info = count_or_info
            print(f"  {symbol:8s} | {info.get('total_bars', 0):>6,} bars | "
                  f"+{info.get('new_bars', 0)} new | {info.get('status', '')}")
    print("=" * 60)
    print(f"\nAll data stored in LocalData/")
    print(f"Manifest: {store.manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
