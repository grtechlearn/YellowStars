#!/usr/bin/env python3
"""
Import Malik's Whitelight text file data into LocalData/ as Parquet files.

Reads space-separated text files from Malik's stockdata directory and
converts them to the LocalData/ parquet format used by the backtest engine.

Malik's File Format:
    Aug 04, 2025 22986.77 23191.04 22973.60 23188.61 23188.61 0
    Columns: Date(Mon DD, YYYY) Open High Low Close AdjClose Volume
    Files: ^NDX, ^NDX-15Min, ^RUT, ^SPX, ^VIX, PSQ, SPHB, SQQQ, TQQQ

Usage:
    python import_malik_data.py --source ~/Dropbox/Whitelight/stockdata/
    python import_malik_data.py --source /path/to/stockdata --symbols NDX TQQQ SQQQ
    python import_malik_data.py --source /path/to/stockdata --include-intraday
    python import_malik_data.py --source /path/to/stockdata --status
"""

from __future__ import annotations

import sys
import argparse
from datetime import date
from pathlib import Path

from loguru import logger

from yellowstars.data.providers.malik_textfile_provider import MalikTextFileProvider
from yellowstars.data.local_data_store import LocalDataStore
from yellowstars.core.models import TimeFrame


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
    )


def show_status(provider: MalikTextFileProvider, source_dir: str) -> int:
    """Display available files and their details."""
    symbols = provider.get_available_symbols()

    print(f"\n{'=' * 65}")
    print(f"  Malik Whitelight Stock Data Directory")
    print(f"{'=' * 65}")
    print(f"  Source: {source_dir}")
    print(f"  Files:  {len(symbols)}")
    print(f"{'=' * 65}")
    print(f"  {'Symbol':<10} {'Filename':<15} {'Timeframe':<12} {'Size (KB)':>10}")
    print(f"  {'-' * 10} {'-' * 15} {'-' * 12} {'-' * 10}")

    for info in symbols:
        print(
            f"  {info['symbol']:<10} {info['filename']:<15} "
            f"{info['timeframe']:<12} {info['size_kb']:>10.1f}"
        )

    print(f"{'=' * 65}\n")
    return 0


def import_data(
    provider: MalikTextFileProvider,
    symbols: list[str],
    output_dir: str,
    include_intraday: bool = False,
) -> int:
    """Import text files to parquet format in LocalData/."""
    store = LocalDataStore(output_dir)

    print(f"\n{'=' * 65}")
    print(f"  Malik Whitelight Data Import")
    print(f"{'=' * 65}")
    print(f"  Source:  {provider._stockdata_dir}")
    print(f"  Output:  {output_dir}/")
    print(f"  Symbols: {', '.join(symbols)}")
    if include_intraday:
        print(f"  Intraday: Enabled (15-minute data)")
    print(f"{'=' * 65}\n")

    # ── Import Daily Data ──────────────────────────────────────────────
    results = {}
    for symbol in symbols:
        try:
            # Parse from Malik's text file
            df = provider.get_historical_data(
                symbol,
                start_date=date(1970, 1, 1),  # Get all available data
                end_date=date.today(),
                timeframe=TimeFrame.DAILY,
            )

            if df.empty:
                print(f"  {symbol:<8} | NO DATA")
                results[symbol] = 0
                continue

            # Strip timezone before saving to parquet
            df = LocalDataStore._strip_tz(df)

            # Save as parquet in LocalData/
            parquet_path = store._parquet_path(symbol)
            df.to_parquet(parquet_path)

            # Update manifest
            store._update_manifest(symbol, df, source="malik_textfile")

            results[symbol] = len(df)
            print(
                f"  {symbol:<8} | {len(df):>8,} bars | "
                f"{df.index[0].date()} to {df.index[-1].date()} | "
                f"Saved to {parquet_path.name}"
            )

        except FileNotFoundError as e:
            print(f"  {symbol:<8} | FILE NOT FOUND: {e}")
            results[symbol] = 0
        except Exception as e:
            print(f"  {symbol:<8} | ERROR: {e}")
            results[symbol] = 0

    # ── Import Intraday Data (optional) ─────────────────────────────────
    intraday_results = {}
    if include_intraday:
        print(f"\n  {'─' * 40}")
        print(f"  15-Minute Intraday Data")
        print(f"  {'─' * 40}")

        for symbol in symbols:
            filename = provider._resolve_filename(symbol, TimeFrame.MINUTE_15)
            filepath = provider._stockdata_dir / filename

            if not filepath.exists():
                continue

            try:
                df = provider.get_historical_data(
                    symbol,
                    start_date=date(1970, 1, 1),
                    end_date=date.today(),
                    timeframe=TimeFrame.MINUTE_15,
                )

                if df.empty:
                    continue

                df = LocalDataStore._strip_tz(df)

                # Save with different filename for intraday
                intraday_path = store.data_dir / f"{symbol.upper()}_15min.parquet"
                df.to_parquet(intraday_path)

                intraday_results[symbol] = len(df)
                print(
                    f"  {symbol:<8} (15min) | {len(df):>10,} bars | "
                    f"{df.index[0]} to {df.index[-1]} | "
                    f"Saved to {intraday_path.name}"
                )

            except Exception as e:
                print(f"  {symbol:<8} (15min) | ERROR: {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Import Summary")
    print(f"{'=' * 65}")

    total_daily = sum(results.values())
    successful = sum(1 for v in results.values() if v > 0)
    print(f"  Daily:    {total_daily:>10,} bars across {successful}/{len(results)} symbols")

    if intraday_results:
        total_intraday = sum(intraday_results.values())
        print(f"  Intraday: {total_intraday:>10,} bars across {len(intraday_results)} symbols")

    print(f"  Output:   {output_dir}/")
    print(f"  Manifest: {store.manifest_path}")
    print(f"{'=' * 65}\n")

    # Show how to run backtest
    if successful > 0:
        print("  Next steps:")
        print("    python run_backtest.py          # Run backtest with imported data")
        print("    python update_data.py --status  # Check data status")
        print()

    return 0 if successful > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Malik's Whitelight text file data into LocalData/ as Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_malik_data.py --source ~/Dropbox/Whitelight/stockdata/
  python import_malik_data.py --source /path/to/stockdata --symbols NDX TQQQ SQQQ
  python import_malik_data.py --source /path/to/stockdata --include-intraday
  python import_malik_data.py --source /path/to/stockdata --status
        """,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to Malik's stockdata directory (e.g., ~/Dropbox/Whitelight/stockdata/)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to import (default: all daily files found)",
    )
    parser.add_argument(
        "--output",
        default="LocalData",
        help="Output directory for parquet files (default: LocalData)",
    )
    parser.add_argument(
        "--include-intraday",
        action="store_true",
        help="Also import 15-minute intraday data (e.g., ^NDX-15Min)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show available files and exit (no import)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: WARNING for clean output)",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    # Expand user home directory
    source_path = Path(args.source).expanduser().resolve()

    # Create and connect provider
    provider = MalikTextFileProvider(stockdata_dir=str(source_path))
    if not provider.connect():
        print(f"\nERROR: Cannot access stockdata directory: {source_path}")
        print("Please check the path and try again.")
        return 1

    # Status mode
    if args.status:
        return show_status(provider, str(source_path))

    # Determine symbols to import
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        # Auto-discover all daily symbols from directory
        symbols = []
        for info in provider.get_available_symbols():
            if info["timeframe"] == TimeFrame.DAILY.value:
                symbols.append(info["symbol"])

    if not symbols:
        print("\nERROR: No symbols to import. Check your --source path or --symbols list.")
        return 1

    return import_data(
        provider=provider,
        symbols=symbols,
        output_dir=args.output,
        include_intraday=args.include_intraday,
    )


if __name__ == "__main__":
    sys.exit(main())
