"""
Malik's Text File Data Provider.

Reads stock data from Malik's Whitelight system which stores OHLCV data
as space-separated text files. Files have no extension and are stored in
a flat directory structure like:

    Dropbox/Whitelight/stockdata/
        ^NDX           (daily OHLCV, ~682 KB)
        ^NDX-15Min     (15-minute intraday, ~9.4 MB)
        ^RUT           (daily)
        ^SPX           (daily)
        ^VIX           (daily)
        PSQ            (daily)
        SPHB           (daily)
        SQQQ           (daily)
        TQQQ           (daily)

File format: space-separated, reverse chronological, no header
    Aug 04, 2025 22986.77 23191.04 22973.60 23188.61 23188.61 0
    Columns: Date(Mon DD, YYYY) Open High Low Close AdjClose Volume

Usage:
    provider = MalikTextFileProvider(stockdata_dir="/path/to/stockdata")
    provider.connect()
    df = provider.get_historical_data("NDX", date(1985, 1, 1), date(2025, 8, 4))
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.models import Asset, MarketType, TimeFrame
from yellowstars.data.providers.base import BaseDataProvider


class MalikTextFileProvider(BaseDataProvider):
    """Data provider that reads Malik's space-separated text files.

    Malik's Whitelight system stores historical OHLCV data as flat text files
    in a stockdata directory. Each file contains one symbol's data with lines
    in reverse chronological order (newest first).

    Symbol Mapping:
        Internal "NDX"  -> file "^NDX"
        Internal "SPX"  -> file "^SPX"
        Internal "TQQQ" -> file "TQQQ"
        Timeframe suffix: "NDX" + MINUTE_15 -> "^NDX-15Min"
    """

    # Maps internal symbol name -> filename in stockdata directory
    SYMBOL_TO_FILE_MAP: dict[str, str] = {
        "NDX": "^NDX",
        "RUT": "^RUT",
        "SPX": "^SPX",
        "VIX": "^VIX",
        "PSQ": "PSQ",
        "SPHB": "SPHB",
        "SQQQ": "SQQQ",
        "TQQQ": "TQQQ",
        "QQQ": "QQQ",
    }

    # Reverse map: filename -> internal symbol
    FILE_TO_SYMBOL_MAP: dict[str, str] = {v: k for k, v in SYMBOL_TO_FILE_MAP.items()}

    # Intraday timeframe file suffixes
    TIMEFRAME_SUFFIX_MAP: dict[TimeFrame, str] = {
        TimeFrame.MINUTE_15: "-15Min",
    }

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        stockdata_dir: str = "",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        # stockdata_dir can come from explicit param or base_url (config)
        dir_path = stockdata_dir or base_url or kwargs.get("stockdata_dir", "")
        self._stockdata_dir = Path(dir_path) if dir_path else Path(".")

    def connect(self) -> bool:
        """Verify the stockdata directory exists and contains data files."""
        if not self._stockdata_dir.exists():
            logger.error(f"Malik stockdata directory not found: {self._stockdata_dir}")
            return False

        # Check for at least one expected file
        found_files = self._list_available_files()
        if not found_files:
            logger.error(
                f"No data files found in {self._stockdata_dir}. "
                f"Expected files like ^NDX, ^SPX, TQQQ, etc."
            )
            return False

        self._connected = True
        logger.info(
            f"Malik TextFile provider connected: {self._stockdata_dir} "
            f"({len(found_files)} files: {', '.join(sorted(found_files))})"
        )
        return True

    def _list_available_files(self) -> list[str]:
        """List all data files in the stockdata directory."""
        if not self._stockdata_dir.exists():
            return []
        return [
            f.name
            for f in self._stockdata_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    def _resolve_filename(
        self, symbol: str, timeframe: TimeFrame = TimeFrame.DAILY
    ) -> str:
        """Map internal symbol + timeframe to Malik's filename.

        Examples:
            ("NDX", DAILY)      -> "^NDX"
            ("NDX", MINUTE_15)  -> "^NDX-15Min"
            ("TQQQ", DAILY)     -> "TQQQ"
            ("^NDX", DAILY)     -> "^NDX"  (already in file format)
        """
        sym = symbol.upper()

        # If already in file format (starts with ^), use as-is
        if sym.startswith("^"):
            base = sym
        else:
            base = self.SYMBOL_TO_FILE_MAP.get(sym, sym)

        suffix = self.TIMEFRAME_SUFFIX_MAP.get(timeframe, "")
        return f"{base}{suffix}"

    def _file_to_internal_symbol(self, filename: str) -> tuple[str, TimeFrame]:
        """Convert a filename back to internal symbol and timeframe.

        Examples:
            "^NDX"       -> ("NDX", DAILY)
            "^NDX-15Min" -> ("NDX", MINUTE_15)
            "TQQQ"       -> ("TQQQ", DAILY)
        """
        # Check for timeframe suffixes
        timeframe = TimeFrame.DAILY
        base_name = filename
        for tf, suffix in self.TIMEFRAME_SUFFIX_MAP.items():
            if filename.endswith(suffix):
                timeframe = tf
                base_name = filename[: -len(suffix)]
                break

        # Map filename to internal symbol
        if base_name in self.FILE_TO_SYMBOL_MAP:
            symbol = self.FILE_TO_SYMBOL_MAP[base_name]
        elif base_name.startswith("^"):
            symbol = base_name[1:]  # Strip ^ prefix
        else:
            symbol = base_name

        return symbol, timeframe

    def _parse_textfile(self, filepath: Path) -> pd.DataFrame:
        """Parse Malik's space-separated text file into a DataFrame.

        Each line has 9 space-separated tokens:
            Mon DD, YYYY Open High Low Close AdjClose Volume

        The date spans 3 tokens: month_abbrev day_with_comma year.
        Files are in reverse chronological order (newest first).
        The base class validate_dataframe() handles sorting ascending.

        Args:
            filepath: Path to the text file.

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and DatetimeIndex named 'timestamp', sorted ascending.
        """
        rows = []
        parse_errors = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 9:
                    parse_errors += 1
                    if parse_errors <= 5:  # Only log first 5 errors
                        logger.warning(
                            f"Skipping malformed line {line_num} in "
                            f"{filepath.name}: expected 9 tokens, got {len(parts)}"
                        )
                    continue

                try:
                    # Date: first 3 tokens — "Aug 04, 2025"
                    date_str = f"{parts[0]} {parts[1]} {parts[2]}"
                    timestamp = datetime.strptime(date_str, "%b %d, %Y")

                    row = {
                        "timestamp": timestamp,
                        "open": float(parts[3]),
                        "high": float(parts[4]),
                        "low": float(parts[5]),
                        "close": float(parts[6]),
                        # parts[7] = AdjClose — skipped (we use close)
                        "volume": float(parts[8]),
                    }
                    rows.append(row)

                except (ValueError, IndexError) as e:
                    parse_errors += 1
                    if parse_errors <= 5:
                        logger.warning(
                            f"Parse error at line {line_num} in "
                            f"{filepath.name}: {e}"
                        )
                    continue

        if parse_errors > 5:
            logger.warning(
                f"Total parse errors in {filepath.name}: {parse_errors}"
            )

        if not rows:
            logger.warning(f"No valid data rows in {filepath.name}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"

        # validate_dataframe handles: sort ascending, dedup, NaN cleanup
        df = self.validate_dataframe(df)

        logger.info(
            f"Parsed {filepath.name}: {len(df):,} bars "
            f"({df.index[0].date()} to {df.index[-1].date()})"
        )
        return df

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Read historical data from Malik's text file.

        Args:
            symbol: Internal symbol (e.g., "NDX", "TQQQ").
            start_date: Start of date range.
            end_date: End of date range.
            timeframe: Bar timeframe (DAILY or MINUTE_15).

        Returns:
            DataFrame with OHLCV data filtered to the requested range.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self._connected:
            self.connect()

        filename = self._resolve_filename(symbol, timeframe)
        filepath = self._stockdata_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"No data file for {symbol} ({timeframe.value}): "
                f"expected {filepath}"
            )

        df = self._parse_textfile(filepath)

        if df.empty:
            return df

        # Filter to requested date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]

        logger.info(
            f"Loaded {len(filtered):,} bars for {symbol} "
            f"({start_date} to {end_date}) from {filepath.name}"
        )
        return filtered

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent close price from the text file.

        Reads only the first line (newest entry in reverse-chrono file)
        for efficiency — no need to parse the entire file.
        """
        try:
            filename = self._resolve_filename(symbol)
            filepath = self._stockdata_dir / filename
            if not filepath.exists():
                return None

            with open(filepath, "r") as f:
                first_line = f.readline().strip()

            if not first_line:
                return None

            parts = first_line.split()
            if len(parts) >= 7:
                return float(parts[6])  # close price
            return None

        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search available symbols in the stockdata directory."""
        results = []
        available = self._list_available_files()

        for filename in available:
            symbol, timeframe = self._file_to_internal_symbol(filename)

            if (
                query.upper() in symbol.upper()
                or query.upper() in filename.upper()
            ):
                results.append(
                    Asset(
                        symbol=symbol,
                        name=f"{symbol} ({timeframe.value}) [Malik TextFile]",
                        market_type=market_type or MarketType.US_EQUITY,
                        exchange="Malik/Whitelight",
                    )
                )

        return results

    def get_available_symbols(self) -> list[dict]:
        """List all available symbols with metadata.

        Returns a list of dicts with symbol, filename, timeframe, and file size.
        """
        symbols = []
        for filename in sorted(self._list_available_files()):
            filepath = self._stockdata_dir / filename
            symbol, timeframe = self._file_to_internal_symbol(filename)
            size_kb = filepath.stat().st_size / 1024

            symbols.append({
                "symbol": symbol,
                "filename": filename,
                "timeframe": timeframe.value,
                "size_kb": round(size_kb, 1),
            })

        return symbols
