"""
Yahoo Finance data provider (free, no API key required).

Best for:
- Prototyping and development
- International equities
- Historical data back to 1985+ for major indices
- No rate limit concerns for moderate usage

Limitations:
- Data quality can be inconsistent
- No real-time data (15-min delay)
- Terms of service may restrict commercial use
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.exceptions import DataProviderError, SymbolNotFoundError
from yellowstars.core.models import Asset, MarketType, TimeFrame
from yellowstars.data.providers.base import BaseDataProvider


class YahooDataProvider(BaseDataProvider):
    """Yahoo Finance data provider using yfinance library."""

    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: "1m",
        TimeFrame.MINUTE_5: "5m",
        TimeFrame.MINUTE_15: "15m",
        TimeFrame.MINUTE_30: "30m",
        TimeFrame.HOUR_1: "1h",
        TimeFrame.DAILY: "1d",
        TimeFrame.WEEKLY: "1wk",
        TimeFrame.MONTHLY: "1mo",
    }

    # Common index symbols: Polygon/standard → Yahoo Finance format
    YAHOO_SYMBOL_MAP = {
        "NDX": "^NDX",       # NASDAQ-100
        "GSPC": "^GSPC",     # S&P 500
        "SPX": "^GSPC",      # S&P 500 (alt)
        "DJI": "^DJI",       # Dow Jones
        "IXIC": "^IXIC",     # NASDAQ Composite
        "RUT": "^RUT",       # Russell 2000
        "VIX": "^VIX",       # CBOE Volatility Index
        "TNX": "^TNX",       # 10-Year Treasury Yield
    }

    # Yahoo Finance symbol mapping for international markets
    MARKET_SUFFIX = {
        "india": ".NS",       # NSE
        "india_bse": ".BO",   # BSE
        "uk": ".L",           # London
        "japan": ".T",        # Tokyo
        "germany": ".DE",     # Frankfurt
        "france": ".PA",      # Paris
        "canada": ".TO",      # Toronto
        "australia": ".AX",   # ASX
        "hong_kong": ".HK",   # HKSE
        "singapore": ".SI",   # SGX
    }

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

    def connect(self) -> bool:
        """Verify yfinance is available (no auth required)."""
        try:
            import yfinance
            self._connected = True
            logger.info("Yahoo Finance provider ready (no auth required)")
            return True
        except ImportError:
            raise DataProviderError(
                "yfinance not installed. Run: pip install yfinance"
            )

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance."""
        import yfinance as yf

        if not self._connected:
            self.connect()

        # Auto-translate common index symbols (e.g., NDX → ^NDX)
        yahoo_symbol = self.YAHOO_SYMBOL_MAP.get(symbol.upper(), symbol)
        interval = self.TIMEFRAME_MAP.get(timeframe, "1d")

        logger.info(
            f"Fetching {symbol} (as {yahoo_symbol}) from Yahoo Finance: "
            f"{start_date} to {end_date} ({timeframe.value})"
        )

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_date.isoformat(),
                end=(end_date + pd.Timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=True,
                actions=False,
            )

            if df.empty:
                logger.warning(f"No data returned from Yahoo for {symbol}")
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "timestamp"

            # Keep only OHLCV
            df = df[["open", "high", "low", "close", "volume"]]

            df = self.validate_dataframe(df)
            logger.info(f"Fetched {len(df)} bars for {symbol} from Yahoo")
            return df

        except Exception as e:
            raise DataProviderError(f"Yahoo Finance error for {symbol}: {e}")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Yahoo Finance."""
        import yfinance as yf

        try:
            yahoo_symbol = self.YAHOO_SYMBOL_MAP.get(symbol.upper(), symbol)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.fast_info
            return getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search for symbols using yfinance."""
        import yfinance as yf

        try:
            # yfinance doesn't have a great search API, so we do basic lookup
            ticker = yf.Ticker(query)
            info = ticker.info
            if info and "symbol" in info:
                mtype = MarketType.US_EQUITY
                if market_type:
                    mtype = market_type
                elif info.get("quoteType") == "CRYPTOCURRENCY":
                    mtype = MarketType.CRYPTO

                return [Asset(
                    symbol=info["symbol"],
                    name=info.get("longName", info.get("shortName", query)),
                    market_type=mtype,
                    exchange=info.get("exchange", ""),
                    currency=info.get("currency", "USD"),
                )]
            return []
        except Exception:
            return []

    def get_supported_markets(self) -> list[MarketType]:
        return [
            MarketType.US_EQUITY,
            MarketType.INTERNATIONAL_EQUITY,
            MarketType.CRYPTO,
            MarketType.FOREX,
        ]

    @classmethod
    def get_international_symbol(cls, symbol: str, market: str) -> str:
        """Convert a base symbol to international market symbol.

        Example: get_international_symbol("RELIANCE", "india") -> "RELIANCE.NS"
        """
        suffix = cls.MARKET_SUFFIX.get(market.lower(), "")
        if suffix and not symbol.endswith(suffix):
            return f"{symbol}{suffix}"
        return symbol
