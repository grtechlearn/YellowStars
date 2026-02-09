"""
Cryptocurrency data provider using CCXT library.

Supports 100+ crypto exchanges including:
- Binance, Coinbase, Kraken, Bybit, OKX, etc.
- Historical OHLCV data
- Real-time price feeds
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.exceptions import DataProviderError
from yellowstars.core.models import Asset, MarketType, TimeFrame
from yellowstars.data.providers.base import BaseDataProvider


class CryptoDataProvider(BaseDataProvider):
    """Cryptocurrency data provider using CCXT."""

    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: "1m",
        TimeFrame.MINUTE_5: "5m",
        TimeFrame.MINUTE_15: "15m",
        TimeFrame.MINUTE_30: "30m",
        TimeFrame.HOUR_1: "1h",
        TimeFrame.HOUR_4: "4h",
        TimeFrame.DAILY: "1d",
        TimeFrame.WEEKLY: "1w",
        TimeFrame.MONTHLY: "1M",
    }

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        exchange_id: str = "binance",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.exchange_id = exchange_id
        self._exchange = None

    def connect(self) -> bool:
        """Connect to the crypto exchange via CCXT."""
        try:
            import ccxt

            exchange_class = getattr(ccxt, self.exchange_id, None)
            if exchange_class is None:
                raise DataProviderError(
                    f"Exchange '{self.exchange_id}' not supported by CCXT. "
                    f"Supported: {', '.join(ccxt.exchanges[:10])}..."
                )

            config = {"enableRateLimit": True}
            if self.api_key:
                config["apiKey"] = self.api_key
            if self.base_url:
                config["urls"] = {"api": self.base_url}

            self._exchange = exchange_class(config)
            self._exchange.load_markets()
            self._connected = True
            logger.info(f"Connected to {self.exchange_id} via CCXT")
            return True

        except ImportError:
            raise DataProviderError("ccxt not installed. Run: pip install ccxt")
        except Exception as e:
            raise DataProviderError(f"Failed to connect to {self.exchange_id}: {e}")

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from crypto exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/BTC').
        """
        if not self._connected or not self._exchange:
            raise DataProviderError("Not connected. Call connect() first.")

        tf = self.TIMEFRAME_MAP.get(timeframe, "1d")

        logger.info(
            f"Fetching {symbol} from {self.exchange_id}: "
            f"{start_date} to {end_date} ({timeframe.value})"
        )

        all_bars = []
        since = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

        while since < end_ms:
            try:
                ohlcv = self._exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    since=since,
                    limit=1000,
                )

                if not ohlcv:
                    break

                for bar in ohlcv:
                    all_bars.append({
                        "timestamp": pd.Timestamp(bar[0], unit="ms"),
                        "open": bar[1],
                        "high": bar[2],
                        "low": bar[3],
                        "close": bar[4],
                        "volume": bar[5] or 0,
                    })

                # Move to after the last candle
                since = ohlcv[-1][0] + 1

                # Rate limiting
                time.sleep(self._exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Error fetching {symbol} data: {e}")
                break

        if not all_bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_bars)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df = self.validate_dataframe(df)

        logger.info(f"Fetched {len(df)} bars for {symbol} from {self.exchange_id}")
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from exchange."""
        if not self._connected or not self._exchange:
            raise DataProviderError("Not connected.")

        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker.get("last") or ticker.get("close")
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def search_symbols(
        self, query: str, market_type: Optional[MarketType] = None
    ) -> list[Asset]:
        """Search for trading pairs on the exchange."""
        if not self._connected or not self._exchange:
            return []

        query_upper = query.upper()
        results = []

        for symbol, market in self._exchange.markets.items():
            if query_upper in symbol.upper():
                results.append(Asset(
                    symbol=symbol,
                    name=symbol,
                    market_type=MarketType.CRYPTO,
                    exchange=self.exchange_id,
                    currency=market.get("quote", "USDT"),
                ))

            if len(results) >= 20:
                break

        return results

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.CRYPTO]
