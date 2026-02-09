"""
Alpaca Broker Integration.

Supports:
- Paper trading (default) and live trading
- Market orders, limit orders, stop orders
- Fractional shares
- Real-time order status
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from loguru import logger

from yellowstars.core.exceptions import (
    BrokerConnectionError,
    ExecutionError,
    InsufficientFundsError,
    MissingAPIKeyError,
    OrderRejectedError,
)
from yellowstars.core.models import (
    Asset,
    BrokerConfig,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionSide,
)
from yellowstars.execution.broker_base import BaseBroker


class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker for paper and live trading."""

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._api = None

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        if not self.config.api_key or not self.config.api_secret:
            raise MissingAPIKeyError(
                "Alpaca API key and secret required. "
                "Set via YS_BROKER__API_KEY and YS_BROKER__API_SECRET"
            )

        try:
            from alpaca.trading.client import TradingClient

            base_url = self.config.base_url or (
                self.PAPER_URL if self.config.paper_trading else self.LIVE_URL
            )

            self._api = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret,
                paper=self.config.paper_trading,
            )

            # Test connection
            account = self._api.get_account()
            mode = "PAPER" if self.config.paper_trading else "LIVE"
            logger.info(
                f"Connected to Alpaca ({mode}): "
                f"Equity=${float(account.equity):,.2f}, "
                f"Buying Power=${float(account.buying_power):,.2f}"
            )
            self._connected = True
            return True

        except ImportError:
            raise BrokerConnectionError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )
        except Exception as e:
            raise BrokerConnectionError(f"Alpaca connection failed: {e}")

    def get_account(self) -> dict:
        """Get Alpaca account info."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        account = self._api.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "day_trade_count": account.daytrade_count,
            "status": account.status,
            "trading_blocked": account.trading_blocked,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state from Alpaca."""
        account = self.get_account()
        positions = self.get_positions()

        return Portfolio(
            cash=account["cash"],
            positions=positions,
            total_equity=account["equity"],
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        raw_positions = self._api.get_all_positions()
        positions = []

        for p in raw_positions:
            asset = Asset(
                symbol=p.symbol,
                name=p.symbol,
                market_type=MarketType.US_EQUITY,
            )
            side = PositionSide.LONG if float(p.qty) > 0 else PositionSide.SHORT
            positions.append(Position(
                asset=asset,
                side=side,
                quantity=abs(float(p.qty)),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
                last_updated=datetime.now(),
            ))

        return positions

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpSide, TimeInForce

            symbol = order.asset.symbol if order.asset else ""
            side = AlpSide.BUY if order.side == OrderSide.BUY else AlpSide.SELL

            # Build order request based on type
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=order.stop_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            # Submit
            response = self._api.submit_order(request)

            order.broker_order_id = str(response.id)
            order.status = self._map_status(str(response.status))

            logger.info(
                f"Order submitted: {order.side.value} {order.quantity} {symbol} "
                f"({order.order_type.value}) → {order.broker_order_id}"
            )
            return order

        except Exception as e:
            error_msg = str(e)
            if "insufficient" in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            raise OrderRejectedError(f"Order rejected: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            self._api.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        """Check order status."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        raw = self._api.get_order_by_id(order_id)
        order = Order(
            broker_order_id=str(raw.id),
            status=self._map_status(str(raw.status)),
            filled_quantity=float(raw.filled_qty or 0),
            filled_price=float(raw.filled_avg_price or 0),
        )
        return order

    def close_all_positions(self) -> list[Order]:
        """Emergency: close all positions."""
        if not self._api:
            raise BrokerConnectionError("Not connected to Alpaca")

        try:
            self._api.close_all_positions(cancel_orders=True)
            logger.warning("ALL POSITIONS CLOSED (emergency)")
            return []
        except Exception as e:
            raise ExecutionError(f"Failed to close all positions: {e}")

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> Order:
        """Wait for an order to be filled."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.get_order_status(order_id)
            if order.is_complete:
                return order
            time.sleep(1)

        logger.warning(f"Order {order_id} not filled within {timeout}s")
        return self.get_order_status(order_id)

    @staticmethod
    def _map_status(alpaca_status: str) -> OrderStatus:
        """Map Alpaca status to our OrderStatus enum."""
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL_FILL,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.SUBMITTED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)
