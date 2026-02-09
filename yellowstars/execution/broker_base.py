"""
Abstract base class for broker integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from yellowstars.core.models import (
    BrokerConfig,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
)


class BaseBroker(ABC):
    """Abstract broker interface."""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._connected = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker."""
        pass

    @abstractmethod
    def get_account(self) -> dict:
        """Get account information (equity, buying power, etc)."""
        pass

    @abstractmethod
    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Check status of an order."""
        pass

    @abstractmethod
    def close_all_positions(self) -> list[Order]:
        """Close all open positions (emergency)."""
        pass

    def is_connected(self) -> bool:
        return self._connected

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> Order:
        """Helper to create and submit a market order."""
        from yellowstars.core.models import Asset, MarketType

        order = Order(
            asset=Asset(symbol=symbol, name=symbol, market_type=MarketType.US_EQUITY),
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        return self.submit_order(order)
