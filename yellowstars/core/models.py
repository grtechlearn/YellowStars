"""
Core data models for the YellowStars trading platform.
All domain objects used across the system are defined here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional


# ============================================================
# Enums
# ============================================================

class MarketType(str, Enum):
    """Supported market types."""
    US_EQUITY = "us_equity"
    INTERNATIONAL_EQUITY = "international_equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"


class OrderSide(str, Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalAction(str, Enum):
    """Strategy signal actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"        # Reduce position size
    INCREASE = "increase"    # Increase position size
    FLIP = "flip"            # Flip from long to short or vice versa
    CLOSE = "close"          # Close all positions


class TimeFrame(str, Enum):
    """Data timeframes."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class TradingMode(str, Enum):
    """System operating mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


# ============================================================
# Data Models
# ============================================================

@dataclass
class OHLCV:
    """Single price bar (Open, High, Low, Close, Volume)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: TimeFrame = TimeFrame.DAILY

    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def bar_range(self) -> float:
        return self.high - self.low


@dataclass
class Asset:
    """Tradeable asset definition."""
    symbol: str
    name: str
    market_type: MarketType
    exchange: str = ""
    currency: str = "USD"
    leverage_factor: float = 1.0  # e.g., 3.0 for TQQQ
    underlying_symbol: str = ""    # e.g., NDX for TQQQ
    is_inverse: bool = False       # True for SQQQ
    min_quantity: float = 1.0
    tick_size: float = 0.01
    metadata: dict = field(default_factory=dict)


@dataclass
class Signal:
    """Strategy signal output."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    asset: Optional[Asset] = None
    action: SignalAction = SignalAction.HOLD
    target_position_pct: float = 0.0   # Target portfolio allocation (0.0 to 1.0)
    confidence: float = 0.0            # Strategy confidence (0.0 to 1.0)
    strategy_name: str = ""
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    """Trade order."""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    asset: Optional[Asset] = None
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    broker_order_id: str = ""
    signal: Optional[Signal] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.EXPIRED)

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        base = self.filled_quantity * self.filled_price
        return base + self.commission + self.slippage


@dataclass
class Position:
    """Current position in an asset."""
    asset: Optional[Asset] = None
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100.0


@dataclass
class Portfolio:
    """Portfolio state at a point in time."""
    timestamp: datetime = field(default_factory=datetime.now)
    cash: float = 0.0
    positions: list[Position] = field(default_factory=list)
    total_equity: float = 0.0
    initial_capital: float = 0.0
    total_commission_paid: float = 0.0
    total_tax_paid: float = 0.0

    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100.0

    def get_position(self, symbol: str) -> Optional[Position]:
        for p in self.positions:
            if p.asset and p.asset.symbol == symbol:
                return p
        return None


@dataclass
class Trade:
    """Completed trade record for reporting."""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    asset: Optional[Asset] = None
    side: OrderSide = OrderSide.BUY
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    net_pnl: float = 0.0
    holding_period_days: int = 0
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_short_term(self) -> bool:
        """Short-term for tax purposes (held < 1 year)."""
        return self.holding_period_days < 365


# ============================================================
# Configuration Models
# ============================================================

@dataclass
class BrokerConfig:
    """Broker connection configuration."""
    name: str = "alpaca"
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    paper_trading: bool = True
    max_order_size: float = 100000.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0


@dataclass
class DataProviderConfig:
    """Data provider configuration."""
    name: str = "polygon"
    api_key: str = ""
    base_url: str = ""
    rate_limit_per_minute: int = 5
    cache_enabled: bool = True
    cache_directory: str = "data/cache"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_portfolio_drawdown_pct: float = 30.0    # Circuit breaker
    max_single_position_pct: float = 100.0      # Max allocation to single asset
    max_daily_loss_pct: float = 10.0            # Max daily loss before halting
    min_cash_reserve_pct: float = 0.0           # Minimum cash to keep
    max_leverage: float = 1.0                    # Max leverage allowed
    stop_loss_pct: Optional[float] = None       # Per-position stop loss


@dataclass
class TaxConfig:
    """Tax calculation configuration."""
    country: str = "US"
    short_term_rate: float = 0.37       # Federal short-term capital gains
    long_term_rate: float = 0.20        # Federal long-term capital gains
    state_tax_rate: float = 0.0         # State tax rate
    wash_sale_rule: bool = True         # Apply wash sale rule (US)
    tax_loss_harvesting: bool = False   # Enable tax loss harvesting


@dataclass
class AlertConfig:
    """Alerting configuration."""
    enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""
    discord_webhook_url: str = ""
    email_smtp_server: str = ""
    email_from: str = ""
    email_to: str = ""
    email_password: str = ""
