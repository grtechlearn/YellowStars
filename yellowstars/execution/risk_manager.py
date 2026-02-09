"""
Risk Manager - Pre-trade risk checks and circuit breakers.

Ensures no single trade or portfolio state violates risk limits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from yellowstars.core.exceptions import (
    CircuitBreakerTriggered,
    RiskLimitExceededError,
)
from yellowstars.core.models import Order, Portfolio, RiskConfig


class RiskManager:
    """Pre-trade and portfolio-level risk management.

    Checks:
    1. Max position size (% of portfolio)
    2. Max daily loss (circuit breaker)
    3. Max portfolio drawdown (circuit breaker)
    4. Minimum cash reserve
    5. Maximum leverage
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._circuit_breaker_active: bool = False
        self._last_reset_date: Optional[datetime] = None

    def initialize(self, current_equity: float) -> None:
        """Initialize risk manager with current portfolio state."""
        self._daily_start_equity = current_equity
        self._peak_equity = max(self._peak_equity, current_equity)
        self._daily_pnl = 0.0
        self._last_reset_date = datetime.now()
        logger.info(
            f"Risk manager initialized: equity=${current_equity:,.2f}, "
            f"peak=${self._peak_equity:,.2f}"
        )

    def reset_daily(self, current_equity: float) -> None:
        """Reset daily tracking (call at market open)."""
        self._daily_start_equity = current_equity
        self._daily_pnl = 0.0
        self._peak_equity = max(self._peak_equity, current_equity)
        self._circuit_breaker_active = False
        self._last_reset_date = datetime.now()

    def check_pre_trade(self, order: Order, portfolio: Portfolio) -> bool:
        """Run all pre-trade risk checks.

        Args:
            order: The proposed order.
            portfolio: Current portfolio state.

        Returns:
            True if order passes all checks.

        Raises:
            CircuitBreakerTriggered: If circuit breaker is active.
            RiskLimitExceededError: If order violates risk limits.
        """
        # Check circuit breaker
        if self._circuit_breaker_active:
            raise CircuitBreakerTriggered(
                "Circuit breaker active. No new trades allowed today."
            )

        current_equity = portfolio.total_value

        # Check max daily loss
        self._check_daily_loss(current_equity)

        # Check max drawdown
        self._check_drawdown(current_equity)

        # Check position size
        if order.asset:
            order_value = order.quantity * (order.limit_price or order.filled_price or 0)
            if order_value == 0:
                # Estimate from portfolio
                order_value = current_equity * 0.5  # Rough estimate

            max_position_value = current_equity * (self.config.max_single_position_pct / 100)
            if order_value > max_position_value:
                raise RiskLimitExceededError(
                    f"Order value ${order_value:,.2f} exceeds max position "
                    f"${max_position_value:,.2f} ({self.config.max_single_position_pct}%)"
                )

        # Check cash reserve
        min_cash = current_equity * (self.config.min_cash_reserve_pct / 100)
        if portfolio.cash < min_cash:
            logger.warning(
                f"Cash ${portfolio.cash:,.2f} below minimum reserve "
                f"${min_cash:,.2f}. Reducing order size."
            )

        logger.debug(f"Pre-trade checks passed for order: {order.order_id}")
        return True

    def _check_daily_loss(self, current_equity: float) -> None:
        """Check if daily loss limit is breached."""
        if self._daily_start_equity <= 0:
            return

        daily_return = (current_equity - self._daily_start_equity) / self._daily_start_equity * 100

        if daily_return < -self.config.max_daily_loss_pct:
            self._circuit_breaker_active = True
            raise CircuitBreakerTriggered(
                f"Daily loss limit breached: {daily_return:.1f}% "
                f"(limit: {self.config.max_daily_loss_pct}%). "
                f"Trading halted for the day."
            )

    def _check_drawdown(self, current_equity: float) -> None:
        """Check if max portfolio drawdown is breached."""
        if self._peak_equity <= 0:
            return

        drawdown = (current_equity - self._peak_equity) / self._peak_equity * 100

        if drawdown < -self.config.max_portfolio_drawdown_pct:
            self._circuit_breaker_active = True
            raise CircuitBreakerTriggered(
                f"Max drawdown breached: {drawdown:.1f}% "
                f"(limit: {self.config.max_portfolio_drawdown_pct}%). "
                f"Trading halted. Manual review required."
            )

    def update_pnl(self, current_equity: float) -> dict:
        """Update daily P&L tracking.

        Returns:
            Dictionary with current risk state.
        """
        self._peak_equity = max(self._peak_equity, current_equity)
        daily_pnl = current_equity - self._daily_start_equity
        daily_return = (
            (daily_pnl / self._daily_start_equity * 100)
            if self._daily_start_equity > 0
            else 0.0
        )
        drawdown = (
            ((current_equity - self._peak_equity) / self._peak_equity * 100)
            if self._peak_equity > 0
            else 0.0
        )

        return {
            "current_equity": round(current_equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return_pct": round(daily_return, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(drawdown, 2),
            "circuit_breaker_active": self._circuit_breaker_active,
            "daily_loss_limit_pct": self.config.max_daily_loss_pct,
            "max_drawdown_limit_pct": self.config.max_portfolio_drawdown_pct,
        }

    @property
    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active
