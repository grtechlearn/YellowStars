"""
Backtesting Engine - Simulates strategy execution on historical data.

Features:
- Event-driven simulation (bar-by-bar)
- Realistic transaction cost modeling (commission + slippage)
- Position sizing with fractional shares
- Automatic benchmarking against Buy & Hold
- Multi-asset support (TQQQ/SQQQ switching)
- 42+ year historical testing capability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.config.settings import BacktestSettings, StrategySettings
from yellowstars.core.models import (
    OrderSide,
    Portfolio,
    Position,
    PositionSide,
    SignalAction,
    Trade,
    Asset,
    MarketType,
)
from yellowstars.strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    """Complete backtest output."""
    # Identification
    strategy_name: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = 0.0

    # Equity curve
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Sell & Hold benchmark (SQQQ buy-and-hold)
    sell_hold_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sell_hold_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Signals
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Trade log
    trades: list[Trade] = field(default_factory=list)

    # Daily returns
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Position history
    position_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Summary metrics (populated by PerformanceMetrics)
    metrics: dict = field(default_factory=dict)

    # Configuration used
    strategy_params: dict = field(default_factory=dict)
    backtest_settings: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates bar-by-bar execution of a strategy on historical data,
    tracking portfolio value, positions, trades, and costs.

    Usage:
        engine = BacktestEngine(settings)
        result = engine.run(strategy, data)
    """

    def __init__(self, settings: Optional[BacktestSettings] = None):
        self.settings = settings or BacktestSettings()
        self._portfolio: Optional[Portfolio] = None
        self._trades: list[Trade] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._position_history: list[dict] = []
        self._current_instrument: str = ""
        self._current_position_pct: float = 0.0

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        sell_hold_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run a complete backtest.

        Args:
            strategy: The strategy to test.
            data: OHLCV data for the primary instrument (or the underlying index).
            benchmark_data: Optional OHLCV data for Buy & Hold benchmark.
            sell_hold_data: Optional OHLCV data for Sell & Hold benchmark (e.g., SQQQ).

        Returns:
            BacktestResult with equity curves, trades, and metrics.
        """
        logger.info(
            f"Starting backtest: {strategy.name} | "
            f"Data: {len(data)} bars ({data.index[0]} to {data.index[-1]}) | "
            f"Capital: ${self.settings.initial_capital:,.0f}"
        )

        # Initialize portfolio
        self._portfolio = Portfolio(
            cash=self.settings.initial_capital,
            initial_capital=self.settings.initial_capital,
            total_equity=self.settings.initial_capital,
        )
        self._trades = []
        self._equity_history = []
        self._position_history = []
        self._current_instrument = ""
        self._current_position_pct = 0.0

        # Generate signals
        signals_df = strategy.generate_signals(data)

        # Simulate bar by bar
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            timestamp = signals_df.index[i]
            price = row["close"]

            # Get target position
            target_pct = row.get("position_pct", 0.0)
            target_instrument = row.get("instrument", strategy.settings.long_asset)
            signal = row.get("signal", SignalAction.HOLD.value)

            # Execute rebalancing
            self._rebalance(
                timestamp=timestamp,
                price=price,
                target_pct=target_pct,
                target_instrument=target_instrument,
                signal=signal,
            )

            # Update portfolio value
            self._update_portfolio_value(price)

            # Record history
            self._equity_history.append((timestamp, self._portfolio.total_value))
            self._position_history.append({
                "timestamp": timestamp,
                "cash": self._portfolio.cash,
                "position_value": self._portfolio.positions_value,
                "total_equity": self._portfolio.total_value,
                "position_pct": self._current_position_pct,
                "instrument": self._current_instrument,
            })

        # Build equity curve
        equity_curve = pd.Series(
            {ts: eq for ts, eq in self._equity_history},
            name="equity",
        )
        equity_curve.index = pd.DatetimeIndex(equity_curve.index)

        # Build benchmark curve (Buy & Hold)
        if benchmark_data is not None and not benchmark_data.empty:
            bench_returns = benchmark_data["close"].pct_change().fillna(0)
            benchmark_curve = (1 + bench_returns).cumprod() * self.settings.initial_capital
        else:
            # Use strategy data as benchmark (Buy & Hold on the underlying)
            bench_returns = data["close"].pct_change().fillna(0)
            benchmark_curve = (1 + bench_returns).cumprod() * self.settings.initial_capital

        # Build sell-and-hold curve (SQQQ Buy & Hold = "Sell & Hold")
        if sell_hold_data is not None and not sell_hold_data.empty:
            sh_returns = sell_hold_data["close"].pct_change().fillna(0)
            sell_hold_curve = (1 + sh_returns).cumprod() * self.settings.initial_capital
            sell_hold_returns_series = sell_hold_curve.pct_change().fillna(0)
        else:
            sell_hold_curve = pd.Series(dtype=float)
            sell_hold_returns_series = pd.Series(dtype=float)

        # Build result
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=signals_df.index[0].date() if len(signals_df) > 0 else None,
            end_date=signals_df.index[-1].date() if len(signals_df) > 0 else None,
            initial_capital=self.settings.initial_capital,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            signals_df=signals_df,
            trades=self._trades,
            daily_returns=equity_curve.pct_change().fillna(0),
            benchmark_returns=benchmark_curve.pct_change().fillna(0),
            sell_hold_curve=sell_hold_curve,
            sell_hold_returns=sell_hold_returns_series,
            position_history=pd.DataFrame(self._position_history),
            strategy_params=strategy.get_params(),
            backtest_settings={
                "initial_capital": self.settings.initial_capital,
                "commission_per_trade": self.settings.commission_per_trade,
                "slippage_pct": self.settings.slippage_pct,
                "benchmark": self.settings.benchmark_symbol,
            },
        )

        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Final equity: ${equity_curve.iloc[-1]:,.0f} | "
            f"Total return: {(equity_curve.iloc[-1] / self.settings.initial_capital - 1) * 100:.1f}% | "
            f"Trades: {len(self._trades)}"
        )

        return result

    def _rebalance(
        self,
        timestamp: datetime,
        price: float,
        target_pct: float,
        target_instrument: str,
        signal: str,
    ) -> None:
        """Rebalance portfolio to match target allocation.

        Handles:
        - Instrument switching (TQQQ -> SQQQ)
        - Position sizing (increase/decrease)
        - Transaction costs
        """
        portfolio = self._portfolio
        current_equity = portfolio.total_value

        # Calculate target position value
        target_value = current_equity * target_pct
        current_value = portfolio.positions_value

        # Check if instrument changed
        instrument_changed = (
            target_instrument != self._current_instrument
            and self._current_instrument != ""
            and target_pct > 0
        )

        # If instrument changed, close current position first
        if instrument_changed and portfolio.positions:
            self._close_all_positions(timestamp, price)
            current_value = 0.0

        # Calculate required trade
        value_diff = target_value - current_value

        # Minimum rebalance threshold (avoid tiny trades)
        min_trade_value = current_equity * 0.01  # 1% of equity

        if abs(value_diff) < min_trade_value:
            self._current_position_pct = target_pct
            return

        if value_diff > 0:
            # Need to buy more
            self._execute_buy(timestamp, target_instrument, value_diff, price)
        elif value_diff < 0:
            # Need to sell
            self._execute_sell(timestamp, abs(value_diff), price)

        self._current_instrument = target_instrument if target_pct > 0 else ""
        self._current_position_pct = target_pct

    def _execute_buy(
        self, timestamp: datetime, instrument: str, value: float, price: float
    ) -> None:
        """Execute a buy order."""
        portfolio = self._portfolio

        # Apply slippage (buy at slightly higher price)
        slippage_cost = value * (self.settings.slippage_pct / 100)
        commission = self.settings.commission_per_trade

        total_cost = value + slippage_cost + commission

        if total_cost > portfolio.cash:
            # Reduce to available cash
            total_cost = portfolio.cash
            value = total_cost - slippage_cost - commission
            if value <= 0:
                return

        quantity = value / price

        # Update portfolio
        portfolio.cash -= total_cost
        portfolio.total_commission_paid += commission

        # Update or create position
        existing = portfolio.get_position(instrument)
        if existing:
            # Average in
            total_qty = existing.quantity + quantity
            existing.avg_entry_price = (
                (existing.avg_entry_price * existing.quantity + price * quantity) / total_qty
            )
            existing.quantity = total_qty
            existing.current_price = price
            existing.last_updated = timestamp
        else:
            asset = Asset(
                symbol=instrument,
                name=instrument,
                market_type=MarketType.US_EQUITY,
            )
            new_pos = Position(
                asset=asset,
                side=PositionSide.LONG,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                opened_at=timestamp,
                last_updated=timestamp,
            )
            portfolio.positions.append(new_pos)

    def _execute_sell(
        self, timestamp: datetime, value: float, price: float
    ) -> None:
        """Execute a sell order."""
        if not self._portfolio.positions:
            return

        position = self._portfolio.positions[0]  # Current active position
        quantity_to_sell = min(value / price, position.quantity)

        # Apply slippage (sell at slightly lower price)
        slippage_cost = quantity_to_sell * price * (self.settings.slippage_pct / 100)
        commission = self.settings.commission_per_trade

        proceeds = (quantity_to_sell * price) - slippage_cost - commission
        self._portfolio.cash += proceeds
        self._portfolio.total_commission_paid += commission

        # Record trade
        pnl = (price - position.avg_entry_price) * quantity_to_sell - slippage_cost - commission
        trade = Trade(
            asset=position.asset,
            side=OrderSide.SELL,
            entry_price=position.avg_entry_price,
            exit_price=price,
            quantity=quantity_to_sell,
            entry_time=position.opened_at,
            exit_time=timestamp,
            pnl=pnl,
            pnl_pct=(price / position.avg_entry_price - 1) * 100 if position.avg_entry_price else 0,
            commission=commission,
            net_pnl=pnl,
            holding_period_days=(timestamp - position.opened_at).days if position.opened_at else 0,
            strategy_name=self._current_instrument,
        )
        self._trades.append(trade)

        # Update position
        position.quantity -= quantity_to_sell
        if position.quantity <= 0.001:  # Essentially closed
            self._portfolio.positions.remove(position)

    def _close_all_positions(self, timestamp: datetime, price: float) -> None:
        """Close all open positions."""
        for position in list(self._portfolio.positions):
            sell_value = position.quantity * price
            self._execute_sell(timestamp, sell_value, price)

    def _update_portfolio_value(self, price: float) -> None:
        """Mark positions to market and update total equity."""
        for position in self._portfolio.positions:
            position.current_price = price
            position.unrealized_pnl = (
                (price - position.avg_entry_price) * position.quantity
            )
        self._portfolio.total_equity = self._portfolio.total_value
