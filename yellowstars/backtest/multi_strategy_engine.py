"""
Multi-Strategy Backtesting Engine.

Runs multiple strategies against the same dataset, tracks order execution
for each strategy independently, and produces consolidated results.

Features:
- Simulate each strategy independently on the same OHLCV data
- Track all orders (BUY/SELL) with timestamps, prices, quantities
- Support for intraday strategies (buy-at-open, sell-at-close)
- Compute performance metrics per strategy
- Generate per-strategy order history CSVs
- Generate consolidated Excel report with comparison charts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.config.settings import BacktestSettings, StrategySettings
from yellowstars.core.models import OrderSide, Trade
from yellowstars.strategies.base import BaseStrategy
from yellowstars.backtest.metrics import PerformanceMetrics


@dataclass
class OrderRecord:
    """Record of a single order execution."""
    order_id: int = 0
    date: Optional[datetime] = None
    strategy: str = ""
    symbol: str = ""
    side: str = ""         # "BUY" or "SELL"
    quantity: float = 0.0
    price: float = 0.0
    value: float = 0.0     # quantity * price
    commission: float = 0.0
    slippage: float = 0.0
    net_value: float = 0.0  # value +/- commission +/- slippage
    portfolio_value: float = 0.0  # portfolio value after this order
    cash_after: float = 0.0
    reason: str = ""       # e.g., "initial_buy", "daily_buy", "daily_sell", "final_sell"


@dataclass
class StrategyResult:
    """Complete result for a single strategy."""
    strategy_name: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = 0.0

    # Equity curve (daily portfolio values)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # All order records
    orders: list[OrderRecord] = field(default_factory=list)

    # Trade pairs (entry + exit)
    trades: list[Trade] = field(default_factory=list)

    # Daily returns
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Performance metrics dict
    metrics: dict = field(default_factory=dict)

    # Strategy parameters
    strategy_params: dict = field(default_factory=dict)


@dataclass
class MultiStrategyResult:
    """Consolidated results from running multiple strategies."""
    strategies: list[StrategyResult] = field(default_factory=list)
    benchmark_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    symbol: str = ""
    initial_capital: float = 0.0
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class MultiStrategyEngine:
    """Run multiple strategies on the same data and compare results.

    Usage:
        engine = MultiStrategyEngine(settings)
        engine.add_strategy(BuyAndHoldStrategy())
        engine.add_strategy(BuySellClosingDayStrategy())
        result = engine.run(data)
    """

    def __init__(self, settings: Optional[BacktestSettings] = None):
        self.settings = settings or BacktestSettings()
        self._strategies: list[BaseStrategy] = []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy for backtesting."""
        self._strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")

    def run(self, data: pd.DataFrame, symbol: str = "NDX") -> MultiStrategyResult:
        """Run all registered strategies on the same data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            symbol: Symbol name for labeling.

        Returns:
            MultiStrategyResult with per-strategy results and benchmark.
        """
        if data.empty:
            logger.error("No data provided to MultiStrategyEngine")
            return MultiStrategyResult()

        logger.info(
            f"MultiStrategyEngine: running {len(self._strategies)} strategies "
            f"on {symbol} ({len(data)} bars, "
            f"{data.index[0].date()} to {data.index[-1].date()})"
        )

        multi_result = MultiStrategyResult(
            symbol=symbol,
            initial_capital=self.settings.initial_capital,
            start_date=data.index[0].date(),
            end_date=data.index[-1].date(),
        )

        # Build benchmark (Buy & Hold using close-to-close returns)
        bench_returns = data["close"].pct_change().fillna(0)
        multi_result.benchmark_curve = (
            (1 + bench_returns).cumprod() * self.settings.initial_capital
        )
        multi_result.benchmark_curve.name = f"{symbol} Buy&Hold"

        # Run each strategy
        for strategy in self._strategies:
            logger.info(f"--- Running strategy: {strategy.name} ---")
            try:
                strat_result = self._run_single_strategy(strategy, data, symbol)
                multi_result.strategies.append(strat_result)
                logger.info(
                    f"  {strategy.name}: "
                    f"Final equity=${strat_result.equity_curve.iloc[-1]:,.2f}, "
                    f"Orders={len(strat_result.orders)}, "
                    f"Trades={len(strat_result.trades)}"
                )
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                import traceback
                traceback.print_exc()

        return multi_result

    def _run_single_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
    ) -> StrategyResult:
        """Run a single strategy with full order tracking.

        Handles two modes:
        1. Standard (close-to-close): Position held overnight, equity = cash + position_value
        2. Intraday (open-to-close): Buy at open, sell at close each day, flat overnight
        """
        initial_capital = self.settings.initial_capital
        commission_per_trade = self.settings.commission_per_trade
        slippage_pct = self.settings.slippage_pct / 100.0  # Convert from bps

        # Generate signals
        signals_df = strategy.generate_signals(data)

        # Check if this is an intraday strategy
        is_intraday = "intraday" in signals_df.columns and signals_df["intraday"].iloc[0]

        if is_intraday:
            return self._run_intraday_strategy(
                strategy, signals_df, symbol, initial_capital,
                commission_per_trade, slippage_pct
            )
        else:
            return self._run_standard_strategy(
                strategy, signals_df, symbol, initial_capital,
                commission_per_trade, slippage_pct
            )

    def _run_standard_strategy(
        self,
        strategy: BaseStrategy,
        signals_df: pd.DataFrame,
        symbol: str,
        initial_capital: float,
        commission_per_trade: float,
        slippage_pct: float,
    ) -> StrategyResult:
        """Standard strategy: hold positions overnight, track close-to-close.

        For Buy-and-Hold: buy on day 1, hold forever.
        """
        cash = initial_capital
        shares = 0.0
        orders: list[OrderRecord] = []
        trades: list[Trade] = []
        equity_values = []
        order_counter = 0

        entry_price = 0.0
        entry_time = None

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            timestamp = signals_df.index[i]
            close_price = row["close"]
            target_pct = row.get("position_pct", 0.0)

            # Calculate current portfolio value
            position_value = shares * close_price
            portfolio_value = cash + position_value

            # Determine target shares
            target_value = portfolio_value * target_pct
            current_value = position_value

            # First bar: initial buy
            if i == 0 and target_pct > 0 and shares == 0:
                buy_price = close_price
                slippage_cost = buy_price * slippage_pct
                effective_price = buy_price + slippage_cost

                available = cash - commission_per_trade
                buy_shares = available / effective_price
                cost = buy_shares * effective_price + commission_per_trade

                order_counter += 1
                order = OrderRecord(
                    order_id=order_counter,
                    date=timestamp,
                    strategy=strategy.name,
                    symbol=symbol,
                    side="BUY",
                    quantity=buy_shares,
                    price=buy_price,
                    value=buy_shares * buy_price,
                    commission=commission_per_trade,
                    slippage=buy_shares * slippage_cost,
                    net_value=cost,
                    portfolio_value=portfolio_value,
                    cash_after=cash - cost,
                    reason="initial_buy",
                )
                orders.append(order)

                cash -= cost
                shares = buy_shares
                entry_price = buy_price
                entry_time = timestamp

            # Last bar: sell everything for accounting
            if i == len(signals_df) - 1 and shares > 0:
                sell_price = close_price
                slippage_cost = sell_price * slippage_pct
                effective_price = sell_price - slippage_cost

                proceeds = shares * effective_price - commission_per_trade

                order_counter += 1
                order = OrderRecord(
                    order_id=order_counter,
                    date=timestamp,
                    strategy=strategy.name,
                    symbol=symbol,
                    side="SELL",
                    quantity=shares,
                    price=sell_price,
                    value=shares * sell_price,
                    commission=commission_per_trade,
                    slippage=shares * slippage_cost,
                    net_value=proceeds,
                    portfolio_value=cash + proceeds,
                    cash_after=cash + proceeds,
                    reason="final_sell",
                )
                orders.append(order)

                # Record trade
                pnl = (sell_price - entry_price) * shares - 2 * commission_per_trade
                trade = Trade(
                    side=OrderSide.SELL,
                    entry_price=entry_price,
                    exit_price=sell_price,
                    quantity=shares,
                    entry_time=entry_time,
                    exit_time=timestamp,
                    pnl=pnl,
                    pnl_pct=((sell_price / entry_price) - 1) * 100 if entry_price > 0 else 0,
                    commission=2 * commission_per_trade,
                    net_pnl=pnl,
                    holding_period_days=(timestamp - entry_time).days if entry_time else 0,
                    strategy_name=strategy.name,
                )
                trades.append(trade)

                cash += proceeds
                shares = 0

            # Record daily equity
            position_value = shares * close_price
            equity_values.append((timestamp, cash + position_value))

        # Build equity curve
        equity_curve = pd.Series(
            {ts: eq for ts, eq in equity_values},
            name=strategy.name,
        )
        equity_curve.index = pd.DatetimeIndex(equity_curve.index)

        # Calculate metrics
        pm = PerformanceMetrics(equity_curve, trades=trades)
        metrics = pm.compute_all()

        return StrategyResult(
            strategy_name=strategy.name,
            start_date=signals_df.index[0].date(),
            end_date=signals_df.index[-1].date(),
            initial_capital=initial_capital,
            equity_curve=equity_curve,
            orders=orders,
            trades=trades,
            daily_returns=equity_curve.pct_change().fillna(0),
            metrics=metrics,
            strategy_params=strategy.get_params(),
        )

    def _run_intraday_strategy(
        self,
        strategy: BaseStrategy,
        signals_df: pd.DataFrame,
        symbol: str,
        initial_capital: float,
        commission_per_trade: float,
        slippage_pct: float,
    ) -> StrategyResult:
        """Intraday strategy: buy at open, sell at close each day.

        Equity only changes during the day (open -> close).
        Position is flat overnight.
        """
        cash = initial_capital
        orders: list[OrderRecord] = []
        trades: list[Trade] = []
        equity_values = []
        order_counter = 0

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            timestamp = signals_df.index[i]
            open_price = row["open"]
            close_price = row["close"]

            # --- BUY at open ---
            buy_price = open_price
            buy_slippage = buy_price * slippage_pct
            effective_buy_price = buy_price + buy_slippage

            available = cash - commission_per_trade
            if available <= 0:
                equity_values.append((timestamp, cash))
                continue

            shares = available / effective_buy_price
            buy_cost = shares * effective_buy_price + commission_per_trade

            order_counter += 1
            buy_order = OrderRecord(
                order_id=order_counter,
                date=timestamp,
                strategy=strategy.name,
                symbol=symbol,
                side="BUY",
                quantity=shares,
                price=buy_price,
                value=shares * buy_price,
                commission=commission_per_trade,
                slippage=shares * buy_slippage,
                net_value=buy_cost,
                portfolio_value=cash,
                cash_after=cash - buy_cost,
                reason="daily_buy",
            )
            orders.append(buy_order)

            cash -= buy_cost

            # --- SELL at close ---
            sell_price = close_price
            sell_slippage = sell_price * slippage_pct
            effective_sell_price = sell_price - sell_slippage

            proceeds = shares * effective_sell_price - commission_per_trade

            order_counter += 1
            sell_order = OrderRecord(
                order_id=order_counter,
                date=timestamp,
                strategy=strategy.name,
                symbol=symbol,
                side="SELL",
                quantity=shares,
                price=sell_price,
                value=shares * sell_price,
                commission=commission_per_trade,
                slippage=shares * sell_slippage,
                net_value=proceeds,
                portfolio_value=cash + proceeds,
                cash_after=cash + proceeds,
                reason="daily_sell",
            )
            orders.append(sell_order)

            # Record round-trip trade
            pnl = (sell_price - buy_price) * shares - 2 * commission_per_trade
            trade = Trade(
                side=OrderSide.SELL,
                entry_price=buy_price,
                exit_price=sell_price,
                quantity=shares,
                entry_time=timestamp,
                exit_time=timestamp,
                pnl=pnl,
                pnl_pct=((sell_price / buy_price) - 1) * 100 if buy_price > 0 else 0,
                commission=2 * commission_per_trade,
                net_pnl=pnl,
                holding_period_days=0,
                strategy_name=strategy.name,
            )
            trades.append(trade)

            cash += proceeds

            # End-of-day equity = cash (flat overnight)
            equity_values.append((timestamp, cash))

        # Build equity curve
        equity_curve = pd.Series(
            {ts: eq for ts, eq in equity_values},
            name=strategy.name,
        )
        equity_curve.index = pd.DatetimeIndex(equity_curve.index)

        # Calculate metrics
        pm = PerformanceMetrics(equity_curve, trades=trades)
        metrics = pm.compute_all()

        return StrategyResult(
            strategy_name=strategy.name,
            start_date=signals_df.index[0].date(),
            end_date=signals_df.index[-1].date(),
            initial_capital=initial_capital,
            equity_curve=equity_curve,
            orders=orders,
            trades=trades,
            daily_returns=equity_curve.pct_change().fillna(0),
            metrics=metrics,
            strategy_params=strategy.get_params(),
        )
