"""
Daily Runner - Autonomous daily trading workflow.

Executes the complete daily lifecycle:
1. Pre-market: Health checks, data refresh, backtest validation
2. Market hours: Monitor positions, execute at EOD
3. Post-market: P&L reporting, tax tracking, next-day prep

Runs as a scheduled process that activates on trading days.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

from loguru import logger

from yellowstars.alerting.notifier import AlertLevel, Notifier
from yellowstars.backtest.engine import BacktestEngine
from yellowstars.backtest.metrics import PerformanceMetrics
from yellowstars.config.settings import Settings
from yellowstars.core.exceptions import (
    CircuitBreakerTriggered,
    ExecutionError,
    YellowStarsError,
)
from yellowstars.core.models import (
    OrderSide,
    Portfolio,
    SignalAction,
    Trade,
    TradingMode,
)
from yellowstars.data.manager import DataManager
from yellowstars.execution.broker_base import BaseBroker
from yellowstars.execution.risk_manager import RiskManager
from yellowstars.reporting.report_generator import ReportGenerator
from yellowstars.reporting.excel_exporter import ExcelExporter
from yellowstars.reporting.tax_calculator import TaxCalculator
from yellowstars.strategies.base import BaseStrategy


class DailyRunner:
    """Orchestrates the complete daily trading workflow.

    Lifecycle:
    ┌─────────────────────────────────────────────────────────┐
    │  6:00 AM  │ Pre-Market: Health check, data refresh      │
    │  6:30 AM  │ Run backtest on latest data                  │
    │  7:00 AM  │ Generate signals for today                   │
    │  3:45 PM  │ Execute trades (15 min before close)         │
    │  4:00 PM  │ Market close                                 │
    │  4:30 PM  │ Post-market: P&L, tax, reports               │
    │  5:00 PM  │ Send daily summary notification              │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        settings: Settings,
        data_manager: DataManager,
        strategy: BaseStrategy,
        broker: Optional[BaseBroker] = None,
        notifier: Optional[Notifier] = None,
    ):
        self.settings = settings
        self.data_manager = data_manager
        self.strategy = strategy
        self.broker = broker
        self.notifier = notifier or Notifier(settings.alerts)
        self.risk_manager = RiskManager(settings.risk)
        self.report_generator = ReportGenerator(settings.reports_directory)
        self.excel_exporter = ExcelExporter(output_dir=settings.data_directory)
        self.tax_calculator = TaxCalculator(settings.tax)

        self._today_trades: list[Trade] = []
        self._today_signals = None
        self._is_running = False

    def run_daily_cycle(self) -> dict:
        """Execute the complete daily trading cycle.

        Returns:
            Dictionary with daily cycle results.
        """
        cycle_start = datetime.now()
        today = date.today()
        results = {
            "date": today.isoformat(),
            "status": "unknown",
            "phases": {},
        }

        logger.info(f"{'='*60}")
        logger.info(f"DAILY CYCLE START: {today}")
        logger.info(f"Mode: {self.settings.trading_mode.value}")
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"{'='*60}")

        self._is_running = True
        self._today_trades = []

        try:
            # Phase 1: Pre-market
            results["phases"]["pre_market"] = self._pre_market_phase()

            # Phase 2: Strategy analysis
            results["phases"]["analysis"] = self._analysis_phase()

            # Phase 3: Execution
            if self.settings.trading_mode != TradingMode.BACKTEST:
                results["phases"]["execution"] = self._execution_phase()

            # Phase 4: Post-market
            results["phases"]["post_market"] = self._post_market_phase()

            results["status"] = "success"

        except CircuitBreakerTriggered as e:
            logger.critical(f"Circuit breaker: {e}")
            self.notifier.send_circuit_breaker_alert(str(e))
            results["status"] = "circuit_breaker"
            results["error"] = str(e)

        except YellowStarsError as e:
            logger.error(f"Trading error: {e}")
            self.notifier.send_error_alert(str(e), "DailyRunner")
            results["status"] = "error"
            results["error"] = str(e)

        except Exception as e:
            logger.exception(f"Unexpected error in daily cycle: {e}")
            self.notifier.send_error_alert(str(e), "DailyRunner")
            results["status"] = "fatal_error"
            results["error"] = str(e)

        finally:
            self._is_running = False
            elapsed = (datetime.now() - cycle_start).total_seconds()
            results["elapsed_seconds"] = round(elapsed, 1)
            logger.info(f"Daily cycle complete in {elapsed:.1f}s: {results['status']}")

        return results

    def _pre_market_phase(self) -> dict:
        """Pre-market phase: health checks and data refresh."""
        logger.info("--- PRE-MARKET PHASE ---")
        phase_result = {"status": "ok", "checks": {}}

        # 1. Check data provider connectivity
        try:
            self.data_manager.initialize()
            phase_result["checks"]["data_provider"] = "connected"
        except Exception as e:
            phase_result["checks"]["data_provider"] = f"error: {e}"
            logger.error(f"Data provider check failed: {e}")

        # 2. Check broker connectivity (if live/paper)
        if self.broker and self.settings.trading_mode != TradingMode.BACKTEST:
            try:
                self.broker.connect()
                account = self.broker.get_account()
                self.risk_manager.reset_daily(account["equity"])
                phase_result["checks"]["broker"] = "connected"
                phase_result["checks"]["equity"] = account["equity"]
                phase_result["checks"]["buying_power"] = account["buying_power"]
            except Exception as e:
                phase_result["checks"]["broker"] = f"error: {e}"
                phase_result["status"] = "degraded"

        # 3. Refresh market data
        try:
            strat_settings = self.settings.strategies[0] if self.settings.strategies else None
            underlying = strat_settings.underlying_index if strat_settings else "NDX"

            # Fetch latest data for the underlying index
            data = self.data_manager.get_data(
                underlying,
                start_date=date(1985, 1, 1),
                end_date=date.today(),
            )
            phase_result["checks"]["data_bars"] = len(data)
            phase_result["checks"]["data_latest"] = str(data.index[-1].date()) if len(data) > 0 else "none"

        except Exception as e:
            phase_result["checks"]["data"] = f"error: {e}"
            phase_result["status"] = "degraded"

        self.notifier.send(
            f"Pre-market checks: {phase_result['status']}",
            AlertLevel.INFO if phase_result["status"] == "ok" else AlertLevel.WARNING,
        )

        return phase_result

    def _analysis_phase(self) -> dict:
        """Analysis phase: run backtest and generate today's signals."""
        logger.info("--- ANALYSIS PHASE ---")
        phase_result = {"status": "ok"}

        strat_settings = self.settings.strategies[0] if self.settings.strategies else None
        underlying = strat_settings.underlying_index if strat_settings else "NDX"

        # Get historical data
        data = self.data_manager.get_data(
            underlying,
            start_date=date(1985, 1, 1),
            end_date=date.today(),
        )

        if data.empty:
            phase_result["status"] = "no_data"
            return phase_result

        # Run backtest on full history
        bt_engine = BacktestEngine(self.settings.backtest)
        bt_result = bt_engine.run(self.strategy, data)

        # Generate metrics
        metrics = PerformanceMetrics(
            bt_result.equity_curve,
            bt_result.benchmark_curve,
            bt_result.trades,
        )
        all_metrics = metrics.compute_all()

        phase_result["backtest"] = {
            "total_return": all_metrics.get("return_metrics", {}).get("total_return_pct", 0),
            "cagr": all_metrics.get("return_metrics", {}).get("cagr_pct", 0),
            "sharpe": all_metrics.get("risk_metrics", {}).get("sharpe_ratio", 0),
            "max_drawdown": all_metrics.get("drawdown_metrics", {}).get("max_drawdown_pct", 0),
        }

        # Get today's signal (last row)
        signals_df = bt_result.signals_df
        if len(signals_df) > 0:
            today_signal = signals_df.iloc[-1]
            self._today_signals = today_signal
            phase_result["today_signal"] = {
                "action": today_signal.get("signal", "hold"),
                "position_pct": round(today_signal.get("position_pct", 0), 4),
                "instrument": today_signal.get("instrument", ""),
            }

            logger.info(
                f"Today's signal: {phase_result['today_signal']['action']} | "
                f"Position: {phase_result['today_signal']['position_pct']:.1%} | "
                f"Instrument: {phase_result['today_signal']['instrument']}"
            )

        return phase_result

    def _execution_phase(self) -> dict:
        """Execution phase: execute trades based on signals."""
        logger.info("--- EXECUTION PHASE ---")
        phase_result = {"status": "ok", "orders": []}

        if self._today_signals is None:
            phase_result["status"] = "no_signals"
            return phase_result

        if not self.broker or not self.broker.is_connected():
            phase_result["status"] = "broker_not_connected"
            return phase_result

        # Get current portfolio
        portfolio = self.broker.get_portfolio()
        current_equity = portfolio.total_value

        # Get target from signals
        target_pct = self._today_signals.get("position_pct", 0)
        target_instrument = self._today_signals.get("instrument", "")
        signal_action = self._today_signals.get("signal", "hold")

        if signal_action == SignalAction.HOLD.value and abs(target_pct - self._current_position_allocation(portfolio)) < 0.05:
            logger.info("No rebalancing needed. Holding current position.")
            phase_result["status"] = "hold"
            return phase_result

        # Calculate target position
        target_value = current_equity * target_pct
        current_positions = self.broker.get_positions()

        # Determine current allocation
        current_value = sum(p.market_value for p in current_positions)
        value_diff = target_value - current_value

        logger.info(
            f"Rebalancing: current=${current_value:,.2f} -> "
            f"target=${target_value:,.2f} ({target_pct:.1%} of ${current_equity:,.2f})"
        )

        # Execute
        try:
            if abs(value_diff) > current_equity * 0.01:  # >1% change
                if value_diff > 0:
                    # Buy
                    quantity = value_diff / self.data_manager.get_latest_price(target_instrument)
                    order = self.broker.create_market_order(
                        target_instrument,
                        OrderSide.BUY,
                        round(quantity, 2),
                    )
                    phase_result["orders"].append({
                        "action": "buy",
                        "symbol": target_instrument,
                        "quantity": round(quantity, 2),
                        "order_id": order.broker_order_id,
                    })
                else:
                    # Sell
                    if current_positions:
                        pos = current_positions[0]
                        sell_qty = min(abs(value_diff) / pos.current_price, pos.quantity)
                        order = self.broker.create_market_order(
                            pos.asset.symbol,
                            OrderSide.SELL,
                            round(sell_qty, 2),
                        )
                        phase_result["orders"].append({
                            "action": "sell",
                            "symbol": pos.asset.symbol,
                            "quantity": round(sell_qty, 2),
                            "order_id": order.broker_order_id,
                        })

                self.notifier.send_trade_alert(
                    "BUY" if value_diff > 0 else "SELL",
                    target_instrument,
                    abs(value_diff / (self.data_manager.get_latest_price(target_instrument) or 1)),
                    self.data_manager.get_latest_price(target_instrument) or 0,
                )

        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.notifier.send_error_alert(str(e), "Execution")
            phase_result["status"] = "execution_error"
            phase_result["error"] = str(e)

        return phase_result

    def _post_market_phase(self) -> dict:
        """Post-market phase: reporting and analysis."""
        logger.info("--- POST-MARKET PHASE ---")
        phase_result = {"status": "ok"}

        # Get final portfolio state
        if self.broker and self.broker.is_connected():
            try:
                portfolio = self.broker.get_portfolio()
                risk_state = self.risk_manager.update_pnl(portfolio.total_value)

                # Generate daily report
                daily_report = self.report_generator.generate_daily_report(
                    portfolio, self._today_trades
                )
                phase_result["daily_report"] = daily_report

                # Send daily summary
                positions_data = [
                    {
                        "symbol": p.asset.symbol if p.asset else "?",
                        "pnl_pct": p.pnl_pct,
                    }
                    for p in portfolio.positions
                ]
                self.notifier.send_daily_summary(
                    equity=portfolio.total_value,
                    daily_pnl=risk_state["daily_pnl"],
                    daily_return=risk_state["daily_return_pct"],
                    positions=positions_data,
                )

                phase_result["risk_state"] = risk_state

            except Exception as e:
                logger.error(f"Post-market reporting error: {e}")
                phase_result["status"] = "error"
                phase_result["error"] = str(e)

        return phase_result

    def _current_position_allocation(self, portfolio: Portfolio) -> float:
        """Calculate current position as fraction of equity."""
        if portfolio.total_value <= 0:
            return 0.0
        return portfolio.positions_value / portfolio.total_value

    def run_backtest_only(self) -> dict:
        """Run just the backtest phase (no execution).

        Useful for validation and strategy development.
        """
        logger.info("Running backtest-only mode...")
        self.data_manager.initialize()

        strat_settings = self.settings.strategies[0] if self.settings.strategies else None
        underlying = strat_settings.underlying_index if strat_settings else "NDX"

        data = self.data_manager.get_data(
            underlying,
            start_date=date(1985, 1, 1),
            end_date=date.today(),
        )

        if data.empty:
            return {"status": "no_data"}

        bt_engine = BacktestEngine(self.settings.backtest)
        result = bt_engine.run(self.strategy, data)

        # Generate full report (JSON/HTML)
        metrics = self.report_generator.generate_backtest_report(result)

        # Export to Excel in LocalData/
        try:
            excel_path = self.excel_exporter.export_full_backtest(result)
            self.excel_exporter.export_history_data(data, underlying)
            logger.info(f"Excel reports saved: {excel_path}")
        except Exception as e:
            logger.warning(f"Excel export failed: {e}")

        return {
            "status": "success",
            "metrics": metrics,
            "equity_final": round(result.equity_curve.iloc[-1], 2),
            "total_trades": len(result.trades),
        }
