"""
Tax Calculator - Comprehensive tax computation for trading profits.

Handles:
- Short-term vs long-term capital gains (US)
- Wash sale rule tracking
- State tax estimation
- Brokerage fees and transaction charges
- Quarterly estimated tax calculations
- Annual tax summary for filing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from yellowstars.core.models import TaxConfig, Trade


@dataclass
class TaxLot:
    """Individual tax lot for tracking cost basis."""
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: date
    sale_price: float = 0.0
    sale_date: Optional[date] = None
    is_wash_sale: bool = False
    disallowed_loss: float = 0.0


@dataclass
class TaxSummary:
    """Tax summary for a period."""
    period_start: date
    period_end: date
    total_proceeds: float = 0.0
    total_cost_basis: float = 0.0
    short_term_gains: float = 0.0
    short_term_losses: float = 0.0
    long_term_gains: float = 0.0
    long_term_losses: float = 0.0
    net_short_term: float = 0.0
    net_long_term: float = 0.0
    total_net_gain: float = 0.0
    wash_sale_disallowed: float = 0.0
    estimated_federal_tax: float = 0.0
    estimated_state_tax: float = 0.0
    total_estimated_tax: float = 0.0
    total_commissions: float = 0.0
    net_after_tax: float = 0.0


class TaxCalculator:
    """Calculate taxes on trading profits.

    Supports US tax rules by default (configurable for other countries).
    """

    def __init__(self, config: TaxConfig):
        self.config = config
        self._tax_lots: list[TaxLot] = []

    def calculate_trade_tax(self, trade: Trade) -> dict:
        """Calculate tax implications for a single trade.

        Returns:
            Dictionary with tax details for the trade.
        """
        if trade.entry_time is None or trade.exit_time is None:
            return {"error": "Trade missing entry/exit times"}

        holding_days = (trade.exit_time - trade.entry_time).days
        is_long_term = holding_days >= 365
        gain_loss = trade.net_pnl

        if gain_loss > 0:
            tax_rate = (
                self.config.long_term_rate if is_long_term
                else self.config.short_term_rate
            )
        else:
            tax_rate = 0.0  # Losses are deductible, not taxed

        federal_tax = max(0, gain_loss * tax_rate)
        state_tax = max(0, gain_loss * self.config.state_tax_rate)
        total_tax = federal_tax + state_tax

        return {
            "trade_id": trade.trade_id,
            "symbol": trade.asset.symbol if trade.asset else "",
            "gain_loss": round(gain_loss, 2),
            "holding_days": holding_days,
            "is_long_term": is_long_term,
            "federal_tax_rate": tax_rate,
            "state_tax_rate": self.config.state_tax_rate,
            "federal_tax": round(federal_tax, 2),
            "state_tax": round(state_tax, 2),
            "total_tax": round(total_tax, 2),
            "net_after_tax": round(gain_loss - total_tax, 2),
            "commission": round(trade.commission, 2),
        }

    def calculate_period_summary(
        self,
        trades: list[Trade],
        period_start: date,
        period_end: date,
    ) -> TaxSummary:
        """Calculate tax summary for a period (quarterly/annual).

        Args:
            trades: All trades in the period.
            period_start: Start of the tax period.
            period_end: End of the tax period.

        Returns:
            TaxSummary with detailed breakdown.
        """
        summary = TaxSummary(
            period_start=period_start,
            period_end=period_end,
        )

        for trade in trades:
            if trade.entry_time is None or trade.exit_time is None:
                continue

            # Check if trade falls in period
            trade_exit = trade.exit_time.date() if isinstance(trade.exit_time, datetime) else trade.exit_time
            if not (period_start <= trade_exit <= period_end):
                continue

            proceeds = trade.exit_price * trade.quantity
            cost_basis = trade.entry_price * trade.quantity

            summary.total_proceeds += proceeds
            summary.total_cost_basis += cost_basis
            summary.total_commissions += trade.commission

            holding_days = (trade.exit_time - trade.entry_time).days
            gain = trade.net_pnl

            if holding_days >= 365:
                # Long-term
                if gain > 0:
                    summary.long_term_gains += gain
                else:
                    summary.long_term_losses += abs(gain)
            else:
                # Short-term
                if gain > 0:
                    summary.short_term_gains += gain
                else:
                    summary.short_term_losses += abs(gain)

        # Check wash sales
        if self.config.wash_sale_rule:
            wash_disallowed = self._check_wash_sales(trades, period_start, period_end)
            summary.wash_sale_disallowed = wash_disallowed

        # Net calculations
        summary.net_short_term = summary.short_term_gains - summary.short_term_losses
        summary.net_long_term = summary.long_term_gains - summary.long_term_losses
        summary.total_net_gain = summary.net_short_term + summary.net_long_term

        # Estimated taxes
        if summary.net_short_term > 0:
            summary.estimated_federal_tax += summary.net_short_term * self.config.short_term_rate
        if summary.net_long_term > 0:
            summary.estimated_federal_tax += summary.net_long_term * self.config.long_term_rate

        # Capital loss deduction limit ($3,000/year for US)
        if summary.total_net_gain < 0:
            deductible = max(summary.total_net_gain, -3000)
            summary.estimated_federal_tax = deductible * self.config.short_term_rate

        # State tax
        if summary.total_net_gain > 0:
            summary.estimated_state_tax = summary.total_net_gain * self.config.state_tax_rate

        summary.total_estimated_tax = (
            summary.estimated_federal_tax + summary.estimated_state_tax
        )
        summary.net_after_tax = summary.total_net_gain - summary.total_estimated_tax

        # Round everything
        for attr in vars(summary):
            val = getattr(summary, attr)
            if isinstance(val, float):
                setattr(summary, attr, round(val, 2))

        return summary

    def calculate_quarterly_estimates(
        self,
        trades: list[Trade],
        year: int,
    ) -> list[TaxSummary]:
        """Calculate quarterly estimated tax payments.

        US estimated taxes due: Apr 15, Jun 15, Sep 15, Jan 15 (next year).
        """
        quarters = [
            (date(year, 1, 1), date(year, 3, 31)),
            (date(year, 4, 1), date(year, 6, 30)),
            (date(year, 7, 1), date(year, 9, 30)),
            (date(year, 10, 1), date(year, 12, 31)),
        ]

        quarterly_summaries = []
        for q_start, q_end in quarters:
            summary = self.calculate_period_summary(trades, q_start, q_end)
            quarterly_summaries.append(summary)

        return quarterly_summaries

    def calculate_annual_summary(
        self,
        trades: list[Trade],
        year: int,
    ) -> TaxSummary:
        """Calculate annual tax summary for filing."""
        return self.calculate_period_summary(
            trades,
            date(year, 1, 1),
            date(year, 12, 31),
        )

    def _check_wash_sales(
        self,
        trades: list[Trade],
        period_start: date,
        period_end: date,
    ) -> float:
        """Check for wash sale rule violations.

        A wash sale occurs when you sell a security at a loss and buy
        a substantially identical security within 30 days before or after.
        """
        disallowed = 0.0

        # Get losing trades
        losing_trades = [
            t for t in trades
            if t.net_pnl < 0 and t.exit_time is not None
        ]

        for loss_trade in losing_trades:
            if loss_trade.exit_time is None or loss_trade.asset is None:
                continue

            loss_date = (
                loss_trade.exit_time.date()
                if isinstance(loss_trade.exit_time, datetime)
                else loss_trade.exit_time
            )
            symbol = loss_trade.asset.symbol

            # Check if same symbol was bought within 30 days
            wash_window_start = loss_date - timedelta(days=30)
            wash_window_end = loss_date + timedelta(days=30)

            for other_trade in trades:
                if other_trade.trade_id == loss_trade.trade_id:
                    continue
                if other_trade.asset is None or other_trade.entry_time is None:
                    continue
                if other_trade.asset.symbol != symbol:
                    continue

                buy_date = (
                    other_trade.entry_time.date()
                    if isinstance(other_trade.entry_time, datetime)
                    else other_trade.entry_time
                )

                if wash_window_start <= buy_date <= wash_window_end:
                    disallowed += abs(loss_trade.net_pnl)
                    logger.warning(
                        f"Wash sale detected: {symbol} loss of "
                        f"${abs(loss_trade.net_pnl):.2f} on {loss_date} "
                        f"(repurchased on {buy_date})"
                    )
                    break

        return disallowed

    def generate_tax_report_text(self, summary: TaxSummary) -> str:
        """Generate human-readable tax report."""
        lines = [
            "=" * 60,
            "TAX SUMMARY REPORT",
            f"Period: {summary.period_start} to {summary.period_end}",
            "=" * 60,
            "",
            f"Total Proceeds:           ${summary.total_proceeds:>12,.2f}",
            f"Total Cost Basis:         ${summary.total_cost_basis:>12,.2f}",
            f"Total Commissions:        ${summary.total_commissions:>12,.2f}",
            "",
            "--- Capital Gains ---",
            f"Short-Term Gains:         ${summary.short_term_gains:>12,.2f}",
            f"Short-Term Losses:       (${summary.short_term_losses:>11,.2f})",
            f"Net Short-Term:           ${summary.net_short_term:>12,.2f}",
            "",
            f"Long-Term Gains:          ${summary.long_term_gains:>12,.2f}",
            f"Long-Term Losses:        (${summary.long_term_losses:>11,.2f})",
            f"Net Long-Term:            ${summary.net_long_term:>12,.2f}",
            "",
            f"Total Net Gain/Loss:      ${summary.total_net_gain:>12,.2f}",
            "",
            "--- Estimated Taxes ---",
            f"Federal Tax:              ${summary.estimated_federal_tax:>12,.2f}",
            f"State Tax:                ${summary.estimated_state_tax:>12,.2f}",
            f"Total Tax:                ${summary.total_estimated_tax:>12,.2f}",
            "",
            f"Net After Tax:            ${summary.net_after_tax:>12,.2f}",
        ]

        if summary.wash_sale_disallowed > 0:
            lines.extend([
                "",
                f"⚠ Wash Sale Disallowed:   ${summary.wash_sale_disallowed:>12,.2f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
