"""Tests for the tax calculator."""

from datetime import date, datetime

import pytest

from yellowstars.core.models import Asset, MarketType, OrderSide, TaxConfig, Trade
from yellowstars.reporting.tax_calculator import TaxCalculator


def _make_trade(
    symbol: str = "TQQQ",
    entry_price: float = 100,
    exit_price: float = 110,
    quantity: float = 100,
    entry_date: str = "2024-01-15",
    exit_date: str = "2024-06-15",
    commission: float = 0,
) -> Trade:
    """Create a test trade."""
    asset = Asset(symbol=symbol, name=symbol, market_type=MarketType.US_EQUITY)
    return Trade(
        asset=asset,
        side=OrderSide.SELL,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        entry_time=datetime.fromisoformat(entry_date),
        exit_time=datetime.fromisoformat(exit_date),
        pnl=(exit_price - entry_price) * quantity - commission,
        pnl_pct=((exit_price - entry_price) / entry_price) * 100,
        commission=commission,
        net_pnl=(exit_price - entry_price) * quantity - commission,
        holding_period_days=(datetime.fromisoformat(exit_date) - datetime.fromisoformat(entry_date)).days,
    )


class TestTaxCalculator:
    """Test suite for tax calculations."""

    def test_short_term_gain(self):
        """Test short-term capital gains tax."""
        config = TaxConfig(short_term_rate=0.37, long_term_rate=0.20)
        calc = TaxCalculator(config)

        trade = _make_trade(
            entry_price=100, exit_price=150, quantity=100,
            entry_date="2024-01-01", exit_date="2024-06-01",
        )

        result = calc.calculate_trade_tax(trade)
        assert result["is_long_term"] is False
        assert result["federal_tax_rate"] == 0.37
        assert result["gain_loss"] == 5000  # (150-100) * 100
        assert result["federal_tax"] == pytest.approx(1850)  # 5000 * 0.37

    def test_long_term_gain(self):
        """Test long-term capital gains tax."""
        config = TaxConfig(short_term_rate=0.37, long_term_rate=0.20)
        calc = TaxCalculator(config)

        trade = _make_trade(
            entry_price=100, exit_price=200, quantity=50,
            entry_date="2023-01-01", exit_date="2024-06-01",
        )

        result = calc.calculate_trade_tax(trade)
        assert result["is_long_term"] is True
        assert result["federal_tax_rate"] == 0.20
        assert result["gain_loss"] == 5000
        assert result["federal_tax"] == pytest.approx(1000)

    def test_loss_trade(self):
        """Test losing trade (no tax on losses)."""
        config = TaxConfig(short_term_rate=0.37, long_term_rate=0.20)
        calc = TaxCalculator(config)

        trade = _make_trade(
            entry_price=100, exit_price=80, quantity=100,
            entry_date="2024-01-01", exit_date="2024-06-01",
        )

        result = calc.calculate_trade_tax(trade)
        assert result["gain_loss"] == -2000
        assert result["federal_tax"] == 0

    def test_period_summary(self):
        """Test quarterly/annual summary calculation."""
        config = TaxConfig(short_term_rate=0.37, long_term_rate=0.20)
        calc = TaxCalculator(config)

        trades = [
            _make_trade(entry_price=100, exit_price=120, quantity=100,
                       entry_date="2024-02-01", exit_date="2024-03-01"),
            _make_trade(entry_price=100, exit_price=90, quantity=50,
                       entry_date="2024-01-15", exit_date="2024-02-15"),
        ]

        summary = calc.calculate_period_summary(
            trades, date(2024, 1, 1), date(2024, 3, 31)
        )

        assert summary.short_term_gains == 2000  # 20 * 100
        assert summary.short_term_losses == 500   # 10 * 50
        assert summary.net_short_term == 1500
        assert summary.estimated_federal_tax > 0

    def test_state_tax(self):
        """Test state tax inclusion."""
        config = TaxConfig(short_term_rate=0.37, state_tax_rate=0.05)
        calc = TaxCalculator(config)

        trade = _make_trade(
            entry_price=100, exit_price=150, quantity=100,
        )

        result = calc.calculate_trade_tax(trade)
        assert result["state_tax"] == pytest.approx(250)  # 5000 * 0.05

    def test_tax_report_text(self):
        """Test human-readable tax report."""
        config = TaxConfig()
        calc = TaxCalculator(config)

        trades = [
            _make_trade(entry_price=100, exit_price=150, quantity=100),
        ]

        summary = calc.calculate_period_summary(
            trades, date(2024, 1, 1), date(2024, 12, 31)
        )

        text = calc.generate_tax_report_text(summary)
        assert "TAX SUMMARY" in text
        assert "Capital Gains" in text
