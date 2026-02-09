"""
Multi-channel notification system.

Supports:
- Telegram
- Discord (webhook)
- SMS (Twilio)
- Email
- Console (always on)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import requests
from loguru import logger

from yellowstars.core.models import AlertConfig


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Notifier:
    """Multi-channel alert and notification system."""

    # Emoji mappings for different channels
    LEVEL_EMOJI = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.ERROR: "🔴",
        AlertLevel.CRITICAL: "🚨",
    }

    def __init__(self, config: AlertConfig):
        self.config = config
        self._channels: list[str] = []
        self._setup_channels()

    def _setup_channels(self) -> None:
        """Detect which notification channels are configured."""
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            self._channels.append("telegram")
        if self.config.discord_webhook_url:
            self._channels.append("discord")
        if self.config.twilio_account_sid:
            self._channels.append("sms")
        if self.config.email_smtp_server:
            self._channels.append("email")

        # Console is always available
        self._channels.append("console")

        logger.info(f"Notifier channels: {', '.join(self._channels)}")

    def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        title: str = "YellowStars",
    ) -> bool:
        """Send notification to all configured channels.

        Args:
            message: The notification message.
            level: Severity level.
            title: Notification title/subject.

        Returns:
            True if at least one channel succeeded.
        """
        if not self.config.enabled:
            return False

        emoji = self.LEVEL_EMOJI.get(level, "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{emoji} [{title}] {message}\n⏰ {timestamp}"

        success = False

        for channel in self._channels:
            try:
                if channel == "telegram":
                    success |= self._send_telegram(formatted)
                elif channel == "discord":
                    success |= self._send_discord(formatted, level)
                elif channel == "sms" and level in (AlertLevel.ERROR, AlertLevel.CRITICAL):
                    # Only send SMS for high-priority alerts
                    success |= self._send_sms(f"{title}: {message}")
                elif channel == "console":
                    self._log_console(message, level)
                    success = True
            except Exception as e:
                logger.error(f"Notification channel {channel} failed: {e}")

        return success

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> None:
        """Send a trade execution alert."""
        msg = f"Trade: {action.upper()} {quantity:.2f} {symbol} @ ${price:.2f}"
        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            msg += f"\n{pnl_emoji} P&L: ${pnl:+,.2f}"

        self.send(msg, AlertLevel.INFO, "Trade Executed")

    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_return: float,
        positions: list[dict],
    ) -> None:
        """Send end-of-day summary."""
        pos_text = ""
        for p in positions[:5]:  # Max 5 positions
            pos_text += f"\n  • {p.get('symbol', '?')}: {p.get('pnl_pct', 0):.1f}%"

        msg = (
            f"Daily Summary\n"
            f"💰 Equity: ${equity:,.2f}\n"
            f"📊 Daily P&L: ${daily_pnl:+,.2f} ({daily_return:+.2f}%)\n"
            f"📋 Positions:{pos_text or ' None'}"
        )

        level = AlertLevel.INFO if daily_pnl >= 0 else AlertLevel.WARNING
        self.send(msg, level, "Daily Report")

    def send_error_alert(self, error_msg: str, component: str = "") -> None:
        """Send an error alert."""
        msg = f"Error in {component}: {error_msg}" if component else error_msg
        self.send(msg, AlertLevel.ERROR, "System Error")

    def send_circuit_breaker_alert(self, reason: str) -> None:
        """Send a critical circuit breaker alert."""
        msg = f"CIRCUIT BREAKER TRIGGERED\n{reason}\nTrading has been halted."
        self.send(msg, AlertLevel.CRITICAL, "CIRCUIT BREAKER")

    # --- Channel Implementations ---

    def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram bot."""
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _send_discord(self, message: str, level: AlertLevel) -> bool:
        """Send message via Discord webhook."""
        color_map = {
            AlertLevel.INFO: 3447003,       # Blue
            AlertLevel.WARNING: 16776960,   # Yellow
            AlertLevel.ERROR: 16711680,     # Red
            AlertLevel.CRITICAL: 16711680,  # Red
        }

        payload = {
            "embeds": [{
                "description": message,
                "color": color_map.get(level, 3447003),
            }]
        }
        try:
            resp = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    def _send_sms(self, message: str) -> bool:
        """Send SMS via Twilio."""
        try:
            from twilio.rest import Client
            client = Client(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
            )
            client.messages.create(
                body=message[:1600],  # SMS character limit
                from_=self.config.twilio_from_number,
                to=self.config.twilio_to_number,
            )
            return True
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return False

    def _log_console(self, message: str, level: AlertLevel) -> None:
        """Log to console with appropriate level."""
        if level == AlertLevel.CRITICAL:
            logger.critical(message)
        elif level == AlertLevel.ERROR:
            logger.error(message)
        elif level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)
