"""
Custom exceptions for the YellowStars trading platform.
"""


class YellowStarsError(Exception):
    """Base exception for all YellowStars errors."""
    pass


# --- Data Errors ---

class DataError(YellowStarsError):
    """Base class for data-related errors."""
    pass


class DataProviderError(DataError):
    """Error communicating with a data provider API."""
    pass


class DataValidationError(DataError):
    """Data failed validation checks."""
    pass


class InsufficientDataError(DataError):
    """Not enough data to perform the requested operation."""
    pass


class SymbolNotFoundError(DataError):
    """Requested symbol not found in data provider."""
    pass


# --- Strategy Errors ---

class StrategyError(YellowStarsError):
    """Base class for strategy-related errors."""
    pass


class StrategyConfigError(StrategyError):
    """Invalid strategy configuration."""
    pass


class StrategyExecutionError(StrategyError):
    """Error during strategy signal generation."""
    pass


# --- Backtest Errors ---

class BacktestError(YellowStarsError):
    """Base class for backtesting errors."""
    pass


class BacktestConfigError(BacktestError):
    """Invalid backtest configuration."""
    pass


# --- Execution Errors ---

class ExecutionError(YellowStarsError):
    """Base class for execution/broker errors."""
    pass


class BrokerConnectionError(ExecutionError):
    """Cannot connect to broker."""
    pass


class OrderRejectedError(ExecutionError):
    """Broker rejected the order."""
    pass


class InsufficientFundsError(ExecutionError):
    """Not enough capital to execute the order."""
    pass


class RiskLimitExceededError(ExecutionError):
    """Order would exceed risk limits."""
    pass


class CircuitBreakerTriggered(ExecutionError):
    """Max drawdown or daily loss limit reached. Trading halted."""
    pass


# --- Orchestrator Errors ---

class OrchestratorError(YellowStarsError):
    """Error in the daily orchestration workflow."""
    pass


class MarketClosedError(OrchestratorError):
    """Attempted to trade when market is closed."""
    pass


# --- Configuration Errors ---

class ConfigError(YellowStarsError):
    """Invalid or missing configuration."""
    pass


class MissingAPIKeyError(ConfigError):
    """Required API key not provided."""
    pass
