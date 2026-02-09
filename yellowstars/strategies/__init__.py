"""Strategy module - All trading strategies and signal generators."""
from yellowstars.strategies.base import BaseStrategy
from yellowstars.strategies.malik_white_light import MalikWhiteLightStrategy
from yellowstars.strategies.moving_average import MovingAverageCrossoverStrategy
from yellowstars.strategies.composite import CompositeStrategy
