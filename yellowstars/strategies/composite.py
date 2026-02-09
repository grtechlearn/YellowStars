"""
Composite Strategy - Combines multiple sub-strategies.

This is the "meta-strategy" that:
1. Runs all enabled sub-strategies
2. Aggregates their signals (weighted voting)
3. Produces a final position target
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from yellowstars.config.settings import StrategySettings
from yellowstars.core.models import SignalAction
from yellowstars.strategies.base import BaseStrategy


class CompositeStrategy(BaseStrategy):
    """Combines multiple strategies via weighted signal aggregation.

    Usage:
        composite = CompositeStrategy(settings)
        composite.add_strategy(strategy1, weight=0.6)
        composite.add_strategy(strategy2, weight=0.4)
        signals = composite.generate_signals(data)
    """

    def __init__(self, settings: Optional[StrategySettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.sub_strategies: list[tuple[BaseStrategy, float]] = []  # (strategy, weight)

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """Add a sub-strategy with a weight."""
        self.sub_strategies.append((strategy, weight))
        logger.info(f"Added sub-strategy: {strategy.name} (weight={weight:.2f})")

    def remove_strategy(self, name: str) -> bool:
        """Remove a sub-strategy by name."""
        original_len = len(self.sub_strategies)
        self.sub_strategies = [
            (s, w) for s, w in self.sub_strategies if s.name != name
        ]
        return len(self.sub_strategies) < original_len

    @property
    def min_required_bars(self) -> int:
        if not self.sub_strategies:
            return 250
        return max(s.min_required_bars for s, _ in self.sub_strategies)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run all sub-strategies and aggregate signals.

        Aggregation method: Weighted average of position_pct values.
        """
        if not self.sub_strategies:
            logger.warning("CompositeStrategy has no sub-strategies")
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            return df

        # Normalize weights
        total_weight = sum(w for _, w in self.sub_strategies)
        if total_weight == 0:
            total_weight = 1.0

        # Run each sub-strategy
        all_signals = []
        all_instruments = []
        for strategy, weight in self.sub_strategies:
            try:
                result = strategy.generate_signals(data)
                all_signals.append((result["position_pct"], weight / total_weight))

                if "instrument" in result.columns:
                    all_instruments.append(result["instrument"])

                logger.debug(
                    f"Sub-strategy {strategy.name}: "
                    f"final_pos={result.iloc[-1]['position_pct']:.2%}"
                )
            except Exception as e:
                logger.error(f"Sub-strategy {strategy.name} failed: {e}")

        if not all_signals:
            df = data.copy()
            df["signal"] = SignalAction.HOLD.value
            df["position_pct"] = 0.0
            return df

        # Aggregate position sizes (weighted average)
        df = data.copy()
        weighted_position = pd.Series(0.0, index=df.index)
        for positions, norm_weight in all_signals:
            weighted_position += positions * norm_weight

        df["position_pct"] = weighted_position.clip(0.0, self.settings.max_position_pct)

        # Determine instrument (majority vote)
        if all_instruments:
            instrument_df = pd.concat(all_instruments, axis=1)
            df["instrument"] = instrument_df.mode(axis=1)[0]
        else:
            df["instrument"] = self.settings.long_asset

        # Generate action signals based on position changes
        df["signal"] = SignalAction.HOLD.value
        pos_diff = df["position_pct"].diff()
        df.loc[pos_diff > 0.05, "signal"] = SignalAction.BUY.value
        df.loc[pos_diff < -0.05, "signal"] = SignalAction.SELL.value
        df.loc[df["position_pct"] == 0, "signal"] = SignalAction.CLOSE.value

        # First valid position is a buy
        first_valid = df[df["position_pct"] > 0].index
        if len(first_valid) > 0:
            df.loc[first_valid[0], "signal"] = SignalAction.BUY.value

        logger.info(
            f"Composite strategy ({len(self.sub_strategies)} subs): "
            f"final position={df.iloc[-1]['position_pct']:.1%}"
        )

        return df

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "strategy_type": "composite",
            "num_sub_strategies": len(self.sub_strategies),
            "sub_strategies": [
                {"name": s.name, "weight": w}
                for s, w in self.sub_strategies
            ],
        }
