"""Self-coding strategy generator using local deterministic synthesis.

External OpenAI/Grok API dependencies are intentionally removed.
"""

import datetime
import json
import logging
import os
from typing import Dict, Any


class StrategyGenerator:
    """Generate executable strategy code from current market state."""

    def __init__(self, algorithm, api_key=None):
        self.algorithm = algorithm
        self.logger = logging.getLogger("StrategyGenerator")
        self.logger.setLevel(logging.INFO)

        self.engine = "local-institutional-synth-v1"
        self.strategies_dir = "/strategies/generated"
        os.makedirs(self.strategies_dir, exist_ok=True)
        self.generated_strategies = []

    def generate_new_logic(self, market_state: Dict[str, Any]):
        self.logger.info("Generating new strategy for market state: %s", market_state)
        strategy_code = self._synthesize_strategy_code(market_state)
        strategy_path = self._save_strategy(strategy_code)
        self.generated_strategies.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "market_state": market_state,
                "path": strategy_path,
            }
        )
        return strategy_path

    def _synthesize_strategy_code(self, market_state: Dict[str, Any]) -> str:
        volatility = str(market_state.get("volatility", "medium")).lower()
        trend = str(market_state.get("trend", "neutral")).lower()

        lookback = 20 if volatility in {"medium", "normal"} else 35
        if volatility in {"high", "extreme"}:
            lookback = 50

        confidence_threshold = 0.58
        if trend in {"strong_bull", "strong_bear"}:
            confidence_threshold = 0.52

        return f'''from AlgorithmImports import *


class GeneratedInstitutionalStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol

        self.fast = self.SMA(self.symbol, {max(5, lookback//2)}, Resolution.Minute)
        self.slow = self.SMA(self.symbol, {lookback}, Resolution.Minute)
        self.atr = self.ATR(self.symbol, 14, MovingAverageType.Wilders, Resolution.Minute)

        self.conf_threshold = {confidence_threshold}
        self.risk_per_trade = 0.01

    def OnData(self, data):
        if not (self.fast.IsReady and self.slow.IsReady and self.atr.IsReady):
            return

        price = self.Securities[self.symbol].Price
        signal = self.fast.Current.Value - self.slow.Current.Value
        vol = self.atr.Current.Value / price if price > 0 else 0
        confidence = max(0.0, min(1.0, 0.55 + (0.25 if signal > 0 else -0.25) - 0.4 * vol))

        invested = self.Portfolio[self.symbol].Invested
        qty = self._position_size(price, vol)

        if confidence >= self.conf_threshold and signal > 0 and not invested:
            self.MarketOrder(self.symbol, qty)

        if invested:
            avg = self.Portfolio[self.symbol].AveragePrice
            stop = avg * (1 - 0.015)
            take = avg * (1 + 0.03)
            if price <= stop or price >= take or signal < 0:
                self.Liquidate(self.symbol)

    def _position_size(self, price, vol):
        risk_budget = self.Portfolio.TotalPortfolioValue * self.risk_per_trade
        stop_distance = max(price * 0.015, price * vol)
        units = int(max(1, risk_budget / max(stop_distance, 1e-6)))
        return units
'''

    def _save_strategy(self, strategy_code: str) -> str:
        filename = f"generated_strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        path = os.path.join(self.strategies_dir, filename)
        with open(path, "w") as f:
            f.write(strategy_code)
        return path

    # Backward-compatible method name used by older callers.
    def _call_openai_api(self, prompt):
        return self._synthesize_strategy_code({"prompt": prompt})
