"""
Example organism alpha module demonstrating new BaseTradingModule API,
auto-registration, event-bus communication, and self-improvement.

This module is auto-discovered by Organism and AutoModuleRegistry.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.base_module import BaseTradingModule, ModuleResult, register_module


@register_module
class ExampleOrganismAlpha(BaseTradingModule):
    module_name = "example_organism_alpha"
    category = "alpha"
    version = "1.0.0"
    dependencies = []

    def initialize(self) -> bool:
        self.lookback = self.config.get("lookback", 20)
        self.threshold = self.config.get("threshold", 0.02)
        return True

    def analyze(self, market_data: Dict[str, Any]) -> ModuleResult:
        # Simple momentum example
        symbol = market_data.get("symbol", market_data.get("history", {}).get("symbol", "BTC/USDT"))
        history = market_data.get("history", {})
        try:
            if isinstance(history, dict) and "1m" in history:
                df = history["1m"]
                if len(df) >= self.lookback:
                    closes = df["Close"].tail(self.lookback).values if "Close" in df.columns else df["close"].tail(self.lookback).values
                    returns = (closes[-1] - closes[0]) / closes[0]
                    if returns > self.threshold:
                        return ModuleResult(module_name=self.module_name, signal="BUY", confidence=min(abs(returns) * 5, 1.0), features={"return": returns})
                    elif returns < -self.threshold:
                        return ModuleResult(module_name=self.module_name, signal="SELL", confidence=min(abs(returns) * 5, 1.0), features={"return": returns})
        except Exception:
            pass
        return ModuleResult(module_name=self.module_name, signal="NEUTRAL", confidence=0.1)

    def generate_signal(self, symbol: str, history_data: Dict[str, Any]) -> ModuleResult:
        data = {"symbol": symbol, "history": history_data}
        return self.analyze(data)

    def on_event(self, event_type: str, payload: Dict[str, Any]):
        # Example: pause on risk alert
        if event_type == "RISK_ALERT":
            # Could reduce confidence or disable temporarily
            pass
        if event_type == "ORDER_FILLED":
            # Feedback learning could be triggered
            pass

    def self_improve(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple self-improvement: adjust threshold based on recent performance
        if not performance_history:
            return {"module": self.module_name, "improved": False}

        # Look at last improvement result if available
        last = performance_history[-1] if performance_history else {}
        weights = last.get("weights", {})
        my_weight = weights.get(self.module_name, 0)

        improved = False
        if my_weight and my_weight < 0.05:
            # Weight low -> tighten threshold to be more selective
            self.threshold *= 1.05
            improved = True
        elif my_weight and my_weight > 0.2:
            # Weight high -> relax threshold slightly to trade more
            self.threshold *= 0.98
            improved = True

        return {"module": self.module_name, "improved": improved, "new_threshold": self.threshold, "weight": my_weight}
