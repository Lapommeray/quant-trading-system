"""
Unified AI Trading Indicator v3.0
"""

import os
import logging
import time
import importlib.util
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger("UnifiedAIIndicator")

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CORE_DIR)
_TRAINER_DIR = os.path.join(_ROOT_DIR, "trainer")


def _load_core(filename, classname):
    path = os.path.join(_CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


def _load_trainer(filename, classname):
    path = os.path.join(_TRAINER_DIR, filename)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


UnifiedIntelligenceCore = _load_core("unified_core.py", "UnifiedIntelligenceCore")
FeatureEngineer = _load_core("feature_engineering.py", "FeatureEngineer")
DataPipeline = _load_core("data_pipeline.py", "DataPipeline")
RiskGuard = _load_core("risk_guard.py", "RiskGuard")
StateBuffer = _load_core("state_buffer.py", "StateBuffer")
PerformanceMetrics = _load_core("metrics.py", "PerformanceMetrics")

SelfEvolutionTrainer = _load_trainer("self_evolution.py", "SelfEvolutionTrainer")


class UnifiedAIIndicator:

    def __init__(self, symbol: str = "XAUUSD", data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = _ROOT_DIR
        self.symbol = symbol
        self._data_dir = data_dir
        self._core = UnifiedIntelligenceCore(data_dir=data_dir)
        self._features = FeatureEngineer()
        self._pipeline = DataPipeline()
        self._risk = RiskGuard()
        self._buffer = StateBuffer()
        self._metrics = PerformanceMetrics()
        self._evolver = None
        if SelfEvolutionTrainer is not None:
            try:
                self._evolver = SelfEvolutionTrainer(data_dir=data_dir)
                logger.info("SelfEvolutionTrainer loaded")
            except Exception as e:
                logger.warning(f"SelfEvolutionTrainer unavailable: {e}")
        self._confidence = 0.0
        self._last_signal = "HOLD"
        self._last_reason = "boot"
        self._cycle_count = 0
        self._signal_history = []
        self._last_price: Dict[str, float] = {}
        self._evolve_interval = 50
        logger.info(f"UnifiedAIIndicator v3 initialized for {symbol}")

    def process_tick(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._cycle_count += 1
        if market_data is None:
            market_data = self._pipeline.fetch(self.symbol)
            if not market_data.get("ohlcv"):
                return self._null_signal("no_data_available")
        current_price = market_data.get("close", 0)
        result = self._core.generate_signal(self.symbol, market_data)
        signal = result.get("final_signal", "HOLD")
        confidence = result.get("confidence", 0.0)
        regime = result.get("regime", "UNKNOWN")
        self._confidence = confidence
        features = {}
        ohlcv = market_data.get("ohlcv", [])
        if ohlcv and len(ohlcv) >= 20:
            try:
                import pandas as pd
                df = pd.DataFrame(ohlcv)
                for col in ["Close", "High", "Low", "Open", "Volume"]:
                    if col not in df.columns:
                        cl = col.lower()
                        if cl in df.columns:
                            df[col] = df[cl]
                features = self._features.extract(df)
            except Exception:
                pass
        self._buffer.record(price=current_price, features=features, signal=signal, confidence=confidence, regime=regime)
        memory_features = self._buffer.get_memory_features()
        if signal in ("BUY", "SELL") and confidence >= 0.7:
            risk_ok, risk_reason = self._risk.check_signal(signal_direction=signal, confidence=confidence, current_volatility=features.get("volatility", 0.01))
            if not risk_ok:
                logger.info(f"Risk guard blocked {signal}: {risk_reason}")
                signal = "HOLD"
                confidence = min(confidence, 0.4)
                result["risk_blocked"] = risk_reason
        if signal in ("BUY", "SELL") and confidence >= 0.7:
            self._last_signal = signal
        else:
            signal = self._last_signal if confidence > 0.4 else "HOLD"
        if self.symbol in self._last_price and current_price > 0:
            prev = self._last_price[self.symbol]
            if prev > 0:
                ret = (current_price - prev) / prev
                self._metrics.record_return(ret)
                self._metrics.record_equity(current_price)
        if current_price > 0:
            self._last_price[self.symbol] = current_price
        prev_eval = result.get("prev_eval")
        if prev_eval is not None:
            pnl = prev_eval.get("change_pct", 0.0)
            self._metrics.record_trade({"pnl": pnl, "signal": signal})
            self._risk.record_trade(pnl, self.symbol)
            true_dir = "BUY" if pnl > 0 else ("SELL" if pnl < 0 else "HOLD")
            self._metrics.record_prediction(signal, true_dir)
        if self._evolver and self._cycle_count % self._evolve_interval == 0:
            try:
                evo_result = self._evolver.evolve()
                if evo_result:
                    logger.info(f"Self-evolution at cycle {self._cycle_count}: {evo_result}")
            except Exception as e:
                logger.warning(f"Self-evolution failed: {e}")
        reason = self._explain(result, market_data, memory_features)
        self._last_reason = reason
        output = {
            "symbol": self.symbol, "signal": signal,
            "confidence": round(confidence, 6), "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle": self._cycle_count, "regime": regime,
            "indicator_signal": result.get("indicator_signal"),
            "rl_action": result.get("rl_action"),
            "micro_bias": result.get("micro_bias"),
            "votes": result.get("votes", {}),
        }
        if result.get("risk_blocked"):
            output["risk_blocked"] = result["risk_blocked"]
        self._signal_history.append({"cycle": self._cycle_count, "signal": signal, "confidence": confidence, "regime": regime})
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]
        return output

    def _explain(self, result: Dict[str, Any], market_data: Dict[str, Any], memory_features: Optional[Dict[str, float]] = None) -> str:
        parts = []
        regime = result.get("regime", "?")
        parts.append(f"regime={regime}")
        votes = result.get("votes", {})
        buy_count = sum(1 for v in votes.values() if v == "BUY")
        sell_count = sum(1 for v in votes.values() if v == "SELL")
        parts.append(f"indicators={buy_count}B/{sell_count}S")
        indicator_sig = result.get("indicator_signal")
        if indicator_sig:
            parts.append(f"consensus={indicator_sig}")
        rl_action = result.get("rl_action")
        if rl_action:
            parts.append(f"rl={rl_action}")
        micro = result.get("micro_bias")
        if micro:
            parts.append(f"micro={micro}")
        htf = result.get("htf_trend") if "htf_trend" in result else None
        if htf:
            parts.append(f"daily={htf}")
        conf = result.get("confidence", 0)
        parts.append(f"conf={conf:.3f}")
        if memory_features:
            drift = memory_features.get("mem_momentum_drift", 0)
            if abs(drift) > 0.001:
                parts.append(f"drift={'up' if drift > 0 else 'down'}")
            consistency = self._buffer.get_signal_consistency()
            if consistency > 0:
                parts.append(f"stability={consistency:.2f}")
        risk_status = self._risk.get_status()
        dd = risk_status.get("current_dd_pct", 0)
        if dd > 1.0:
            parts.append(f"dd={dd:.1f}%")
        if result.get("risk_blocked"):
            rb = result["risk_blocked"]
            parts.append(f"BLOCKED:{rb}")
        prev_eval = result.get("prev_eval")
        if prev_eval:
            correct = "WIN" if prev_eval.get("correct") else "LOSS"
            change = prev_eval.get("change_pct", 0)
            acc = prev_eval.get("accuracy_pct", 0)
            parts.append(f"prev={correct}({change:+.2f}%)")
            parts.append(f"accuracy={acc:.1f}%")
        return ", ".join(parts)

    def _null_signal(self, reason: str) -> Dict[str, Any]:
        return {"symbol": self.symbol, "signal": "HOLD", "confidence": 0.0, "reason": reason, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cycle": self._cycle_count, "regime": "UNKNOWN"}

    def reset(self):
        self._confidence = 0.0
        self._last_signal = "HOLD"
        self._last_reason = "reset"
        self._signal_history = []
        self._risk.reset_session()
        self._buffer = StateBuffer()
        self._metrics.reset()

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def last_signal(self) -> str:
        return self._last_signal

    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        self.symbol = symbol
        tick_result = self.process_tick(market_data)
        return {"final_signal": tick_result.get("signal"), "confidence": tick_result.get("confidence", 0.0), "regime": tick_result.get("regime"), "reason": tick_result.get("reason", ""), "indicator_signal": tick_result.get("indicator_signal"), "rl_action": tick_result.get("rl_action"), "micro_bias": tick_result.get("micro_bias"), "votes": tick_result.get("votes", {})}

    def get_stats(self) -> Dict[str, Any]:
        core_stats = self._core.get_stats()
        metrics_summary = self._metrics.get_summary()
        risk_status = self._risk.get_status()
        buffer_stats = self._buffer.get_stats()
        return {"symbol": self.symbol, "cycle_count": self._cycle_count, "last_signal": self._last_signal, "confidence": self._confidence, "last_reason": self._last_reason, "history_length": len(self._signal_history), "metrics": metrics_summary, "risk": risk_status, "buffer": buffer_stats, **core_stats}
