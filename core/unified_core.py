"""
Unified Intelligence Core v1.0

Fuses all components into a single self-evolving decision engine:
- 8 technical indicators (from qmp_engine_standalone)
- Reinforcement learning (Q-learning)
- Bayesian regime detection
- Genetic parameter evolution
- Microstructure analysis
- Self-learning feedback loop

No external AI. No LLM. No API calls. Pure self-contained intelligence.

The system improves itself by:
1. Tracking every signal outcome
2. Adjusting indicator weights based on accuracy
3. Evolving parameters via genetic algorithms
4. Learning profitable state-action pairs via RL
5. Detecting regime changes via Bayesian inference
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("UnifiedCore")

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CORE_DIR)


def _load_module(filename, classname):
    import importlib.util
    path = os.path.join(_CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


QMPStandaloneEngine = _load_module("qmp_engine_standalone.py", "QMPStandaloneEngine")
SelfLearningLoop = _load_module("self_learning.py", "SelfLearningLoop")
ReinforcementAgent = _load_module("rl_agent.py", "ReinforcementAgent")
BayesianState = _load_module("bayesian_state.py", "BayesianState")
GeneticEvolver = _load_module("genetic_evolver.py", "GeneticEvolver")
MicrostructureDetector = _load_module("microstructure_detector.py", "MicrostructureDetector")


class UnifiedIntelligenceCore:
    """
    Self-evolving trading intelligence.

    Combines indicator consensus with RL, Bayesian belief, genetic evolution,
    and microstructure detection. Learns from its own outcomes.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = _ROOT_DIR

        self._engine = QMPStandaloneEngine()
        self._learner = SelfLearningLoop(
            state_path=os.path.join(data_dir, "learning_state.json")
        )
        self._rl = ReinforcementAgent(
            state_path=os.path.join(data_dir, "rl_qtable.json")
        )
        self._bayes = BayesianState(
            state_path=os.path.join(data_dir, "bayesian_state.json")
        )
        self._genetic = GeneticEvolver(
            state_path=os.path.join(data_dir, "genetic_state.json")
        )
        self._micro = MicrostructureDetector()

        self._last_prices: Dict[str, float] = {}
        self._cycle_count = 0

        logger.info("UnifiedIntelligenceCore initialized — all components loaded")

    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full intelligence pipeline:
        1. Evaluate previous signal (self-learning)
        2. Extract microstructure features
        3. Update Bayesian beliefs
        4. Get indicator consensus (with learned weights)
        5. RL proposes action
        6. Fuse all signals into final decision
        7. Genetic evolution if needed
        8. Record signal for future evaluation
        """
        self._cycle_count += 1
        current_price = market_data.get("close", 0)

        # --- Step 1: Evaluate previous signal ---
        eval_result = None
        if symbol in self._last_prices and current_price > 0:
            eval_result = self._learner.evaluate_previous(symbol, current_price)
            if eval_result is not None:
                reward = self._compute_reward(eval_result)
                self._rl.reinforce(reward)
                fitness = 1.0 if eval_result["correct"] else -0.5
                self._genetic.report_fitness(fitness)

        # --- Step 2: Prepare dataframe for microstructure ---
        df = self._engine._prepare_dataframe(market_data.get("ohlcv", []))

        # --- Step 3: Microstructure features ---
        micro_features = {}
        micro_bias = None
        micro_conf = 0.0
        if df is not None and len(df) >= 10:
            micro_features = self._micro.extract_features(df)
            micro_bias, micro_conf = self._micro.get_signal_bias(micro_features)

        # --- Step 4: Indicator consensus with learned weights ---
        indicator_result = self._engine.generate_signal(symbol, market_data)
        indicator_signal = indicator_result.get("final_signal")
        indicator_confidence = indicator_result.get("confidence", 0)
        indicator_votes = indicator_result.get("votes", {})

        learned_weights = self._learner.get_adjusted_weights()
        if learned_weights:
            adjusted_confidence = self._reweight_confidence(
                indicator_votes, learned_weights, indicator_signal
            )
            indicator_confidence = max(indicator_confidence, adjusted_confidence)

        # --- Step 5: Bayesian regime update ---
        evidence = self._build_evidence(indicator_result, micro_features)
        beliefs, info_gain = self._bayes.update(evidence)
        regime = self._bayes.current_regime

        # --- Step 6: RL decision ---
        rl_features = self._build_rl_features(indicator_result, micro_features, regime)
        rl_action, rl_q = self._rl.decide(rl_features)
        rl_probs = self._rl.get_action_confidence(rl_features)

        # --- Step 7: Fuse all signals ---
        final_signal, final_confidence = self._fuse_signals(
            indicator_signal=indicator_signal,
            indicator_confidence=indicator_confidence,
            rl_action=rl_action,
            rl_q=rl_q,
            rl_probs=rl_probs,
            micro_bias=micro_bias,
            micro_conf=micro_conf,
            regime=regime,
            bayes_confidence=self._bayes.confidence,
            info_gain=info_gain,
        )

        # --- Step 8: Record for future evaluation ---
        if current_price > 0:
            self._last_prices[symbol] = current_price
            self._learner.record_signal(
                symbol=symbol,
                signal=final_signal,
                confidence=final_confidence,
                price_at_signal=current_price,
                indicator_votes=indicator_votes,
            )

        result = {
            "final_signal": final_signal,
            "confidence": round(final_confidence, 4),
            "regime": regime,
            "bayes_confidence": round(self._bayes.confidence, 4),
            "info_gain": round(info_gain, 4),
            "rl_action": rl_action,
            "rl_q": round(rl_q, 4),
            "micro_bias": micro_bias,
            "indicator_signal": indicator_signal,
            "votes": indicator_votes,
            "cycle": self._cycle_count,
        }

        if eval_result:
            result["prev_eval"] = eval_result

        logger.info(
            f"[Unified] {symbol} cycle={self._cycle_count}: "
            f"signal={final_signal}, conf={final_confidence:.3f}, "
            f"regime={regime}, rl={rl_action}, micro={micro_bias}"
        )

        return result

    def _fuse_signals(
        self,
        indicator_signal: Optional[str],
        indicator_confidence: float,
        rl_action: str,
        rl_q: float,
        rl_probs: Dict[str, float],
        micro_bias: Optional[str],
        micro_conf: float,
        regime: str,
        bayes_confidence: float,
        info_gain: float,
    ) -> Tuple[Optional[str], float]:
        """
        Weighted fusion of all signal sources.

        Weights:
        - Indicator consensus: 0.45 (backbone)
        - RL agent: 0.20 (learned patterns)
        - Microstructure: 0.20 (hidden features)
        - Regime alignment: 0.15 (Bayesian context)
        """
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0

        # Indicator consensus (weight: 0.45)
        if indicator_signal == "BUY":
            buy_score += 0.45 * indicator_confidence
        elif indicator_signal == "SELL":
            sell_score += 0.45 * indicator_confidence
        else:
            hold_score += 0.45 * 0.5

        # RL agent (weight: 0.20)
        buy_score += 0.20 * rl_probs.get("BUY", 0.33)
        sell_score += 0.20 * rl_probs.get("SELL", 0.33)
        hold_score += 0.20 * rl_probs.get("HOLD", 0.34)

        # Microstructure (weight: 0.20)
        if micro_bias == "BUY":
            buy_score += 0.20 * micro_conf
        elif micro_bias == "SELL":
            sell_score += 0.20 * micro_conf
        else:
            hold_score += 0.20 * 0.4

        # Regime alignment (weight: 0.15)
        if regime == "BULL":
            buy_score += 0.15 * bayes_confidence
        elif regime == "BEAR":
            sell_score += 0.15 * bayes_confidence
        else:
            hold_score += 0.15 * 0.5

        # High information gain = high uncertainty = prefer HOLD
        if info_gain > 0.5:
            hold_score += 0.1

        total = buy_score + sell_score + hold_score
        if total == 0:
            return "HOLD", 0.3

        # Require clear margin to emit directional signal
        if buy_score > sell_score and buy_score > hold_score:
            margin = buy_score - max(sell_score, hold_score)
            if margin > 0.05:
                confidence = min(0.98, buy_score / total + margin)
                return "BUY", confidence

        if sell_score > buy_score and sell_score > hold_score:
            margin = sell_score - max(buy_score, hold_score)
            if margin > 0.05:
                confidence = min(0.98, sell_score / total + margin)
                return "SELL", confidence

        return "HOLD", round(hold_score / total, 4)

    def _reweight_confidence(
        self, votes: Dict[str, Optional[str]], weights: Dict[str, float], signal: Optional[str]
    ) -> float:
        if signal is None or signal == "HOLD":
            return 0.0

        aligned_weight = 0.0
        for name, vote in votes.items():
            if vote == signal:
                aligned_weight += weights.get(name, 0.1)

        return min(0.95, 0.4 + aligned_weight)

    def _build_evidence(
        self, indicator_result: Dict[str, Any], micro_features: Dict[str, float]
    ) -> Dict[str, float]:
        details = indicator_result.get("details", {})

        trend_spread = 0.0
        if "trend_sma" in details:
            trend_spread = details["trend_sma"].get("spread", 0)

        rsi = 50.0
        if "rsi" in details:
            rsi = details["rsi"].get("rsi", 50)

        adx = 20.0
        if "adx" in details:
            adx = details["adx"].get("adx", 20)

        return {
            "trend_spread": trend_spread,
            "rsi": rsi,
            "adx": adx,
            "volatility": micro_features.get("volatility", 0.01),
        }

    def _build_rl_features(
        self, indicator_result: Dict[str, Any], micro_features: Dict[str, float], regime: str
    ) -> Dict[str, float]:
        details = indicator_result.get("details", {})
        return {
            "trend_spread": details.get("trend_sma", {}).get("spread", 0),
            "rsi": details.get("rsi", {}).get("rsi", 50),
            "macd_histogram": details.get("macd", {}).get("histogram", 0),
            "adx": details.get("adx", {}).get("adx", 20),
            "volatility": micro_features.get("volatility", 0.01),
            "depth_imbalance": micro_features.get("depth_imbalance", 0),
            "entropy_shift": micro_features.get("entropy_shift", 0),
        }

    def _compute_reward(self, eval_result: Dict[str, Any]) -> float:
        change_pct = eval_result.get("change_pct", 0)
        correct = eval_result.get("correct", False)

        if correct:
            return min(1.0, abs(change_pct) * 10)
        else:
            return max(-1.0, -abs(change_pct) * 10)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cycle_count": self._cycle_count,
            "learning": self._learner.get_stats(),
            "rl": self._rl.get_stats(),
            "bayesian": self._bayes.get_stats(),
            "genetic": self._genetic.get_stats(),
        }
