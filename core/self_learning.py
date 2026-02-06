"""
Self-Learning Feedback Loop

Tracks every signal emitted, compares against actual price movement,
and auto-adjusts indicator weights based on accuracy.

No external AI. No LLM. Pure math-based self-improvement.

How it works:
1. After emitting a signal, record: symbol, direction, price, indicator votes
2. On next cycle, evaluate: did price move in predicted direction?
3. Score each indicator: correct prediction = +1, wrong = -1
4. Exponential moving average of scores adjusts weights
5. Persist state to disk so learning survives restarts
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("SelfLearning")

DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "learning_state.json"
)

LOOKBACK_BARS = 3
LEARNING_RATE = 0.05
MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.40
DECAY_FACTOR = 0.995


class SelfLearningLoop:
    """
    Tracks signal outcomes and adjusts indicator weights automatically.

    Lifecycle per cycle:
        1. evaluate_previous() — check if last signal was correct
        2. get_adjusted_weights() — return current learned weights
        3. record_signal() — store this cycle's signal for future evaluation
    """

    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self._state_path = state_path
        self._indicator_names = [
            "trend_sma", "rsi", "macd", "bollinger",
            "adx", "stochastic", "obv", "ichimoku"
        ]

        self._weights: Dict[str, float] = {}
        self._scores: Dict[str, float] = {}
        self._hit_counts: Dict[str, int] = {}
        self._miss_counts: Dict[str, int] = {}
        self._pending_signals: Dict[str, Dict[str, Any]] = {}
        self._total_evaluated: int = 0
        self._total_correct: int = 0

        self._load_state()

    def _load_state(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                self._weights = state.get("weights", {})
                self._scores = state.get("scores", {})
                self._hit_counts = state.get("hit_counts", {})
                self._miss_counts = state.get("miss_counts", {})
                self._total_evaluated = state.get("total_evaluated", 0)
                self._total_correct = state.get("total_correct", 0)
                logger.info(
                    f"Loaded learning state: {self._total_evaluated} evaluations, "
                    f"{self._total_correct} correct"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")

        for name in self._indicator_names:
            self._weights[name] = 1.0 / len(self._indicator_names)
            self._scores[name] = 0.0
            self._hit_counts[name] = 0
            self._miss_counts[name] = 0

    def _save_state(self):
        state = {
            "weights": self._weights,
            "scores": self._scores,
            "hit_counts": self._hit_counts,
            "miss_counts": self._miss_counts,
            "total_evaluated": self._total_evaluated,
            "total_correct": self._total_correct,
            "last_updated": time.time(),
        }
        try:
            tmp_path = self._state_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self._state_path)
        except Exception as e:
            logger.warning(f"Failed to save learning state: {e}")

    def record_signal(
        self,
        symbol: str,
        signal: Optional[str],
        confidence: float,
        price_at_signal: float,
        indicator_votes: Dict[str, Optional[str]],
    ):
        if signal is None or signal == "HOLD":
            return

        self._pending_signals[symbol] = {
            "signal": signal,
            "confidence": confidence,
            "price": price_at_signal,
            "votes": dict(indicator_votes),
            "timestamp": time.time(),
        }

    def evaluate_previous(
        self, symbol: str, current_price: float
    ) -> Optional[Dict[str, Any]]:
        pending = self._pending_signals.pop(symbol, None)
        if pending is None:
            return None

        prev_price = pending["price"]
        prev_signal = pending["signal"]
        prev_votes = pending["votes"]

        if prev_price == 0:
            return None

        price_change_pct = (current_price - prev_price) / prev_price

        if prev_signal == "BUY":
            correct = price_change_pct > 0.0001
        elif prev_signal == "SELL":
            correct = price_change_pct < -0.0001
        else:
            return None

        self._total_evaluated += 1
        if correct:
            self._total_correct += 1

        for name, vote in prev_votes.items():
            if name not in self._scores:
                continue

            if vote == prev_signal:
                if correct:
                    self._scores[name] += LEARNING_RATE
                    self._hit_counts[name] = self._hit_counts.get(name, 0) + 1
                else:
                    self._scores[name] -= LEARNING_RATE * 1.5
                    self._miss_counts[name] = self._miss_counts.get(name, 0) + 1
            elif vote is not None and vote != "HOLD":
                if not correct:
                    self._scores[name] += LEARNING_RATE * 0.5
                    self._hit_counts[name] = self._hit_counts.get(name, 0) + 1
                else:
                    self._scores[name] -= LEARNING_RATE * 0.5
                    self._miss_counts[name] = self._miss_counts.get(name, 0) + 1

        for name in self._scores:
            self._scores[name] *= DECAY_FACTOR

        self._recompute_weights()
        self._save_state()

        accuracy = (self._total_correct / self._total_evaluated * 100
                     if self._total_evaluated > 0 else 0)

        result = {
            "correct": correct,
            "signal": prev_signal,
            "price_at_signal": prev_price,
            "price_now": current_price,
            "change_pct": round(price_change_pct * 100, 4),
            "total_evaluated": self._total_evaluated,
            "total_correct": self._total_correct,
            "accuracy_pct": round(accuracy, 2),
        }

        logger.info(
            f"[SelfLearn] {symbol} {prev_signal} was {'CORRECT' if correct else 'WRONG'} "
            f"({price_change_pct*100:+.3f}%) | "
            f"Running accuracy: {accuracy:.1f}% ({self._total_correct}/{self._total_evaluated})"
        )

        return result

    def _recompute_weights(self):
        raw = {}
        for name in self._indicator_names:
            score = self._scores.get(name, 0.0)
            raw[name] = np.exp(score)

        total = sum(raw.values())
        if total == 0:
            total = 1.0

        for name in self._indicator_names:
            w = raw[name] / total
            self._weights[name] = float(np.clip(w, MIN_WEIGHT, MAX_WEIGHT))

        w_sum = sum(self._weights.values())
        if w_sum > 0:
            for name in self._weights:
                self._weights[name] /= w_sum

    def get_adjusted_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def get_stats(self) -> Dict[str, Any]:
        accuracy = (self._total_correct / self._total_evaluated * 100
                     if self._total_evaluated > 0 else 0)
        return {
            "total_evaluated": self._total_evaluated,
            "total_correct": self._total_correct,
            "accuracy_pct": round(accuracy, 2),
            "weights": dict(self._weights),
            "scores": dict(self._scores),
            "hit_counts": dict(self._hit_counts),
            "miss_counts": dict(self._miss_counts),
        }
