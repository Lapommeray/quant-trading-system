"""
Bayesian State Estimator

Maintains a probabilistic belief about market state (trending up, down, or ranging).
Updates beliefs using Bayes' theorem as new evidence arrives.
Calculates information gain to measure how much each new observation changes beliefs.

No external AI. Pure probability math.
"""

import json
import os
import time
import logging
import math
from typing import Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("BayesianState")

STATES = ["BULL", "BEAR", "RANGE"]
STATE_TO_IDX = {"BULL": 0, "BEAR": 1, "RANGE": 2}

DEFAULT_BAYES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bayesian_state.json"
)

TRANSITION_MATRIX = np.array([
    [0.80, 0.10, 0.10],
    [0.10, 0.80, 0.10],
    [0.15, 0.15, 0.70],
])


class BayesianState:
    """
    Bayesian regime estimator.

    Maintains P(state) for each market regime and updates with evidence.
    Provides:
    - Current belief distribution over regimes
    - Information gain (how surprising the latest observation was)
    - Confidence = max(beliefs) - measure of certainty
    """

    def __init__(self, state_path: str = DEFAULT_BAYES_PATH):
        self._state_path = state_path
        self._beliefs: np.ndarray = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self._history_length = 0
        self._confidence = 0.0
        self._information_gain = 0.0
        self._regime_durations: Dict[str, int] = {"BULL": 0, "BEAR": 0, "RANGE": 0}
        self._load()

    def _load(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                self._beliefs = np.array(data.get("beliefs", [1/3, 1/3, 1/3]))
                self._history_length = data.get("history_length", 0)
                self._regime_durations = data.get("regime_durations", {"BULL": 0, "BEAR": 0, "RANGE": 0})
                self._confidence = float(np.max(self._beliefs))
                logger.info(
                    f"Bayesian state loaded: beliefs={self._beliefs.round(3).tolist()}, "
                    f"history={self._history_length}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load Bayesian state: {e}")

    def _save(self):
        data = {
            "beliefs": self._beliefs.tolist(),
            "history_length": self._history_length,
            "regime_durations": self._regime_durations,
            "last_updated": time.time(),
        }
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning(f"Failed to save Bayesian state: {e}")

    def _likelihood(self, evidence: Dict[str, float], state_idx: int) -> float:
        trend = evidence.get("trend_spread", 0)
        rsi = evidence.get("rsi", 50)
        adx = evidence.get("adx", 20)
        volatility = evidence.get("volatility", 0.01)

        if state_idx == 0:  # BULL
            p_trend = self._gaussian(trend, 0.005, 0.008)
            p_rsi = self._gaussian(rsi, 60, 15)
            p_adx = self._gaussian(adx, 35, 15)
        elif state_idx == 1:  # BEAR
            p_trend = self._gaussian(trend, -0.005, 0.008)
            p_rsi = self._gaussian(rsi, 40, 15)
            p_adx = self._gaussian(adx, 35, 15)
        else:  # RANGE
            p_trend = self._gaussian(trend, 0.0, 0.003)
            p_rsi = self._gaussian(rsi, 50, 10)
            p_adx = self._gaussian(adx, 15, 10)

        return max(1e-10, p_trend * p_rsi * p_adx)

    def _gaussian(self, x: float, mu: float, sigma: float) -> float:
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    def update(self, evidence: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        prior = TRANSITION_MATRIX.T @ self._beliefs

        likelihoods = np.array([
            self._likelihood(evidence, i) for i in range(3)
        ])

        posterior_unnorm = prior * likelihoods
        total = posterior_unnorm.sum()
        if total == 0:
            posterior = np.array([1/3, 1/3, 1/3])
        else:
            posterior = posterior_unnorm / total

        kl_div = 0.0
        for i in range(3):
            if posterior[i] > 1e-10 and self._beliefs[i] > 1e-10:
                kl_div += posterior[i] * math.log(posterior[i] / self._beliefs[i])
        self._information_gain = max(0.0, kl_div)

        self._beliefs = posterior
        self._confidence = float(np.max(posterior))
        self._history_length += 1

        regime = STATES[int(np.argmax(posterior))]
        self._regime_durations[regime] = self._regime_durations.get(regime, 0) + 1

        self._save()

        belief_dict = {
            STATES[i]: round(float(posterior[i]), 4) for i in range(3)
        }

        logger.debug(
            f"Bayesian update: regime={regime}, confidence={self._confidence:.3f}, "
            f"IG={self._information_gain:.4f}"
        )

        return belief_dict, self._information_gain

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def information_gain(self) -> float:
        return self._information_gain

    @property
    def current_regime(self) -> str:
        return STATES[int(np.argmax(self._beliefs))]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "beliefs": {STATES[i]: round(float(self._beliefs[i]), 4) for i in range(3)},
            "regime": self.current_regime,
            "confidence": round(self._confidence, 4),
            "information_gain": round(self._information_gain, 4),
            "history_length": self._history_length,
            "regime_durations": dict(self._regime_durations),
        }
