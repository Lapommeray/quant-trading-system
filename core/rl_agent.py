"""
Reinforcement Learning Agent - Tabular Q-Learning

Pure numpy. No torch. No external AI.

Learns which actions (BUY/SELL/HOLD) are profitable in which market states.
State is discretized from indicator features. Rewards come from actual P&L.

Self-improving: every trade outcome updates the Q-table, so the agent
gets better at recognizing profitable setups over time.
"""

import json
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("RLAgent")

ACTIONS = ["BUY", "SELL", "HOLD"]
ACTION_TO_IDX = {"BUY": 0, "SELL": 1, "HOLD": 2}

DEFAULT_Q_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "rl_qtable.json"
)

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999


class ReinforcementAgent:
    """
    Tabular Q-learning agent for trading decisions.

    State space: discretized market features (trend, volatility, momentum, volume)
    Action space: BUY, SELL, HOLD
    Reward: profit/loss from actual price movement after signal
    """

    def __init__(self, state_path: str = DEFAULT_Q_PATH):
        self._state_path = state_path
        self._q_table: Dict[str, list] = {}
        self._epsilon = EPSILON_START
        self._total_updates = 0
        self._last_state: Optional[str] = None
        self._last_action: Optional[int] = None
        self._cumulative_reward = 0.0
        self._load()

    def _load(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                self._q_table = data.get("q_table", {})
                self._epsilon = data.get("epsilon", EPSILON_START)
                self._total_updates = data.get("total_updates", 0)
                self._cumulative_reward = data.get("cumulative_reward", 0.0)
                logger.info(
                    f"RL agent loaded: {len(self._q_table)} states, "
                    f"{self._total_updates} updates, epsilon={self._epsilon:.4f}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load RL state: {e}")
        self._q_table = {}
        self._epsilon = EPSILON_START

    def _save(self):
        data = {
            "q_table": self._q_table,
            "epsilon": self._epsilon,
            "total_updates": self._total_updates,
            "cumulative_reward": self._cumulative_reward,
            "last_updated": time.time(),
        }
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning(f"Failed to save RL state: {e}")

    def discretize_state(self, features: Dict[str, float]) -> str:
        trend = self._bucket(features.get("trend_spread", 0), [-0.005, -0.001, 0.001, 0.005])
        rsi = self._bucket(features.get("rsi", 50), [30, 40, 60, 70])
        macd_hist = self._bucket(features.get("macd_histogram", 0), [-2, -0.5, 0.5, 2])
        adx = self._bucket(features.get("adx", 20), [15, 25, 40])
        vol_regime = self._bucket(features.get("volatility", 0.01), [0.005, 0.015, 0.03])

        return f"{trend}_{rsi}_{macd_hist}_{adx}_{vol_regime}"

    def _bucket(self, value: float, thresholds: list) -> int:
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)

    def _get_q(self, state: str) -> list:
        if state not in self._q_table:
            self._q_table[state] = [0.0, 0.0, 0.0]
        return self._q_table[state]

    def decide(self, features: Dict[str, float]) -> Tuple[str, float]:
        state = self.discretize_state(features)
        q_values = self._get_q(state)

        if self._total_updates < 50:
            action_idx = int(np.argmax(q_values))
        elif np.random.random() < self._epsilon:
            action_idx = int(np.random.randint(0, 3))
        else:
            action_idx = int(np.argmax(q_values))

        self._last_state = state
        self._last_action = action_idx

        action = ACTIONS[action_idx]
        q_val = q_values[action_idx]

        return action, q_val

    def reinforce(self, reward: float, new_features: Optional[Dict[str, float]] = None):
        if self._last_state is None or self._last_action is None:
            return

        q_values = self._get_q(self._last_state)

        if new_features is not None:
            new_state = self.discretize_state(new_features)
            new_q = self._get_q(new_state)
            max_future_q = max(new_q)
        else:
            max_future_q = 0.0

        old_q = q_values[self._last_action]
        new_q_val = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
        q_values[self._last_action] = new_q_val
        self._q_table[self._last_state] = q_values

        self._total_updates += 1
        self._cumulative_reward += reward
        self._epsilon = max(EPSILON_MIN, self._epsilon * EPSILON_DECAY)

        self._save()

        logger.debug(
            f"RL update: state={self._last_state}, action={ACTIONS[self._last_action]}, "
            f"reward={reward:.4f}, Q: {old_q:.4f}->{new_q_val:.4f}"
        )

    def get_action_confidence(self, features: Dict[str, float]) -> Dict[str, float]:
        state = self.discretize_state(features)
        q_values = self._get_q(state)

        q_sum = sum(abs(q) for q in q_values)
        if q_sum == 0:
            return {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34}

        softmax_vals = np.exp(np.array(q_values) - max(q_values))
        probs = softmax_vals / softmax_vals.sum()

        return {
            "BUY": float(probs[0]),
            "SELL": float(probs[1]),
            "HOLD": float(probs[2]),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "states_known": len(self._q_table),
            "total_updates": self._total_updates,
            "epsilon": round(self._epsilon, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
        }
