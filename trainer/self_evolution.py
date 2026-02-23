"""
Self-Evolution Trainer - Continual Adaptation Engine

Periodically re-fits RL weights, Bayesian priors, and genetic parameters
based on accumulated trading data. Runs after each session or on-demand.

Pipeline:
1. Load recent trade history and signal logs
2. Evaluate each component's contribution to accuracy
3. Re-weight fusion coefficients (indicator/RL/micro/regime)
4. Trigger genetic evolution if performance dropped
5. Reset RL exploration if regime changed
6. Save updated state to disk

Pure Python/numpy. No external AI. Runs offline.
"""

import os
import json
import logging
import time
import importlib.util
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("SelfEvolution")

_CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core")


def _load_class(filename, classname):
    path = os.path.join(_CORE_DIR, filename)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname, None)


class SelfEvolutionTrainer:
    """
    End-of-session re-training loop.

    After each trading day (or on-demand), this trainer:
    1. Reviews signal accuracy from the learning log
    2. Adjusts fusion weights based on component performance
    3. Triggers genetic evolution if accuracy dropped
    4. Resets RL exploration if regime changed significantly
    5. Persists all updates to disk
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.dirname(_CORE_DIR)
        self.data_dir = data_dir
        self.evolution_log_path = os.path.join(data_dir, "evolution_log.json")

        self._fusion_weights = {
            "indicators": 0.45,
            "rl": 0.20,
            "microstructure": 0.20,
            "regime": 0.15,
        }

        self._load_evolution_log()
        logger.info("SelfEvolutionTrainer initialized")

    def evolve(self) -> Dict[str, Any]:
        """
        Run one evolution cycle.

        Loads learning state, evaluates component accuracy,
        adjusts fusion weights, and triggers sub-component evolution.
        """
        results: Dict[str, Any] = {"timestamp": time.time(), "actions": []}

        learning_state = self._load_json("learning_state.json")
        rl_state = self._load_json("rl_qtable.json")
        bayesian_state = self._load_json("bayesian_state.json")
        genetic_state = self._load_json("genetic_state.json")

        if learning_state:
            accuracy = learning_state.get("accuracy_pct", 50.0)
            total_eval = learning_state.get("total_evaluated", 0)

            if total_eval >= 10:
                self._adjust_fusion_weights(learning_state, accuracy)
                results["actions"].append(f"fusion_weights_adjusted: {self._fusion_weights}")
                results["accuracy"] = accuracy

            if accuracy < 55.0 and total_eval >= 20:
                self._trigger_genetic_evolution(genetic_state)
                results["actions"].append("genetic_evolution_triggered")

            if accuracy > 70.0:
                self._reduce_rl_exploration(rl_state)
                results["actions"].append("rl_exploration_reduced")

        if bayesian_state:
            beliefs = bayesian_state.get("beliefs", [0.33, 0.33, 0.34])
            regime_certainty = max(beliefs) if beliefs else 0.33

            if regime_certainty > 0.8:
                results["actions"].append(f"regime_stable: certainty={regime_certainty:.3f}")
            elif regime_certainty < 0.5:
                self._reset_rl_exploration(rl_state)
                results["actions"].append("regime_uncertain: rl_exploration_reset")

        self._save_fusion_weights()
        self._save_evolution_log(results)

        logger.info(f"Evolution cycle complete: {len(results['actions'])} actions taken")
        return results

    def _adjust_fusion_weights(self, learning_state: Dict, accuracy: float):
        weights = learning_state.get("weights", {})
        if not weights:
            return

        indicator_scores = list(weights.values())
        avg_indicator_score = np.mean(indicator_scores) if indicator_scores else 0.5

        if accuracy > 65:
            self._fusion_weights["indicators"] = min(0.60, self._fusion_weights["indicators"] + 0.02)
            self._fusion_weights["rl"] = max(0.10, self._fusion_weights["rl"] - 0.01)
        elif accuracy < 50:
            self._fusion_weights["indicators"] = max(0.30, self._fusion_weights["indicators"] - 0.02)
            self._fusion_weights["rl"] = min(0.30, self._fusion_weights["rl"] + 0.01)

        total = sum(self._fusion_weights.values())
        if total > 0:
            for k in self._fusion_weights:
                self._fusion_weights[k] /= total

    def _trigger_genetic_evolution(self, genetic_state: Optional[Dict]):
        GeneticEvolver = _load_class("genetic_evolver.py", "GeneticEvolver")
        if GeneticEvolver is None:
            return
        try:
            evolver = GeneticEvolver(data_dir=self.data_dir)
            evolver.report_fitness(-1.0)
            evolver.evolve()
            logger.info("Genetic evolution triggered due to low accuracy")
        except Exception as e:
            logger.warning(f"Genetic evolution failed: {e}")

    def _reduce_rl_exploration(self, rl_state: Optional[Dict]):
        if rl_state is None:
            return
        current_eps = rl_state.get("epsilon", 0.3)
        new_eps = max(0.01, current_eps * 0.9)
        rl_state["epsilon"] = new_eps
        self._save_json("rl_qtable.json", rl_state)
        logger.info(f"RL exploration reduced: {current_eps:.3f} -> {new_eps:.3f}")

    def _reset_rl_exploration(self, rl_state: Optional[Dict]):
        if rl_state is None:
            return
        rl_state["epsilon"] = 0.3
        self._save_json("rl_qtable.json", rl_state)
        logger.info("RL exploration reset to 0.3 due to regime uncertainty")

    def get_fusion_weights(self) -> Dict[str, float]:
        return dict(self._fusion_weights)

    def _load_json(self, filename: str) -> Optional[Dict]:
        path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_json(self, filename: str, data: Dict):
        path = os.path.join(self.data_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save {filename}: {e}")

    def _save_fusion_weights(self):
        path = os.path.join(self.data_dir, "fusion_weights.json")
        try:
            with open(path, "w") as f:
                json.dump(self._fusion_weights, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save fusion weights: {e}")

    def _load_evolution_log(self):
        if os.path.isfile(self.evolution_log_path):
            try:
                with open(self.evolution_log_path, "r") as f:
                    log = json.load(f)
                    self._fusion_weights = log.get("fusion_weights", self._fusion_weights)
            except Exception:
                pass

    def _save_evolution_log(self, result: Dict):
        log = {
            "last_evolution": result,
            "fusion_weights": self._fusion_weights,
            "timestamp": time.time(),
        }
        try:
            with open(self.evolution_log_path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save evolution log: {e}")
