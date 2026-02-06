"""
Genetic Parameter Evolver

Evolves trading parameters using genetic algorithms.
When performance drops, mutates parameters to explore new configurations.
When performance is good, preserves current parameters.

No external AI. Pure evolutionary optimization.

Parameters evolved:
- RSI thresholds (overbought/oversold levels)
- MACD signal sensitivity
- Bollinger Band width multiplier
- ADX trending threshold
- Stochastic levels
- Consensus minimum
"""

import json
import os
import time
import logging
from typing import Dict, Any, List, Optional
from copy import deepcopy

import numpy as np

logger = logging.getLogger("GeneticEvolver")

DEFAULT_GENETIC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "genetic_state.json"
)

POPULATION_SIZE = 8
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.1
ELITE_COUNT = 2
MIN_EVALUATIONS_BEFORE_EVOLVE = 10


def default_genome() -> Dict[str, float]:
    return {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "bollinger_width": 2.0,
        "adx_trend_threshold": 25.0,
        "stoch_oversold": 20.0,
        "stoch_overbought": 80.0,
        "sma_spread_threshold": 0.001,
        "min_consensus": 4.0,
        "confidence_threshold": 0.7,
    }


GENE_BOUNDS = {
    "rsi_oversold": (15.0, 45.0),
    "rsi_overbought": (55.0, 85.0),
    "bollinger_width": (1.0, 3.5),
    "adx_trend_threshold": (15.0, 40.0),
    "stoch_oversold": (10.0, 35.0),
    "stoch_overbought": (65.0, 90.0),
    "sma_spread_threshold": (0.0002, 0.005),
    "min_consensus": (2.0, 6.0),
    "confidence_threshold": (0.5, 0.9),
}


class GeneticEvolver:
    """
    Genetic algorithm that evolves trading parameters.

    Lifecycle:
    1. get_active_genome() — returns current best parameters
    2. report_fitness(score) — report how well current genome performed
    3. evolve() — when enough data, breed next generation
    """

    def __init__(self, state_path: str = DEFAULT_GENETIC_PATH):
        self._state_path = state_path
        self._population: List[Dict[str, float]] = []
        self._fitness: List[float] = []
        self._generation = 0
        self._active_idx = 0
        self._evaluations_this_gen = 0
        self._best_fitness_ever = -float("inf")
        self._best_genome_ever: Dict[str, float] = default_genome()
        self._load()

    def _load(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                self._population = data.get("population", [])
                self._fitness = data.get("fitness", [])
                self._generation = data.get("generation", 0)
                self._active_idx = data.get("active_idx", 0)
                self._evaluations_this_gen = data.get("evaluations_this_gen", 0)
                self._best_fitness_ever = data.get("best_fitness_ever", -float("inf"))
                self._best_genome_ever = data.get("best_genome_ever", default_genome())
                if self._population:
                    logger.info(
                        f"Genetic evolver loaded: gen={self._generation}, "
                        f"pop={len(self._population)}, best_fitness={self._best_fitness_ever:.4f}"
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to load genetic state: {e}")

        self._initialize_population()

    def _initialize_population(self):
        self._population = [default_genome()]
        for _ in range(POPULATION_SIZE - 1):
            genome = default_genome()
            for gene, (lo, hi) in GENE_BOUNDS.items():
                genome[gene] = float(np.random.uniform(lo, hi))
            self._population.append(genome)
        self._fitness = [0.0] * POPULATION_SIZE
        self._generation = 0
        self._active_idx = 0
        self._save()

    def _save(self):
        data = {
            "population": self._population,
            "fitness": self._fitness,
            "generation": self._generation,
            "active_idx": self._active_idx,
            "evaluations_this_gen": self._evaluations_this_gen,
            "best_fitness_ever": self._best_fitness_ever,
            "best_genome_ever": self._best_genome_ever,
            "last_updated": time.time(),
        }
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning(f"Failed to save genetic state: {e}")

    def get_active_genome(self) -> Dict[str, float]:
        if not self._population:
            return default_genome()
        idx = min(self._active_idx, len(self._population) - 1)
        return deepcopy(self._population[idx])

    def report_fitness(self, score: float):
        if not self._population:
            return

        idx = min(self._active_idx, len(self._fitness) - 1)
        self._fitness[idx] = self._fitness[idx] * 0.9 + score * 0.1
        self._evaluations_this_gen += 1

        if self._fitness[idx] > self._best_fitness_ever:
            self._best_fitness_ever = self._fitness[idx]
            self._best_genome_ever = deepcopy(self._population[idx])

        if self._evaluations_this_gen >= MIN_EVALUATIONS_BEFORE_EVOLVE:
            self.evolve()
        else:
            self._save()

    def evolve(self):
        if len(self._population) < 2:
            return

        sorted_indices = sorted(range(len(self._fitness)), key=lambda i: self._fitness[i], reverse=True)

        new_population = []
        for i in range(min(ELITE_COUNT, len(sorted_indices))):
            new_population.append(deepcopy(self._population[sorted_indices[i]]))

        while len(new_population) < POPULATION_SIZE:
            parent_a_idx = self._tournament_select(sorted_indices)
            parent_b_idx = self._tournament_select(sorted_indices)
            child = self._crossover(
                self._population[parent_a_idx],
                self._population[parent_b_idx]
            )
            child = self._mutate(child)
            new_population.append(child)

        self._population = new_population
        self._fitness = [self._fitness[sorted_indices[0]] * 0.5 if i < ELITE_COUNT else 0.0
                         for i in range(POPULATION_SIZE)]
        self._generation += 1
        self._active_idx = 0
        self._evaluations_this_gen = 0

        logger.info(
            f"[Genetic] Evolved to generation {self._generation}, "
            f"best_fitness={self._best_fitness_ever:.4f}"
        )
        self._save()

    def _tournament_select(self, sorted_indices: list, k: int = 3) -> int:
        candidates = np.random.choice(sorted_indices[:max(4, len(sorted_indices)//2)], size=min(k, len(sorted_indices)), replace=False)
        best = min(candidates, key=lambda i: sorted_indices.index(i) if i in sorted_indices else 999)
        return int(best)

    def _crossover(self, parent_a: Dict[str, float], parent_b: Dict[str, float]) -> Dict[str, float]:
        child = {}
        for gene in parent_a:
            if np.random.random() < 0.5:
                child[gene] = parent_a[gene]
            else:
                child[gene] = parent_b[gene]
        return child

    def _mutate(self, genome: Dict[str, float]) -> Dict[str, float]:
        mutated = deepcopy(genome)
        for gene, (lo, hi) in GENE_BOUNDS.items():
            if gene in mutated and np.random.random() < MUTATION_RATE:
                spread = (hi - lo) * MUTATION_STRENGTH
                mutated[gene] += float(np.random.normal(0, spread))
                mutated[gene] = float(np.clip(mutated[gene], lo, hi))
        return mutated

    def get_stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "active_idx": self._active_idx,
            "evaluations_this_gen": self._evaluations_this_gen,
            "best_fitness_ever": round(self._best_fitness_ever, 4),
            "current_fitness": [round(f, 4) for f in self._fitness],
        }
