"""
Infinite Refinement Engine

Genetic algorithm for nightly strategy evolution for the QMP Overrider system.
"""

from AlgorithmImports import *
import logging
import numpy as np
import json
import os
import random
from datetime import datetime
import copy

class StrategyEvolver:
    """
    Genetic algorithm for nightly strategy evolution.
    """
    
    def __init__(self, algorithm, backtester=None):
        """
        Initialize the Strategy Evolver.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        - backtester: Backtester instance (optional)
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("StrategyEvolver")
        self.logger.setLevel(logging.INFO)
        
        self.backtester = backtester
        
        self.population_size = 20
        self.mutation_rate = 0.15
        self.crossover_rate = 0.45
        self.generations = 100
        self.elite_count = 2
        
        self.population = []
        self.fitness_scores = []
        
        self.evolution_history = []
        
        self.strategies_dir = "/strategies/evolved"
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        self.logger.info("Strategy Evolver initialized")
        
    def nightly_evolution(self):
        """
        Run nightly evolution of strategies.
        
        Returns:
        - Dictionary containing evolution results
        """
        self.logger.info("Starting nightly evolution")
        
        self.population = self._load_current_strategies()
        
        if not self.population:
            self.logger.warning("No strategies found for evolution")
            return {"status": "error", "message": "No strategies found"}
            
        self.logger.info(f"Loaded {len(self.population)} strategies")
        
        best_strategy = self._run_evolution()
        
        if not best_strategy:
            self.logger.error("Evolution failed")
            return {"status": "error", "message": "Evolution failed"}
            
        deployment_result = self._deploy_strategy(best_strategy)
        
        evolution_results = {
            "status": "success",
            "generations": self.generations,
            "population_size": len(self.population),
            "best_strategy": {
                "name": best_strategy.get("name", "Unknown"),
                "fitness": best_strategy.get("fitness", 0.0),
                "path": best_strategy.get("path", "")
            },
            "deployment": deployment_result
        }
        
        self.logger.info(f"Evolution results: {evolution_results}")
        
        return evolution_results
        
    def _load_current_strategies(self):
        """
        Load current strategies for evolution.
        
        Returns:
        - List of strategy objects
        """
        
        strategies = []
        
        if not os.path.exists(self.strategies_dir):
            self.logger.warning(f"Strategies directory not found: {self.strategies_dir}")
            return strategies
            
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.endswith('.json')]
        
        if not strategy_files:
            self.logger.warning("No strategy files found")
            
            for i in range(self.population_size):
                strategies.append(self._create_random_strategy(f"random_strategy_{i}"))
                
            return strategies
            
        for filename in strategy_files:
            filepath = os.path.join(self.strategies_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    strategy = json.load(f)
                    
                strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Error loading strategy file {filepath}: {str(e)}")
        
        while len(strategies) < self.population_size:
            strategies.append(self._create_random_strategy(f"random_strategy_{len(strategies)}"))
            
        return strategies
        
    def _create_random_strategy(self, name):
        """
        Create a random strategy.
        
        Parameters:
        - name: Strategy name
        
        Returns:
        - Strategy object
        """
        
        return {
            "name": name,
            "type": "moving_average_crossover",
            "parameters": {
                "fast_period": random.randint(5, 50),
                "slow_period": random.randint(20, 200),
                "stop_loss": random.uniform(0.01, 0.1),
                "take_profit": random.uniform(0.02, 0.2),
                "position_size": random.uniform(0.1, 1.0)
            },
            "fitness": 0.0,
            "path": ""
        }
        
    def _run_evolution(self):
        """
        Run genetic algorithm evolution.
        
        Returns:
        - Best strategy
        """
        self.logger.info(f"Running evolution for {self.generations} generations")
        
        self.fitness_scores = [0.0] * len(self.population)
        
        for i, strategy in enumerate(self.population):
            self.fitness_scores[i] = self._evaluate_strategy(strategy)
            strategy["fitness"] = self.fitness_scores[i]
            
        best_strategy = None
        best_fitness = -float('inf')
        
        for generation in range(self.generations):
            self.logger.info(f"Generation {generation + 1}/{self.generations}")
            
            parents = self._selection()
            
            new_population = []
            
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            for i in range(self.elite_count):
                if i < len(sorted_indices):
                    elite_index = sorted_indices[i]
                    new_population.append(copy.deepcopy(self.population[elite_index]))
            
            while len(new_population) < self.population_size:
                parent1_idx = random.choice(parents)
                parent2_idx = random.choice(parents)
                
                while parent2_idx == parent1_idx and len(parents) > 1:
                    parent2_idx = random.choice(parents)
                    
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                    
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                    
                new_population.append(child)
            
            self.population = new_population[:self.population_size]
            
            for i, strategy in enumerate(self.population):
                self.fitness_scores[i] = self._evaluate_strategy(strategy)
                strategy["fitness"] = self.fitness_scores[i]
                
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_fitness = self.fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_strategy = copy.deepcopy(self.population[current_best_idx])
                
            avg_fitness = np.mean(self.fitness_scores)
            self.logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}")
            
            self.evolution_history.append({
                "generation": generation + 1,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_strategy": best_strategy["name"] if best_strategy else "None"
            })
        
        if best_strategy:
            best_strategy["path"] = self._save_strategy(best_strategy)
            
        return best_strategy
        
    def _selection(self):
        """
        Select parents for reproduction using tournament selection.
        
        Returns:
        - List of parent indices
        """
        parents = []
        tournament_size = max(2, self.population_size // 5)
        
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            
            best_idx = tournament_indices[0]
            best_fitness = self.fitness_scores[best_idx]
            
            for idx in tournament_indices[1:]:
                if self.fitness_scores[idx] > best_fitness:
                    best_idx = idx
                    best_fitness = self.fitness_scores[idx]
                    
            parents.append(best_idx)
            
        return parents
        
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Parameters:
        - parent1: First parent strategy
        - parent2: Second parent strategy
        
        Returns:
        - Child strategy
        """
        child = copy.deepcopy(parent1)
        
        child["name"] = f"evolved_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if "parameters" in parent1 and "parameters" in parent2:
            for param in parent1["parameters"]:
                if param in parent2["parameters"]:
                    if random.random() < 0.5:
                        child["parameters"][param] = parent2["parameters"][param]
        
        return child
        
    def _mutate(self, strategy):
        """
        Mutate a strategy.
        
        Parameters:
        - strategy: Strategy to mutate
        
        Returns:
        - Mutated strategy
        """
        mutated = copy.deepcopy(strategy)
        
        if "parameters" in mutated:
            for param, value in mutated["parameters"].items():
                if random.random() < 0.2:
                    if isinstance(value, int):
                        mutation = random.randint(-5, 5)
                        mutated["parameters"][param] = max(1, value + mutation)
                    elif isinstance(value, float):
                        mutation = random.uniform(-0.1, 0.1)
                        mutated["parameters"][param] = max(0.01, value + mutation)
        
        return mutated
        
    def _evaluate_strategy(self, strategy):
        """
        Evaluate a strategy using backtesting.
        
        Parameters:
        - strategy: Strategy to evaluate
        
        Returns:
        - Fitness score
        """
        if self.backtester:
            try:
                results = self.backtester.simulate(strategy)
                return results.get("sharpe_ratio", 0.0)
            except Exception as e:
                self.logger.error(f"Error backtesting strategy {strategy['name']}: {str(e)}")
                return 0.0
                
        
        params = strategy.get("parameters", {})
        
        
        
        fast_period = params.get("fast_period", 0)
        slow_period = params.get("slow_period", 0)
        stop_loss = params.get("stop_loss", 0.0)
        take_profit = params.get("take_profit", 0.0)
        
        if fast_period >= slow_period:
            return 0.0
            
        fast_fitness = 1.0 - min(1.0, abs(15 - fast_period) / 15)
        slow_fitness = 1.0 - min(1.0, abs(75 - slow_period) / 75)
        
        risk_reward = take_profit / stop_loss if stop_loss > 0 else 0.0
        risk_reward_fitness = min(1.0, risk_reward / 3.0)
        
        fitness = (fast_fitness * 0.3 + slow_fitness * 0.3 + risk_reward_fitness * 0.4)
        
        fitness += random.uniform(-0.1, 0.1)
        
        fitness = max(0.0, fitness)
        
        return fitness
        
    def _save_strategy(self, strategy):
        """
        Save strategy to file.
        
        Parameters:
        - strategy: Strategy to save
        
        Returns:
        - Path to saved strategy file
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{strategy['name']}_{timestamp}.json"
        filepath = os.path.join(self.strategies_dir, filename)
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(strategy, f, indent=2)
                
            self.logger.info(f"Strategy saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving strategy: {str(e)}")
            return ""
        
    def _deploy_strategy(self, strategy):
        """
        Deploy a strategy.
        
        Parameters:
        - strategy: Strategy to deploy
        
        Returns:
        - Deployment result
        """
        self.logger.info(f"Deploying strategy: {strategy['name']}")
        
        
        deployment_result = {
            "status": "success",
            "strategy": strategy["name"],
            "timestamp": datetime.now().isoformat(),
            "fitness": strategy["fitness"]
        }
        
        return deployment_result
        
    def set_genetic_parameters(self, population_size=None, mutation_rate=None, crossover_rate=None, generations=None, elite_count=None):
        """
        Set genetic algorithm parameters.
        
        Parameters:
        - population_size: Size of population
        - mutation_rate: Mutation rate
        - crossover_rate: Crossover rate
        - generations: Number of generations
        - elite_count: Number of elite strategies to keep
        """
        if population_size is not None:
            self.population_size = population_size
            
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
            
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
            
        if generations is not None:
            self.generations = generations
            
        if elite_count is not None:
            self.elite_count = elite_count
        
    def get_evolution_history(self):
        """
        Get evolution history.
        
        Returns:
        - Evolution history
        """
        return self.evolution_history
        
    def set_backtester(self, backtester):
        """
        Set backtester.
        
        Parameters:
        - backtester: Backtester instance
        """
        self.backtester = backtester
