"""
darwinian_ga.py

Strategy Evolver for AgentLab

Evolves trading strategies through genetic algorithms and survival testing.
"""

import numpy as np
from datetime import datetime
import random

class StrategyEvolver:
    """
    Strategy Evolver for QMP Overrider
    
    Evolves trading strategies through genetic algorithms and survival testing.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Strategy Evolver
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.fitness_history = []
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population_size = 50
        self.elitism_count = 5
        self.tournament_size = 3
        self.initialized = False
    
    def initialize(self, initial_parameters=None):
        """
        Initialize the population
        
        Parameters:
        - initial_parameters: Dictionary with initial parameters (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        self.population = []
        
        for i in range(self.population_size):
            individual = self._create_individual(initial_parameters)
            self.population.append(individual)
        
        self.initialized = True
        
        if self.algorithm:
            self.algorithm.Debug(f"Strategy Evolver: Initialized population with {self.population_size} individuals")
        
        return True
    
    def _create_individual(self, initial_parameters=None):
        """
        Create an individual with random parameters
        
        Parameters:
        - initial_parameters: Dictionary with initial parameters (optional)
        
        Returns:
        - Dictionary with individual parameters
        """
        parameter_ranges = {
            "alien_threshold": (0.7, 0.95),
            "cosmic_threshold": (0.7, 0.95),
            "quantum_threshold": (0.7, 0.95),
            "emotion_threshold": (0.5, 0.9),
            "angelic_threshold": (0.3, 0.8),
            "divine_timing_threshold": (0.3, 0.8),
            "sacred_date_threshold": (0.0, 0.5),
            "big_move_threshold": (0.3, 0.8),
            "timeline_threshold": (0.3, 0.8),
            "macro_threshold": (0.5, 0.9),
            "confidence_threshold": (0.6, 0.9),
            "position_size_multiplier": (0.5, 2.0),
            "stop_loss_multiplier": (1.0, 3.0),
            "take_profit_multiplier": (1.0, 3.0)
        }
        
        individual = {}
        
        if initial_parameters:
            for param, value in initial_parameters.items():
                individual[param] = value
        
        for param, (min_val, max_val) in parameter_ranges.items():
            if param not in individual:
                individual[param] = random.uniform(min_val, max_val)
        
        individual["fitness"] = 0.0
        
        return individual
    
    def evaluate_fitness(self, market_data):
        """
        Evaluate fitness of all individuals
        
        Parameters:
        - market_data: Dictionary with market data for fitness evaluation
        
        Returns:
        - List of individuals sorted by fitness
        """
        if not self.initialized:
            self.initialize()
        
        for individual in self.population:
            individual["fitness"] = self._calculate_fitness(individual, market_data)
        
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        self.best_individual = self.population[0].copy()
        
        self.fitness_history.append({
            "generation": self.generation,
            "best_fitness": self.best_individual["fitness"],
            "avg_fitness": sum(ind["fitness"] for ind in self.population) / len(self.population),
            "timestamp": datetime.now()
        })
        
        if self.algorithm:
            self.algorithm.Debug(f"Strategy Evolver: Generation {self.generation}")
            self.algorithm.Debug(f"Best Fitness: {self.best_individual['fitness']:.4f}")
            self.algorithm.Debug(f"Avg Fitness: {self.fitness_history[-1]['avg_fitness']:.4f}")
        
        return self.population
    
    def _calculate_fitness(self, individual, market_data):
        """
        Calculate fitness of an individual
        
        Parameters:
        - individual: Dictionary with individual parameters
        - market_data: Dictionary with market data for fitness evaluation
        
        Returns:
        - Fitness score
        """
        fitness = 0.0
        
        if not market_data:
            return fitness
        
        if "historical_trades" in market_data:
            trades = market_data["historical_trades"]
            
            wins = 0
            losses = 0
            profit = 0.0
            
            for trade in trades:
                would_take = self._would_take_trade(individual, trade)
                
                if would_take:
                    if trade["result"] > 0:
                        wins += 1
                        profit += trade["result"]
                    else:
                        losses += 1
                        profit += trade["result"]
            
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            fitness = profit * 0.7 + win_rate * 0.3
        
        elif "backtest_results" in market_data:
            results = market_data["backtest_results"]
            
            if "sharpe_ratio" in results:
                fitness += results["sharpe_ratio"] * 0.3
            
            if "profit_factor" in results:
                fitness += results["profit_factor"] * 0.3
            
            if "win_rate" in results:
                fitness += results["win_rate"] * 0.2
            
            if "max_drawdown" in results:
                fitness += (1.0 - results["max_drawdown"]) * 0.2
        
        fitness = max(0.0, fitness)
        
        return fitness
    
    def _would_take_trade(self, individual, trade):
        """
        Check if a trade would be taken with individual parameters
        
        Parameters:
        - individual: Dictionary with individual parameters
        - trade: Dictionary with trade information
        
        Returns:
        - True if trade would be taken, False otherwise
        """
        if "gate_scores" not in trade:
            return False
        
        gate_scores = trade["gate_scores"]
        
        passed_all_gates = all([
            gate_scores.get("alien", 0.0) > individual["alien_threshold"],
            gate_scores.get("cosmic", 0.0) > individual["cosmic_threshold"],
            gate_scores.get("quantum", 0.0) > individual["quantum_threshold"],
            gate_scores.get("emotion", 0.0) > individual["emotion_threshold"],
            gate_scores.get("angelic", 0.0) > individual["angelic_threshold"],
            gate_scores.get("divine_timing", 0.0) > individual["divine_timing_threshold"],
            gate_scores.get("sacred_date", 0.0) > individual["sacred_date_threshold"],
            gate_scores.get("big_move", 0.0) > individual["big_move_threshold"],
            gate_scores.get("timeline", 0.0) > individual["timeline_threshold"]
        ])
        
        if not passed_all_gates:
            return False
        
        confidence = trade.get("confidence", 0.0)
        if confidence < individual["confidence_threshold"]:
            return False
        
        return True
    
    def evolve(self):
        """
        Evolve the population to the next generation
        
        Returns:
        - List of individuals in the new generation
        """
        if not self.initialized:
            self.initialize()
        
        new_population = []
        
        for i in range(self.elitism_count):
            if i < len(self.population):
                new_population.append(self.population[i].copy())
        
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        
        self.generation += 1
        
        if self.algorithm:
            self.algorithm.Debug(f"Strategy Evolver: Evolved to generation {self.generation}")
        
        return self.population
    
    def _tournament_selection(self):
        """
        Select an individual using tournament selection
        
        Returns:
        - Selected individual
        """
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        
        return max(tournament, key=lambda x: x["fitness"]).copy()
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Parameters:
        - parent1: First parent
        - parent2: Second parent
        
        Returns:
        - Two children
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        params = [p for p in parent1.keys() if p != "fitness"]
        
        crossover_point = random.randint(1, len(params) - 1)
        
        for i, param in enumerate(params):
            if i >= crossover_point:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        child1["fitness"] = 0.0
        child2["fitness"] = 0.0
        
        return child1, child2
    
    def _mutate(self, individual):
        """
        Mutate an individual
        
        Parameters:
        - individual: Individual to mutate
        
        Returns:
        - Mutated individual
        """
        params = [p for p in individual.keys() if p != "fitness"]
        
        for param in params:
            if random.random() < self.mutation_rate:
                if param.endswith("threshold"):
                    min_val, max_val = 0.0, 1.0
                elif param == "position_size_multiplier":
                    min_val, max_val = 0.1, 3.0
                elif param.endswith("multiplier"):
                    min_val, max_val = 0.5, 5.0
                else:
                    min_val, max_val = 0.0, 1.0
                
                mutation_amount = random.uniform(-0.2, 0.2)
                individual[param] = max(min_val, min(max_val, individual[param] + mutation_amount))
        
        individual["fitness"] = 0.0
        
        return individual
    
    def optimize(self, signal):
        """
        Optimize a signal using the best individual
        
        Parameters:
        - signal: Dictionary with signal information
        
        Returns:
        - Optimized signal
        """
        if not self.initialized or not self.best_individual:
            return signal
        
        optimized = signal.copy()
        
        if "position_size" in optimized:
            optimized["position_size"] *= self.best_individual["position_size_multiplier"]
        
        if "stop_loss" in optimized:
            optimized["stop_loss"] *= self.best_individual["stop_loss_multiplier"]
        
        if "take_profit" in optimized:
            optimized["take_profit"] *= self.best_individual["take_profit_multiplier"]
        
        optimized["optimized"] = True
        optimized["optimizer"] = "darwin"
        optimized["optimizer_generation"] = self.generation
        
        return optimized
    
    def get_best_parameters(self):
        """
        Get the best parameters
        
        Returns:
        - Dictionary with best parameters
        """
        if not self.best_individual:
            return {}
        
        best_params = self.best_individual.copy()
        if "fitness" in best_params:
            del best_params["fitness"]
        
        return best_params
    
    def get_status(self):
        """
        Get Strategy Evolver status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "generation": self.generation,
            "population_size": self.population_size,
            "best_individual": self.best_individual,
            "fitness_history": self.fitness_history[-10:] if self.fitness_history else []
        }
    
    def darwin_cycle(self, market_data):
        """
        Run a complete Darwin cycle (evaluate fitness and evolve)
        
        Parameters:
        - market_data: Dictionary with market data for fitness evaluation
        
        Returns:
        - Dictionary with cycle results
        """
        if not self.initialized:
            self.initialize()
        
        self.evaluate_fitness(market_data)
        
        self.evolve()
        
        return {
            "generation": self.generation,
            "best_fitness": self.best_individual["fitness"] if self.best_individual else 0.0,
            "best_parameters": self.get_best_parameters(),
            "population_size": len(self.population)
        }
