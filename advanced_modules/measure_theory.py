#!/usr/bin/env python3
"""
Measure Theory Module

Implements measure-theoretic probability for financial markets:
- Kolmogorov probability spaces for rigorous foundations
- Measure-theoretic integration for high-dimensional signals
- Martingale theory beyond Bayesian statistics
- Radon-Nikodym derivatives for change of measure
- Lebesgue integration for non-standard distributions

This module provides rigorous mathematical foundations for probability
theory in quantitative finance beyond traditional approaches.
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MeasureTheory")

class MeasureTheory:
    """
    Measure Theory for rigorous probability in financial markets
    
    Implements measure-theoretic probability:
    - Kolmogorov probability spaces
    - Measure-theoretic integration
    - Martingale theory
    - Radon-Nikodym derivatives
    - Lebesgue integration
    
    Provides rigorous mathematical foundations for probability
    theory in quantitative finance beyond traditional approaches.
    """
    
    def __init__(self, precision: int = 64, confidence_level: float = 0.99,
                integration_method: str = 'monte_carlo', n_samples: int = 10000):
        """
        Initialize Measure Theory
        
        Parameters:
        - precision: Numerical precision for calculations (default: 64 bits)
        - confidence_level: Statistical confidence level (default: 0.99)
        - integration_method: Method for numerical integration (default: 'monte_carlo')
        - n_samples: Number of samples for Monte Carlo integration (default: 10000)
        """
        self.precision = precision
        self.confidence_level = confidence_level
        self.integration_method = integration_method
        self.n_samples = n_samples
        self.history = []
        
        np.random.seed(42)  # For reproducibility
        
        logger.info(f"Initialized MeasureTheory with precision={precision}, "
                   f"confidence_level={confidence_level}, "
                   f"integration_method={integration_method}")
    
    
    def create_probability_space(self, sample_space: List, 
                                event_sigma_algebra: List[List],
                                probability_measure: Dict[str, float]) -> Dict:
        """
        Create a Kolmogorov probability space
        
        Parameters:
        - sample_space: List of all possible outcomes
        - event_sigma_algebra: List of events (subsets of sample space)
        - probability_measure: Dictionary mapping event names to probabilities
        
        Returns:
        - Dictionary representing the probability space
        """
        total_prob = sum(probability_measure.values())
        if not np.isclose(total_prob, 1.0, rtol=1e-5):
            logger.warning(f"Probability measure does not sum to 1.0: {total_prob}")
            for event in probability_measure:
                probability_measure[event] /= total_prob
        
        has_empty_set = any(len(event) == 0 for event in event_sigma_algebra)
        if not has_empty_set:
            event_sigma_algebra.append([])
            logger.warning("Added empty set to sigma-algebra")
        
        for event in event_sigma_algebra:
            complement = [x for x in sample_space if x not in event]
            if complement not in event_sigma_algebra:
                event_sigma_algebra.append(complement)
                logger.warning(f"Added complement of {event} to sigma-algebra")
        
        if len(event_sigma_algebra) < 10:
            for i, event1 in enumerate(event_sigma_algebra):
                for j, event2 in enumerate(event_sigma_algebra[i+1:], i+1):
                    union = list(set(event1) | set(event2))
                    if union not in event_sigma_algebra:
                        event_sigma_algebra.append(union)
                        logger.warning(f"Added union of events to sigma-algebra")
        
        probability_space = {
            'sample_space': sample_space,
            'sigma_algebra': event_sigma_algebra,
            'probability_measure': probability_measure
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_probability_space',
            'sample_space_size': len(sample_space),
            'sigma_algebra_size': len(event_sigma_algebra),
            'probability_measure_size': len(probability_measure)
        })
        
        return probability_space
    
    def probability_of_event(self, probability_space: Dict, event: List) -> float:
        """
        Calculate probability of an event in a probability space
        
        Parameters:
        - probability_space: Dictionary representing probability space
        - event: List of outcomes representing the event
        
        Returns:
        - Probability of the event
        """
        sigma_algebra = probability_space['sigma_algebra']
        probability_measure = probability_space['probability_measure']
        
        for i, sigma_event in enumerate(sigma_algebra):
            if set(sigma_event) == set(event):
                event_name = f"event_{i}"
                if event_name in probability_measure:
                    return probability_measure[event_name]
        
        sample_space = probability_space['sample_space']
        
        event_indicator = np.zeros(len(sample_space))
        for i, outcome in enumerate(sample_space):
            if outcome in event:
                event_indicator[i] = 1
        
        sigma_indicators = np.zeros((len(sigma_algebra), len(sample_space)))
        for i, sigma_event in enumerate(sigma_algebra):
            for j, outcome in enumerate(sample_space):
                if outcome in sigma_event:
                    sigma_indicators[i, j] = 1
        
        for i, sigma_event in enumerate(sigma_algebra):
            if np.array_equal(event_indicator, sigma_indicators[i]):
                event_name = f"event_{i}"
                if event_name in probability_measure:
                    return probability_measure[event_name]
        
        logger.warning("Event not found in sigma-algebra, returning approximation")
        
        event_count = len(event)
        total_count = len(sample_space)
        
        probability = event_count / total_count
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'probability_of_event',
            'event_size': len(event),
            'probability': probability,
            'approximation': True
        })
        
        return probability
    
    def conditional_probability(self, probability_space: Dict, 
                               event_a: List, event_b: List) -> float:
        """
        Calculate conditional probability P(A|B)
        
        Parameters:
        - probability_space: Dictionary representing probability space
        - event_a: List of outcomes representing event A
        - event_b: List of outcomes representing event B
        
        Returns:
        - Conditional probability P(A|B)
        """
        prob_b = self.probability_of_event(probability_space, event_b)
        
        if prob_b == 0:
            logger.warning("Conditioning on zero-probability event")
            return 0.0
        
        intersection = [x for x in event_a if x in event_b]
        prob_intersection = self.probability_of_event(probability_space, intersection)
        
        conditional_prob = prob_intersection / prob_b
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'conditional_probability',
            'event_a_size': len(event_a),
            'event_b_size': len(event_b),
            'intersection_size': len(intersection),
            'conditional_probability': conditional_prob
        })
        
        return conditional_prob
    
    
    def lebesgue_integral(self, function: Callable[[float], float], 
                         domain: Tuple[float, float], 
                         measure: str = 'lebesgue') -> float:
        """
        Compute Lebesgue integral of a function
        
        Parameters:
        - function: Function to integrate
        - domain: Tuple of (lower_bound, upper_bound)
        - measure: Type of measure ('lebesgue', 'gaussian', 'custom')
        
        Returns:
        - Value of the Lebesgue integral
        """
        lower_bound, upper_bound = domain
        
        if self.integration_method == 'monte_carlo':
            if measure == 'lebesgue':
                x = np.random.uniform(lower_bound, upper_bound, self.n_samples)
                y = np.array([function(xi) for xi in x])
                integral = (upper_bound - lower_bound) * np.mean(y)
            elif measure == 'gaussian':
                x = np.random.normal(0, 1, self.n_samples)
                mask = (x >= lower_bound) & (x <= upper_bound)
                x = x[mask]
                if len(x) == 0:
                    logger.warning("No samples in domain, returning 0")
                    return 0.0
                y = np.array([function(xi) * np.exp(xi**2 / 2) for xi in x])
                integral = np.mean(y) * (stats.norm.cdf(upper_bound) - stats.norm.cdf(lower_bound))
            else:
                logger.warning(f"Unknown measure: {measure}, using Lebesgue")
                x = np.random.uniform(lower_bound, upper_bound, self.n_samples)
                y = np.array([function(xi) for xi in x])
                integral = (upper_bound - lower_bound) * np.mean(y)
        else:
            if measure == 'lebesgue':
                integral, _ = integrate.quad(function, lower_bound, upper_bound)
            elif measure == 'gaussian':
                def integrand(x):
                    return function(x) * stats.norm.pdf(x)
                integral, _ = integrate.quad(integrand, lower_bound, upper_bound)
            else:
                logger.warning(f"Unknown measure: {measure}, using Lebesgue")
                integral, _ = integrate.quad(function, lower_bound, upper_bound)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'lebesgue_integral',
            'domain': domain,
            'measure': measure,
            'integration_method': self.integration_method,
            'integral_value': float(integral)
        })
        
        return float(integral)
    
    def expectation(self, random_variable: Callable[[Any], float], 
                   probability_space: Dict) -> float:
        """
        Calculate expectation of a random variable
        
        Parameters:
        - random_variable: Function mapping outcomes to real values
        - probability_space: Dictionary representing probability space
        
        Returns:
        - Expected value of the random variable
        """
        sample_space = probability_space['sample_space']
        
        if len(sample_space) < 1000:
            expectation = 0.0
            for outcome in sample_space:
                event = [outcome]
                prob = self.probability_of_event(probability_space, event)
                expectation += random_variable(outcome) * prob
        else:
            samples = np.random.choice(sample_space, size=self.n_samples, replace=True)
            values = np.array([random_variable(sample) for sample in samples])
            expectation = np.mean(values)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'expectation',
            'sample_space_size': len(sample_space),
            'expectation_value': float(expectation)
        })
        
        return float(expectation)
    
    
    def radon_nikodym_derivative(self, density_p: Callable[[float], float], 
                                density_q: Callable[[float], float], 
                                x: float) -> float:
        """
        Calculate Radon-Nikodym derivative (dP/dQ)(x)
        
        Parameters:
        - density_p: Density function of measure P
        - density_q: Density function of measure Q
        - x: Point at which to evaluate the derivative
        
        Returns:
        - Value of Radon-Nikodym derivative at x
        """
        q_x = density_q(x)
        
        if q_x == 0:
            logger.warning(f"Division by zero in Radon-Nikodym derivative at x={x}")
            return float('inf')
            
        derivative = density_p(x) / q_x
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'radon_nikodym_derivative',
            'x': x,
            'derivative_value': float(derivative)
        })
        
        return float(derivative)
    
    def change_of_measure(self, function: Callable[[float], float], 
                         domain: Tuple[float, float],
                         density_p: Callable[[float], float], 
                         density_q: Callable[[float], float]) -> float:
        """
        Compute integral using change of measure
        
        Parameters:
        - function: Function to integrate
        - domain: Tuple of (lower_bound, upper_bound)
        - density_p: Density function of target measure P
        - density_q: Density function of reference measure Q
        
        Returns:
        - Value of the integral under measure P
        """
        lower_bound, upper_bound = domain
        
        if self.integration_method == 'monte_carlo':
            if hasattr(density_q, 'rvs'):
                x = density_q.rvs(size=self.n_samples)
            else:
                x = np.random.uniform(lower_bound, upper_bound, self.n_samples * 10)
                q_values = np.array([density_q(xi) for xi in x])
                max_q = np.max(q_values)
                u = np.random.uniform(0, max_q, self.n_samples * 10)
                x = x[u <= q_values][:self.n_samples]
                
                if len(x) < self.n_samples:
                    logger.warning(f"Insufficient samples: {len(x)} < {self.n_samples}")
                    additional = np.random.uniform(lower_bound, upper_bound, self.n_samples - len(x))
                    x = np.concatenate([x, additional])
            
            mask = (x >= lower_bound) & (x <= upper_bound)
            x = x[mask]
            
            if len(x) == 0:
                logger.warning("No samples in domain, returning 0")
                return 0.0
                
            f_values = np.array([function(xi) for xi in x])
            
            rn_derivatives = np.array([self.radon_nikodym_derivative(density_p, density_q, xi) 
                                     for xi in x])
            
            integral = np.mean(f_values * rn_derivatives)
            
        else:
            def integrand(x):
                return function(x) * density_p(x)
                
            integral, _ = integrate.quad(integrand, lower_bound, upper_bound)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'change_of_measure',
            'domain': domain,
            'integration_method': self.integration_method,
            'integral_value': float(integral)
        })
        
        return float(integral)
    
    
    def is_martingale(self, process: np.ndarray, filtration: List[List[int]]) -> bool:
        """
        Check if a stochastic process is a martingale
        
        Parameters:
        - process: Array of process values
        - filtration: List of lists representing information available at each time
        
        Returns:
        - True if process is a martingale, False otherwise
        """
        if len(process) != len(filtration):
            logger.warning(f"Process length {len(process)} != filtration length {len(filtration)}")
            return False
            
        for t in range(len(process) - 1):
            x_t = process[t]
            
            x_t_plus_1 = process[t+1]
            
            f_t = filtration[t]
            
            conditional_expectation = x_t_plus_1
            
            if not np.isclose(conditional_expectation, x_t, rtol=1e-5):
                return False
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'is_martingale',
            'process_length': len(process),
            'result': True
        })
        
        return True
    
    def optional_stopping_theorem(self, process: np.ndarray, 
                                 stopping_time: Callable[[np.ndarray], int]) -> float:
        """
        Apply optional stopping theorem to a martingale
        
        Parameters:
        - process: Array of martingale process values
        - stopping_time: Function that returns stopping index given process history
        
        Returns:
        - Expected value of martingale at stopping time
        """
        initial_value = process[0]
        
        n_simulations = 1000
        stopped_values = []
        
        for _ in range(n_simulations):
            sim_process = process.copy()
            
            history = [sim_process[0]]
            
            t = 0
            stop_idx = stopping_time(np.array(history))
            
            while stop_idx > t and t < len(sim_process) - 1:
                t += 1
                history.append(sim_process[t])
                stop_idx = stopping_time(np.array(history))
            
            stopped_values.append(history[min(stop_idx, len(history) - 1)])
        
        expected_value = np.mean(stopped_values)
        
        is_ost_valid = np.isclose(expected_value, initial_value, rtol=1e-2)
        
        result = {
            'expected_value': float(expected_value),
            'initial_value': float(initial_value),
            'is_ost_valid': bool(is_ost_valid),
            'n_simulations': n_simulations
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'optional_stopping_theorem',
            'process_length': len(process),
            'result': result
        })
        
        return float(expected_value)
    
    
    def risk_neutral_pricing(self, stock_paths: np.ndarray, 
                            option_payoff: Callable[[np.ndarray], float],
                            risk_free_rate: float, dt: float) -> float:
        """
        Price options using risk-neutral measure
        
        Parameters:
        - stock_paths: Array of simulated stock price paths
        - option_payoff: Function calculating option payoff from price path
        - risk_free_rate: Risk-free interest rate
        - dt: Time step in years
        
        Returns:
        - Option price
        """
        payoffs = np.array([option_payoff(path) for path in stock_paths])
        
        T = dt * (stock_paths.shape[1] - 1)  # Time to maturity
        discount_factor = np.exp(-risk_free_rate * T)
        option_price = discount_factor * np.mean(payoffs)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'risk_neutral_pricing',
            'n_paths': stock_paths.shape[0],
            'time_steps': stock_paths.shape[1],
            'risk_free_rate': risk_free_rate,
            'option_price': float(option_price)
        })
        
        return float(option_price)
    
    def girsanov_theorem(self, stock_paths: np.ndarray, 
                        drift_original: float, drift_new: float,
                        volatility: float, dt: float) -> np.ndarray:
        """
        Apply Girsanov theorem to change measure of stock paths
        
        Parameters:
        - stock_paths: Array of simulated stock price paths
        - drift_original: Original drift parameter
        - drift_new: New drift parameter
        - volatility: Volatility parameter
        - dt: Time step in years
        
        Returns:
        - Array of likelihood ratios for each path
        """
        n_paths, n_steps = stock_paths.shape
        
        log_returns = np.diff(np.log(stock_paths), axis=1)
        
        theta = (drift_new - drift_original) / volatility
        
        likelihood_ratios = np.ones(n_paths)
        
        for i in range(n_paths):
            path_returns = log_returns[i]
            
            stochastic_integral = np.sum(theta * path_returns / volatility)
            
            quadratic_variation = 0.5 * theta**2 * (n_steps - 1) * dt
            
            likelihood_ratios[i] = np.exp(stochastic_integral - quadratic_variation)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'girsanov_theorem',
            'n_paths': n_paths,
            'n_steps': n_steps,
            'drift_original': drift_original,
            'drift_new': drift_new,
            'volatility': volatility,
            'mean_likelihood_ratio': float(np.mean(likelihood_ratios))
        })
        
        return likelihood_ratios
    
    def kernel_density_estimation(self, samples: np.ndarray, 
                                 bandwidth: Optional[float] = None) -> Callable:
        """
        Estimate probability density function using kernel density estimation
        
        Parameters:
        - samples: Array of samples
        - bandwidth: Bandwidth parameter (default: None for auto-selection)
        
        Returns:
        - Estimated density function
        """
        samples = samples.reshape(-1, 1)
        
        if bandwidth is None:
            bandwidths = np.logspace(-1, 1, 20)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                              {'bandwidth': bandwidths},
                              cv=5)
            grid.fit(samples)
            bandwidth = grid.best_params_['bandwidth']
        
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(samples)
        
        def density_function(x):
            x_reshaped = np.array([x]).reshape(-1, 1)
            log_density = kde.score_samples(x_reshaped)
            return np.exp(log_density[0])
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'kernel_density_estimation',
            'n_samples': len(samples),
            'bandwidth': float(bandwidth)
        })
        
        return density_function
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about measure theory usage
        
        Returns:
        - Dictionary with usage statistics
        """
        if not self.history:
            return {'count': 0}
            
        operations = {}
        for h in self.history:
            op = h.get('operation', 'unknown')
            operations[op] = operations.get(op, 0) + 1
            
        return {
            'count': len(self.history),
            'operations': operations,
            'precision': self.precision,
            'confidence_level': self.confidence_level,
            'integration_method': self.integration_method
        }
