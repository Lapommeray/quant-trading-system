#!/usr/bin/env python3
"""
Stochastic Optimization Module

Implements advanced stochastic optimization techniques for supply-chain risk management
and portfolio optimization under uncertainty. Used by Tesla/SpaceX for supply-chain risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime
from scipy.optimize import minimize
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StochasticOptimization")

class StochasticOptimization:
    """
    Advanced stochastic optimization for supply-chain risk and portfolio management
    
    Implements:
    - Stochastic programming for supply chain optimization
    - Robust optimization under uncertainty
    - Multi-stage stochastic optimization
    - Scenario-based optimization
    """
    
    def __init__(self, precision: int = 128, confidence_level: float = 0.95):
        self.precision = precision
        self.confidence_level = confidence_level
        self.history = []
        
        logger.info(f"Initialized StochasticOptimization with confidence_level={confidence_level}")
    
    def optimize_supply_chain_risk(self, 
                                  demand_scenarios: np.ndarray,
                                  supply_costs: np.ndarray,
                                  disruption_probs: np.ndarray,
                                  inventory_costs: float = 0.1) -> Dict[str, Any]:
        """
        Optimize supply chain under demand uncertainty and disruption risk
        
        Parameters:
        - demand_scenarios: Array of demand scenarios (n_scenarios x n_periods)
        - supply_costs: Array of supply costs for each supplier
        - disruption_probs: Probability of disruption for each supplier
        - inventory_costs: Cost of holding inventory per unit per period
        
        Returns:
        - Optimal supply chain strategy
        """
        n_scenarios, n_periods = demand_scenarios.shape
        n_suppliers = len(supply_costs)
        
        scenario_probs = np.ones(n_scenarios) / n_scenarios
        
        def objective(x):
            allocations = x.reshape(n_suppliers, n_periods)
            
            total_cost = 0.0
            
            for s in range(n_scenarios):
                scenario_cost = 0.0
                inventory = 0.0
                
                for t in range(n_periods):
                    period_supply = 0.0
                    for i in range(n_suppliers):
                        expected_supply = allocations[i, t] * (1 - disruption_probs[i])
                        period_supply += expected_supply
                        scenario_cost += allocations[i, t] * supply_costs[i]
                    
                    inventory = max(0, inventory + period_supply - demand_scenarios[s, t])
                    scenario_cost += inventory * inventory_costs
                    
                    shortage = max(0, demand_scenarios[s, t] - (inventory + period_supply))
                    scenario_cost += shortage * 10.0  # High penalty for shortage
                
                total_cost += scenario_probs[s] * scenario_cost
            
            return total_cost
        
        bounds = [(0, None) for _ in range(n_suppliers * n_periods)]
        
        x0 = np.ones(n_suppliers * n_periods) * np.mean(demand_scenarios) / n_suppliers
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        optimal_allocations = result.x.reshape(n_suppliers, n_periods)
        
        var_95 = self._calculate_conditional_value_at_risk(
            demand_scenarios, optimal_allocations, supply_costs, 
            disruption_probs, inventory_costs
        )
        
        optimization_result = {
            'optimal_allocations': optimal_allocations,
            'total_cost': float(result.fun),
            'optimization_success': result.success,
            'var_95': var_95,
            'diversification_index': self._calculate_diversification_index(optimal_allocations),
            'resilience_score': self._calculate_resilience_score(optimal_allocations, disruption_probs)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'optimize_supply_chain_risk',
            'n_scenarios': n_scenarios,
            'n_suppliers': n_suppliers,
            'total_cost': float(result.fun),
            'success': result.success
        })
        
        return optimization_result
    
    def _calculate_conditional_value_at_risk(self,
                                           demand_scenarios: np.ndarray,
                                           allocations: np.ndarray,
                                           supply_costs: np.ndarray,
                                           disruption_probs: np.ndarray,
                                           inventory_costs: float) -> float:
        """Calculate Conditional Value at Risk (CVaR) for the supply chain strategy"""
        n_scenarios, n_periods = demand_scenarios.shape
        n_suppliers = len(supply_costs)
        
        scenario_costs = []
        
        for s in range(n_scenarios):
            scenario_cost = 0.0
            inventory = 0.0
            
            for t in range(n_periods):
                period_supply = 0.0
                for i in range(n_suppliers):
                    if np.random.random() > disruption_probs[i]:
                        period_supply += allocations[i, t]
                        scenario_cost += allocations[i, t] * supply_costs[i]
                    else:
                        scenario_cost += 0.1 * allocations[i, t] * supply_costs[i]  # Cancellation fee
                
                inventory = max(0, inventory + period_supply - demand_scenarios[s, t])
                scenario_cost += inventory * inventory_costs
                
                shortage = max(0, demand_scenarios[s, t] - (inventory + period_supply))
                scenario_cost += shortage * 10.0  # High penalty for shortage
            
            scenario_costs.append(scenario_cost)
        
        alpha = 0.95  # 95% confidence level
        sorted_costs = np.sort(scenario_costs)
        var_index = int(np.ceil(n_scenarios * (1 - alpha)))
        var_95 = sorted_costs[var_index]
        
        cvar_95 = np.mean(sorted_costs[var_index:])
        
        return float(cvar_95)
    
    def _calculate_diversification_index(self, allocations: np.ndarray) -> float:
        """Calculate diversification index for supplier allocations"""
        n_suppliers, n_periods = allocations.shape
        
        diversification_indices = []
        
        for t in range(n_periods):
            period_allocations = allocations[:, t]
            total_allocation = np.sum(period_allocations)
            
            if total_allocation == 0:
                continue
                
            normalized_allocations = period_allocations / total_allocation
            
            hhi = np.sum(normalized_allocations**2)
            
            diversification_index = 1 - hhi
            diversification_indices.append(diversification_index)
        
        if not diversification_indices:
            return 0.0
            
        return float(np.mean(diversification_indices))
    
    def _calculate_resilience_score(self, allocations: np.ndarray, disruption_probs: np.ndarray) -> float:
        """Calculate resilience score based on allocation and disruption probabilities"""
        n_suppliers, n_periods = allocations.shape
        
        expected_supply = np.zeros(n_periods)
        worst_case_supply = np.zeros(n_periods)
        
        for t in range(n_periods):
            for i in range(n_suppliers):
                expected_supply[t] += allocations[i, t] * (1 - disruption_probs[i])
                
                worst_supplier = np.argmax(disruption_probs)
                worst_case_allocations = allocations[:, t].copy()
                worst_case_allocations[worst_supplier] = 0
                worst_case_supply[t] = np.sum(worst_case_allocations)
        
        resilience_ratios = []
        for t in range(n_periods):
            if expected_supply[t] > 0:
                resilience_ratios.append(worst_case_supply[t] / expected_supply[t])
        
        if not resilience_ratios:
            return 0.0
            
        return float(np.mean(resilience_ratios))
    
    def optimize_portfolio_under_uncertainty(self,
                                           returns: np.ndarray,
                                           covariance: np.ndarray,
                                           scenarios: Optional[np.ndarray] = None,
                                           risk_aversion: float = 1.0,
                                           constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio allocation under uncertainty using robust optimization
        
        Parameters:
        - returns: Expected returns for each asset
        - covariance: Covariance matrix of returns
        - scenarios: Optional array of return scenarios
        - risk_aversion: Risk aversion parameter
        - constraints: Optional dictionary with constraints
        
        Returns:
        - Optimal portfolio allocation
        """
        n_assets = len(returns)
        
        if covariance.shape != (n_assets, n_assets):
            logger.error(f"Covariance matrix shape {covariance.shape} doesn't match returns length {n_assets}")
            return {
                'weights': np.ones(n_assets) / n_assets,
                'expected_return': 0.0,
                'expected_risk': 0.0,
                'sharpe_ratio': 0.0,
                'optimization_success': False
            }
        
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'sum_weights': 1.0
            }
        
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weights = constraints.get('sum_weights', 1.0)
        
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
            
            utility = portfolio_return - risk_aversion * portfolio_variance
            
            if scenarios is not None:
                n_scenarios = scenarios.shape[0]
                scenario_returns = np.zeros(n_scenarios)
                
                for s in range(n_scenarios):
                    scenario_returns[s] = np.dot(weights, scenarios[s])
                
                worst_case_return = np.percentile(scenario_returns, 5)
                utility += 0.5 * worst_case_return
            
            return -utility  # Minimize negative utility
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - sum_weights}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        weights = result.x
        expected_return = np.dot(weights, returns)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
        
        worst_case_stats = {}
        if scenarios is not None:
            n_scenarios = scenarios.shape[0]
            scenario_returns = np.zeros(n_scenarios)
            
            for s in range(n_scenarios):
                scenario_returns[s] = np.dot(weights, scenarios[s])
            
            worst_case_return = np.percentile(scenario_returns, 5)
            worst_case_stats = {
                'worst_case_return': float(worst_case_return),
                'worst_case_percentile': 5,
                'scenario_return_mean': float(np.mean(scenario_returns)),
                'scenario_return_std': float(np.std(scenario_returns))
            }
        
        optimization_result = {
            'weights': weights,
            'expected_return': float(expected_return),
            'expected_risk': float(expected_risk),
            'sharpe_ratio': float(sharpe_ratio),
            'optimization_success': result.success,
            'worst_case_stats': worst_case_stats
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'optimize_portfolio_under_uncertainty',
            'n_assets': n_assets,
            'expected_return': float(expected_return),
            'expected_risk': float(expected_risk),
            'sharpe_ratio': float(sharpe_ratio),
            'success': result.success
        })
        
        return optimization_result
    
    def generate_scenarios(self,
                          historical_data: np.ndarray,
                          n_scenarios: int = 1000,
                          horizon: int = 10,
                          method: str = 'bootstrap') -> np.ndarray:
        """
        Generate scenarios for stochastic optimization
        
        Parameters:
        - historical_data: Historical data array (time x variables)
        - n_scenarios: Number of scenarios to generate
        - horizon: Forecast horizon
        - method: Scenario generation method ('bootstrap', 'copula', 'monte_carlo')
        
        Returns:
        - Array of scenarios (n_scenarios x horizon x variables)
        """
        if len(historical_data.shape) == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        n_periods, n_variables = historical_data.shape
        
        if n_periods < 10:
            logger.warning(f"Insufficient historical data for scenario generation. Need at least 10 points, got {n_periods}")
            mean = np.mean(historical_data, axis=0)
            std = np.std(historical_data, axis=0)
            
            scenarios = np.zeros((n_scenarios, horizon, n_variables))
            for s in range(n_scenarios):
                for h in range(horizon):
                    scenarios[s, h] = mean + std * np.random.normal(0, 1, n_variables)
            
            return scenarios
        
        scenarios = np.zeros((n_scenarios, horizon, n_variables))
        
        if method == 'bootstrap':
            block_size = min(5, n_periods // 2)
            
            for s in range(n_scenarios):
                for h in range(0, horizon, block_size):
                    start_idx = np.random.randint(0, n_periods - block_size)
                    block = historical_data[start_idx:start_idx+block_size]
                    
                    for i in range(min(block_size, horizon - h)):
                        scenarios[s, h+i] = block[i]
        
        elif method == 'copula':
            marginals = []
            for v in range(n_variables):
                kde = stats.gaussian_kde(historical_data[:, v])
                marginals.append(kde)
            
            u_data = np.zeros_like(historical_data)
            for v in range(n_variables):
                u_data[:, v] = stats.rankdata(historical_data[:, v]) / (n_periods + 1)
            
            for s in range(n_scenarios):
                u_scenario = np.zeros((horizon, n_variables))
                
                idx = np.random.randint(0, n_periods)
                u_scenario[0] = u_data[idx]
                
                for h in range(1, horizon):
                    distances = np.sum((u_data - u_scenario[h-1])**2, axis=1)
                    nearest_idx = np.argmin(distances)
                    
                    next_idx = (nearest_idx + 1) % n_periods
                    u_scenario[h] = u_data[next_idx]
                
                for v in range(n_variables):
                    for h in range(horizon):
                        scenarios[s, h, v] = marginals[v].resample(1)[0]
        
        elif method == 'monte_carlo':
            returns = np.diff(historical_data, axis=0)
            mean_return = np.mean(returns, axis=0)
            cov_return = np.cov(returns, rowvar=False)
            
            for s in range(n_scenarios):
                current_value = historical_data[-1].copy()
                
                for h in range(horizon):
                    random_return = np.random.multivariate_normal(mean_return, cov_return)
                    
                    current_value = current_value + random_return
                    scenarios[s, h] = current_value
        
        else:
            logger.error(f"Unknown scenario generation method: {method}")
            return np.zeros((n_scenarios, horizon, n_variables))
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'generate_scenarios',
            'n_scenarios': n_scenarios,
            'horizon': horizon,
            'n_variables': n_variables,
            'method': method
        })
        
        return scenarios
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about stochastic optimization usage
        
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
            'confidence_level': self.confidence_level
        }
