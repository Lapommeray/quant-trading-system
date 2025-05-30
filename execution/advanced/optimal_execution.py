"""
Optimal Execution using Almgren-Chriss Model

Implementation of optimal execution strategies using convex optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available. OptimalExecution will have limited functionality.")

class OptimalExecution:
    """
    Optimal execution using Almgren-Chriss model with convex optimization.
    """
    
    def __init__(self, risk_aversion: float = 1e-6):
        self.risk_aversion = risk_aversion
        self.logger = logging.getLogger('OptimalExecution')
        
    def solve_optimal_strategy(self, 
                             target_shares: float,
                             time_horizon: int,
                             volatility: float,
                             liquidity_param: float = 0.1,
                             temporary_impact: float = 0.01,
                             permanent_impact: float = 0.005) -> Dict[str, Any]:
        """
        Solve for optimal execution strategy.
        
        Parameters:
        - target_shares: Total shares to execute
        - time_horizon: Number of time periods
        - volatility: Asset volatility
        - liquidity_param: Liquidity parameter (higher = less liquid)
        - temporary_impact: Temporary market impact coefficient
        - permanent_impact: Permanent market impact coefficient
        
        Returns:
        - Dictionary with optimal execution schedule
        """
        if not CVXPY_AVAILABLE:
            self.logger.error("cvxpy not available. Cannot solve optimal execution.")
            return self._fallback_strategy(target_shares, time_horizon)
            
        try:
            x = cp.Variable(time_horizon)
            
            constraints = [cp.sum(x) == target_shares]
            
            temporary_cost = cp.sum(temporary_impact * cp.abs(x))
            permanent_cost = cp.sum(permanent_impact * cp.square(x))
            
            remaining_inventory = cp.Variable(time_horizon + 1)
            constraints.append(remaining_inventory[0] == target_shares)
            
            for t in range(time_horizon):
                constraints.append(remaining_inventory[t+1] == remaining_inventory[t] - x[t])
            
            constraints.append(remaining_inventory[time_horizon] == 0)
            
            variance_cost = cp.sum([
                self.risk_aversion * (volatility ** 2) * cp.square(remaining_inventory[t])
                for t in range(time_horizon)
            ])
            
            total_cost = temporary_cost + permanent_cost + variance_cost
            
            problem = cp.Problem(cp.Minimize(total_cost), constraints)
            problem.solve(solver=cp.ECOS)
            
            if problem.status != cp.OPTIMAL:
                self.logger.error(f"Optimization failed with status: {problem.status}")
                return self._fallback_strategy(target_shares, time_horizon)
            
            optimal_trades = x.value
            optimal_inventory = remaining_inventory.value
            
            return {
                'optimal_trades': optimal_trades.tolist(),
                'optimal_inventory': optimal_inventory.tolist(),
                'total_cost': problem.value,
                'temporary_cost': temporary_cost.value,
                'permanent_cost': permanent_cost.value,
                'variance_cost': variance_cost.value,
                'status': 'optimal',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in optimal execution: {str(e)}")
            return self._fallback_strategy(target_shares, time_horizon)
    
    def _fallback_strategy(self, target_shares, time_horizon):
        """Fallback to TWAP when optimization fails"""
        self.logger.warning("Using TWAP fallback strategy")
        
        trades = [target_shares / time_horizon] * time_horizon
        
        inventory = [target_shares]
        for trade in trades:
            inventory.append(inventory[-1] - trade)
            
        return {
            'optimal_trades': trades,
            'optimal_inventory': inventory,
            'status': 'fallback_twap',
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_implementation_shortfall(self,
                                         executed_prices: List[float],
                                         executed_quantities: List[float],
                                         benchmark_price: float) -> Dict[str, float]:
        """
        Calculate implementation shortfall vs benchmark.
        
        Parameters:
        - executed_prices: Prices at which trades were executed
        - executed_quantities: Quantities executed
        - benchmark_price: Benchmark price (e.g., arrival price)
        
        Returns:
        - Implementation shortfall metrics
        """
        if len(executed_prices) != len(executed_quantities):
            raise ValueError("Price and quantity arrays must have same length")
        
        executed_prices = np.array(executed_prices)
        executed_quantities = np.array(executed_quantities)
        
        total_quantity = np.sum(executed_quantities)
        if total_quantity == 0:
            return {'error': 'No executed quantity'}
        
        weighted_avg_price = np.sum(executed_prices * executed_quantities) / total_quantity
        
        shortfall = (weighted_avg_price - benchmark_price) / benchmark_price
        
        return {
            'implementation_shortfall': shortfall,
            'weighted_avg_price': weighted_avg_price,
            'benchmark_price': benchmark_price,
            'total_quantity': total_quantity
        }
