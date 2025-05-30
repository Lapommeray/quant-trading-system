import numpy as np
import pandas as pd
import cvxpy as cp
import logging
from datetime import datetime

def black_litterman_optimization(market_weights, cov_matrix, P=None, Q=None, risk_aversion=2.5, tau=0.05, constraints=None):
    """
    Wrapper function for Black-Litterman optimization to maintain compatibility
    
    Args:
        market_weights: Array of market capitalization weights
        cov_matrix: Covariance matrix of asset returns
        P: View matrix (k x n) where k is number of views and n is number of assets
        Q: View returns (k x 1)
        risk_aversion: Risk aversion parameter
        tau: Scaling parameter for uncertainty in prior
        constraints: Dictionary of constraints
        
    Returns:
        Array of optimal portfolio weights
    """
    optimizer = BlackLittermanOptimizer(risk_aversion=risk_aversion, tau=tau)
    
    if P is None or Q is None:
        implied_returns = optimizer.market_implied_returns(market_weights, cov_matrix)
        return optimizer.optimize_portfolio(implied_returns, cov_matrix, constraints)
    else:
        return optimizer.optimize_with_views(market_weights, cov_matrix, P, Q, constraints)

class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization with institutional-grade implementation
    """
    def __init__(self, risk_aversion=2.5, tau=0.05):
        self.risk_aversion = risk_aversion  # Risk aversion parameter
        self.tau = tau  # Scaling parameter for uncertainty in prior
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = []
        
    def market_implied_returns(self, market_weights, cov_matrix):
        """
        Calculate market implied returns using reverse optimization
        
        Args:
            market_weights: Array of market capitalization weights
            cov_matrix: Covariance matrix of asset returns
            
        Returns:
            Array of market implied returns
        """
        try:
            market_weights = np.array(market_weights)
            market_weights = market_weights / np.sum(market_weights)
            
            implied_returns = self.risk_aversion * np.dot(cov_matrix, market_weights)
            
            return implied_returns
        except Exception as e:
            self.logger.error(f"Error calculating market implied returns: {str(e)}")
            return np.zeros_like(market_weights)
            
    def incorporate_views(self, prior_returns, cov_matrix, P, Q, omega=None):
        """
        Incorporate investor views into prior returns
        
        Args:
            prior_returns: Array of prior expected returns
            cov_matrix: Covariance matrix of asset returns
            P: View matrix (k x n) where k is number of views and n is number of assets
            Q: View returns (k x 1)
            omega: Uncertainty matrix for views (k x k), if None, calculated from tau and P
            
        Returns:
            Array of posterior expected returns
        """
        try:
            n = len(prior_returns)
            k = len(Q)
            
            prior_returns = np.array(prior_returns).reshape(-1)
            P = np.array(P)
            Q = np.array(Q).reshape(-1)
            
            if omega is None:
                omega = np.diag(np.diag(self.tau * np.dot(np.dot(P, cov_matrix), P.T)))
                
            
            prior_precision = np.linalg.inv(self.tau * cov_matrix)
            omega_precision = np.linalg.inv(omega)
            
            posterior_precision = prior_precision + np.dot(np.dot(P.T, omega_precision), P)
            
            posterior_cov = np.linalg.inv(posterior_precision)
            
            term1 = np.dot(prior_precision, prior_returns)
            term2 = np.dot(np.dot(P.T, omega_precision), Q)
            posterior_returns = np.dot(posterior_cov, term1 + term2)
            
            return posterior_returns
        except Exception as e:
            self.logger.error(f"Error incorporating views: {str(e)}")
            return prior_returns
            
    def optimize_portfolio(self, expected_returns, cov_matrix, constraints=None):
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix of asset returns
            constraints: Dictionary of constraints (default: None)
                - min_weight: Minimum weight for each asset
                - max_weight: Maximum weight for each asset
                - target_return: Target portfolio return
                
        Returns:
            Array of optimal portfolio weights
        """
        try:
            n = len(expected_returns)
            
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 1.0
                }
                
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
            
            weights = cp.Variable(n)
            
            portfolio_return = expected_returns @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
            
            constraints_list = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= min_weight,  # Minimum weight constraint
                weights <= max_weight   # Maximum weight constraint
            ]
            
            if 'target_return' in constraints:
                constraints_list.append(portfolio_return >= constraints['target_return'])
                
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == 'optimal':
                optimal_weights = weights.value
                
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                
                result = {
                    'weights': optimal_weights,
                    'expected_return': float(portfolio_return.value),
                    'expected_risk': float(np.sqrt(portfolio_risk.value)),
                    'sharpe_ratio': float(portfolio_return.value / np.sqrt(portfolio_risk.value)) if portfolio_risk.value > 0 else 0.0,
                    'timestamp': datetime.now()
                }
                
                self.history.append(result)
                
                return optimal_weights
            else:
                self.logger.error(f"Optimization failed with status: {problem.status}")
                return np.ones(n) / n  # Equal weights as fallback
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return np.ones(n) / n  # Equal weights as fallback
            
    def optimize_with_views(self, market_weights, cov_matrix, P, Q, constraints=None):
        """
        Optimize portfolio using Black-Litterman model with investor views
        
        Args:
            market_weights: Array of market capitalization weights
            cov_matrix: Covariance matrix of asset returns
            P: View matrix (k x n) where k is number of views and n is number of assets
            Q: View returns (k x 1)
            constraints: Dictionary of constraints (default: None)
                
        Returns:
            Array of optimal portfolio weights
        """
        try:
            implied_returns = self.market_implied_returns(market_weights, cov_matrix)
            
            posterior_returns = self.incorporate_views(implied_returns, cov_matrix, P, Q)
            
            optimal_weights = self.optimize_portfolio(posterior_returns, cov_matrix, constraints)
            
            return optimal_weights
        except Exception as e:
            self.logger.error(f"Error optimizing with views: {str(e)}")
            return np.array(market_weights)  # Market weights as fallback
            
    def get_optimization_history(self):
        """
        Get optimization history
        
        Returns:
            List of optimization results
        """
        return self.history
