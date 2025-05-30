#!/usr/bin/env python3
"""
Microstructure Modeling Module

Implements advanced microstructure models for limit-order-book dynamics,
bid-ask spread modeling, and high-frequency trading pattern detection.
Used by elite hedge funds for market making and latency arbitrage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MicrostructureModeling")

class MicrostructureModeling:
    """
    Advanced microstructure modeling for limit-order-book dynamics
    
    Implements cutting-edge techniques used by top HFT firms:
    - Hawkes processes for order arrival modeling
    - Kyle's lambda model for price impact
    - Almgren-Chriss optimal execution
    - Order book imbalance models
    """
    
    def __init__(self, precision: int = 128, tick_size: float = 0.01):
        self.precision = precision
        self.tick_size = tick_size
        self.history = []
        
        self.hawkes_mu = 1.0  # Baseline intensity
        self.hawkes_alpha = 0.5  # Self-excitement
        self.hawkes_beta = 2.0  # Decay rate
        
        logger.info(f"Initialized MicrostructureModeling with precision={precision}")
    
    def simulate_limit_order_book(self, 
                                 duration: float, 
                                 dt: float = 0.001,
                                 initial_price: float = 100.0,
                                 spread: float = 0.02) -> Dict[str, Any]:
        """
        Simulate limit order book dynamics using Hawkes processes
        
        Parameters:
        - duration: Simulation duration in seconds
        - dt: Time step size
        - initial_price: Starting mid-price
        - spread: Initial bid-ask spread
        
        Returns:
        - Dictionary with order book simulation results
        """
        n_steps = int(duration / dt)
        times = np.linspace(0, duration, n_steps)
        
        mid_prices = np.zeros(n_steps)
        bid_prices = np.zeros(n_steps)
        ask_prices = np.zeros(n_steps)
        bid_volumes = np.zeros(n_steps)
        ask_volumes = np.zeros(n_steps)
        
        mid_prices[0] = initial_price
        bid_prices[0] = initial_price - spread/2
        ask_prices[0] = initial_price + spread/2
        bid_volumes[0] = np.random.exponential(1000)
        ask_volumes[0] = np.random.exponential(1000)
        
        order_arrivals = self._simulate_hawkes_process(duration, dt)
        
        for i in range(1, n_steps):
            imbalance = (bid_volumes[i-1] - ask_volumes[i-1]) / (bid_volumes[i-1] + ask_volumes[i-1])
            
            price_change = 0.001 * imbalance * np.random.normal(0, 1)
            mid_prices[i] = mid_prices[i-1] + price_change
            
            recent_vol = np.std(np.diff(mid_prices[max(0, i-100):i])) if i > 100 else spread
            dynamic_spread = max(float(self.tick_size), float(spread * (1 + recent_vol)))
            
            bid_prices[i] = mid_prices[i] - dynamic_spread/2
            ask_prices[i] = mid_prices[i] + dynamic_spread/2
            
            bid_volumes[i] = max(0, bid_volumes[i-1] * 0.99 + np.random.exponential(100))
            ask_volumes[i] = max(0, ask_volumes[i-1] * 0.99 + np.random.exponential(100))
            
            if i < len(order_arrivals):
                if order_arrivals[i] > 0.5:  # Buy order
                    ask_volumes[i] = max(0, ask_volumes[i] - np.random.exponential(50))
                    if ask_volumes[i] < 10:  # Refill
                        ask_volumes[i] = np.random.exponential(1000)
                        ask_prices[i] += self.tick_size
                else:  # Sell order
                    bid_volumes[i] = max(0, bid_volumes[i] - np.random.exponential(50))
                    if bid_volumes[i] < 10:  # Refill
                        bid_volumes[i] = np.random.exponential(1000)
                        bid_prices[i] -= self.tick_size
        
        result = {
            'times': times,
            'mid_prices': mid_prices,
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
            'spreads': ask_prices - bid_prices,
            'imbalances': (bid_volumes - ask_volumes) / (bid_volumes + ask_volumes)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'simulate_limit_order_book',
            'duration': duration,
            'final_price': float(mid_prices[-1]),
            'avg_spread': float(np.mean(result['spreads']))
        })
        
        return result
    
    def _simulate_hawkes_process(self, duration: float, dt: float) -> np.ndarray:
        """Simulate Hawkes process for order arrivals"""
        n_steps = int(duration / dt)
        intensity = np.zeros(n_steps)
        arrivals = np.zeros(n_steps)
        
        intensity[0] = self.hawkes_mu
        
        for i in range(1, n_steps):
            decay = np.exp(-self.hawkes_beta * dt)
            intensity[i] = self.hawkes_mu + (intensity[i-1] - self.hawkes_mu) * decay
            
            arrival_prob = intensity[i] * dt
            if np.random.random() < arrival_prob:
                arrivals[i] = 1
                intensity[i] += self.hawkes_alpha  # Self-excitement
            
        return arrivals
    
    def analyze_order_book_imbalance(self, 
                                    bids: Dict[float, float], 
                                    asks: Dict[float, float],
                                    levels: int = 5) -> Dict[str, Any]:
        """
        Analyze order book imbalance for trading signals
        
        Parameters:
        - bids: Dictionary mapping bid prices to volumes
        - asks: Dictionary mapping ask prices to volumes
        - levels: Number of price levels to consider
        
        Returns:
        - Dictionary with order book analysis
        """
        if not bids or not asks:
            logger.warning("Empty order book provided")
            return {
                'imbalance': 0.0,
                'pressure': 'neutral',
                'confidence': 0.5
            }
        
        bid_prices = sorted(bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(asks.keys())[:levels]
        
        if not bid_prices or not ask_prices:
            return {
                'imbalance': 0.0,
                'pressure': 'neutral',
                'confidence': 0.5
            }
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        bid_volume = sum(bids[price] for price in bid_prices)
        ask_volume = sum(asks[price] for price in ask_prices)
        
        if bid_volume + ask_volume == 0:
            imbalance = 0.0
        else:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        weighted_bid_volume = sum(bids[price] * (1 - (mid_price - price) / mid_price) 
                                for price in bid_prices)
        weighted_ask_volume = sum(asks[price] * (1 - (price - mid_price) / mid_price) 
                                for price in ask_prices)
        
        if weighted_bid_volume + weighted_ask_volume == 0:
            weighted_imbalance = 0.0
        else:
            weighted_imbalance = ((weighted_bid_volume - weighted_ask_volume) / 
                                (weighted_bid_volume + weighted_ask_volume))
        
        combined_imbalance = 0.7 * imbalance + 0.3 * weighted_imbalance
        
        if combined_imbalance > 0.1:
            pressure = 'buy'
            confidence = min(0.5 + abs(combined_imbalance), 0.99)
        elif combined_imbalance < -0.1:
            pressure = 'sell'
            confidence = min(0.5 + abs(combined_imbalance), 0.99)
        else:
            pressure = 'neutral'
            confidence = 0.5
        
        result = {
            'imbalance': float(combined_imbalance),
            'pressure': pressure,
            'confidence': float(confidence),
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume),
            'spread': float(ask_prices[0] - bid_prices[0]),
            'mid_price': float(mid_price)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'analyze_order_book_imbalance',
            'imbalance': float(combined_imbalance),
            'pressure': pressure,
            'confidence': float(confidence)
        })
        
        return result
    
    def calculate_kyle_lambda(self, 
                             prices: np.ndarray, 
                             volumes: np.ndarray,
                             window_size: int = 50) -> Dict[str, Any]:
        """
        Calculate Kyle's lambda (price impact parameter)
        
        Parameters:
        - prices: Array of price observations
        - volumes: Array of volume observations
        - window_size: Window size for rolling calculation
        
        Returns:
        - Dictionary with Kyle's lambda estimates
        """
        if len(prices) != len(volumes):
            logger.error("Price and volume arrays must have the same length")
            return {
                'kyle_lambda': 0.0,
                'confidence': 0.5,
                'r_squared': 0.0
            }
        
        if len(prices) < window_size + 1:
            logger.warning(f"Insufficient data for Kyle's lambda calculation. Need at least {window_size+1} points")
            return {
                'kyle_lambda': 0.0,
                'confidence': 0.5,
                'r_squared': 0.0
            }
        
        price_changes = np.diff(prices)
        
        signed_volumes = np.sign(price_changes) * volumes[:-1]
        
        lambdas = []
        r_squareds = []
        
        for i in range(len(price_changes) - window_size + 1):
            window_price_changes = price_changes[i:i+window_size]
            window_signed_volumes = signed_volumes[i:i+window_size]
            
            if np.sum(window_signed_volumes**2) > 0:
                lambda_est = np.sum(window_price_changes * window_signed_volumes) / np.sum(window_signed_volumes**2)
                
                predictions = lambda_est * window_signed_volumes
                ss_total = np.sum((window_price_changes - np.mean(window_price_changes))**2)
                ss_residual = np.sum((window_price_changes - predictions)**2)
                
                if ss_total > 0:
                    r_squared = 1 - (ss_residual / ss_total)
                else:
                    r_squared = 0.0
            else:
                lambda_est = 0.0
                r_squared = 0.0
            
            lambdas.append(lambda_est)
            r_squareds.append(r_squared)
        
        kyle_lambda = lambdas[-1] if lambdas else 0.0
        r_squared = r_squareds[-1] if r_squareds else 0.0
        
        confidence = min(0.5 + r_squared/2, 0.99)
        
        result = {
            'kyle_lambda': float(kyle_lambda),
            'confidence': float(confidence),
            'r_squared': float(r_squared),
            'lambda_history': lambdas,
            'r_squared_history': r_squareds
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'calculate_kyle_lambda',
            'kyle_lambda': float(kyle_lambda),
            'r_squared': float(r_squared),
            'confidence': float(confidence)
        })
        
        return result
    
    def optimal_execution_strategy(self, 
                                  total_shares: int, 
                                  time_horizon: float,
                                  price: float,
                                  volatility: float,
                                  market_impact: float,
                                  risk_aversion: float = 1.0) -> Dict[str, Any]:
        """
        Calculate optimal execution strategy using Almgren-Chriss model
        
        Parameters:
        - total_shares: Total number of shares to execute
        - time_horizon: Time horizon in hours
        - price: Current price
        - volatility: Price volatility (annualized)
        - market_impact: Market impact parameter (Kyle's lambda)
        - risk_aversion: Risk aversion parameter
        
        Returns:
        - Dictionary with optimal execution strategy
        """
        vol_adjusted = volatility * np.sqrt(time_horizon / 8760)  # 8760 hours in a year
        
        n_periods = max(int(time_horizon * 4), 2)  # 15-minute intervals
        
        tau = time_horizon / n_periods
        
        kappa = market_impact  # Temporary impact
        lambda_param = market_impact * 0.1  # Permanent impact
        
        alpha = 0.5 * risk_aversion * (vol_adjusted**2) / kappa
        sinh_term = np.sinh(alpha * time_horizon)
        
        optimal_shares = []
        remaining_shares = []
        times = []
        
        shares_remaining = total_shares
        
        for i in range(n_periods + 1):
            t = i * tau
            times.append(t)
            
            if i < n_periods:
                x_t = total_shares * np.sinh(alpha * (time_horizon - t)) / sinh_term
                
                if i > 0:
                    shares_to_trade = shares_remaining - x_t
                else:
                    shares_to_trade = total_shares - x_t
                
                shares_remaining = x_t
            else:
                shares_to_trade = shares_remaining
                shares_remaining = 0
            
            optimal_shares.append(shares_to_trade)
            remaining_shares.append(shares_remaining)
        
        temp_impact_cost = kappa * sum(s**2 for s in optimal_shares[:-1])
        
        perm_impact_cost = lambda_param * total_shares * price * 0.5
        
        risk_cost = 0.5 * risk_aversion * (vol_adjusted**2) * sum(
            remaining_shares[i]**2 * tau for i in range(n_periods)
        )
        
        total_cost = temp_impact_cost + perm_impact_cost + risk_cost
        total_cost_pct = total_cost / (total_shares * price) * 100
        
        result = {
            'times': times,
            'optimal_shares': optimal_shares,
            'remaining_shares': remaining_shares,
            'total_cost': float(total_cost),
            'total_cost_pct': float(total_cost_pct),
            'temp_impact_cost': float(temp_impact_cost),
            'perm_impact_cost': float(perm_impact_cost),
            'risk_cost': float(risk_cost)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'optimal_execution_strategy',
            'total_shares': total_shares,
            'time_horizon': time_horizon,
            'total_cost_pct': float(total_cost_pct)
        })
        
        return result
    
    def detect_spoofing_patterns(self, 
                                order_history: List[Dict[str, Any]],
                                price_history: np.ndarray) -> Dict[str, Any]:
        """
        Detect potential spoofing patterns in order book
        
        Parameters:
        - order_history: List of order events with timestamps, prices, volumes, and actions
        - price_history: Array of price observations
        
        Returns:
        - Dictionary with spoofing detection results
        """
        if not order_history or len(order_history) < 10:
            logger.warning("Insufficient order history for spoofing detection")
            return {
                'spoofing_detected': False,
                'confidence': 0.0,
                'pattern': None
            }
        
        large_orders = [order for order in order_history 
                       if order.get('volume', 0) > np.percentile([o.get('volume', 0) for o in order_history], 90)]
        
        if not large_orders:
            return {
                'spoofing_detected': False,
                'confidence': 0.0,
                'pattern': None
            }
        
        cancellations = []
        
        for i, order in enumerate(large_orders):
            if order.get('action') == 'add':
                order_id = order.get('order_id')
                if not order_id:
                    continue
                
                for j in range(i+1, len(large_orders)):
                    if (large_orders[j].get('action') == 'cancel' and 
                        large_orders[j].get('order_id') == order_id):
                        
                        time_diff = (large_orders[j].get('timestamp', 0) - 
                                    order.get('timestamp', 0))
                        
                        if time_diff < 1.0:  # Less than 1 second
                            cancellations.append({
                                'order': order,
                                'cancellation': large_orders[j],
                                'time_diff': time_diff
                            })
                        
                        break
        
        price_movements = []
        
        for cancel in cancellations:
            cancel_time = cancel['cancellation'].get('timestamp', 0)
            cancel_price = cancel['order'].get('price', 0)
            
            for i in range(len(price_history) - 1):
                if i > 0 and price_history[i-1] == cancel_time:
                    price_change = (price_history[i+1] - price_history[i]) / price_history[i]
                    
                    if (cancel['order'].get('side') == 'buy' and price_change > 0) or \
                       (cancel['order'].get('side') == 'sell' and price_change < 0):
                        price_movements.append({
                            'cancellation': cancel,
                            'price_change': price_change
                        })
                    
                    break
        
        if not cancellations:
            spoofing_score = 0.0
        else:
            cancel_ratio = len(cancellations) / len(large_orders)
            
            if not price_movements:
                movement_ratio = 0.0
            else:
                movement_ratio = len(price_movements) / len(cancellations)
            
            spoofing_score = 0.7 * cancel_ratio + 0.3 * movement_ratio
        
        spoofing_detected = spoofing_score > 0.6
        
        if spoofing_detected:
            if all(c['order'].get('side') == 'buy' for c in cancellations):
                pattern = 'buy_side_spoofing'
            elif all(c['order'].get('side') == 'sell' for c in cancellations):
                pattern = 'sell_side_spoofing'
            else:
                pattern = 'mixed_spoofing'
        else:
            pattern = None
        
        result = {
            'spoofing_detected': spoofing_detected,
            'confidence': float(spoofing_score),
            'pattern': pattern,
            'rapid_cancellations': len(cancellations),
            'price_movements': len(price_movements),
            'large_orders': len(large_orders)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_spoofing_patterns',
            'spoofing_detected': spoofing_detected,
            'confidence': float(spoofing_score),
            'pattern': pattern
        })
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about microstructure modeling usage
        
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
            'tick_size': self.tick_size
        }
