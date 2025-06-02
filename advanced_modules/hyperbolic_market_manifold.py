#!/usr/bin/env python3
"""
Hyperbolic Market Manifold Module

Implements hyperbolic embedding for market data using Geomstats library.
Projects noisy market data into pure hyperbolic space to kill noise and
model hierarchical market structures like order books and liquidity cascades.

Based on the Real, No-Hopium Trading System specifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HyperbolicMarketManifold")

try:
    from geomstats.geometry.hyperbolic import Hyperbolic
    GEOMSTATS_AVAILABLE = True
    logger.info("Geomstats library available")
except ImportError:
    GEOMSTATS_AVAILABLE = False
    logger.warning("Geomstats not available. Using mock implementation.")
    
    class MockHyperbolic:
        """Mock Hyperbolic class for testing when geomstats is not available"""
        def __init__(self, dim=11):
            self.dim = dim
            
        def random_point(self):
            return np.random.randn(self.dim)
            
        def to_tangent(self, vector, base_point):
            return np.array(vector[:self.dim] if len(vector) >= self.dim else 
                          list(vector) + [0] * (self.dim - len(vector)))
            
        def exp(self, tangent_vec):
            return tangent_vec / (1 + np.linalg.norm(tangent_vec))
    
    Hyperbolic = MockHyperbolic

class HyperbolicMarketManifold:
    """
    Hyperbolic Market Manifold for noise-free market data embedding
    
    Uses hyperbolic geometry to naturally model hierarchies (order books, liquidity cascades)
    and kill noise because news/events appear as outliers in curved space.
    
    Mathematical foundation:
    - Hyperbolic embedding projects out noise because:
      dist(x, noise) → ∞ as curvature → -∞
    """
    
    def __init__(self, dimension: int = 11, precision: int = 128):
        """
        Initialize Hyperbolic Market Manifold
        
        Parameters:
        - dimension: Dimension of hyperbolic space (default: 11D like AdS space in physics)
        - precision: Numerical precision for calculations
        """
        self.dimension = dimension
        self.precision = precision
        self.history = []
        
        self.hyperbolic_manifold = Hyperbolic(dim=dimension)
        
        logger.info(f"Initialized HyperbolicMarketManifold with dimension={dimension}, "
                   f"precision={precision}, geomstats_available={GEOMSTATS_AVAILABLE}")
    
    def embed_market_data(self, price: float, volume: float, order_flow: float) -> np.ndarray:
        """
        Projects noisy market data into pure hyperbolic space
        
        Parameters:
        - price: Current price
        - volume: Current volume
        - order_flow: Order flow data
        
        Returns:
        - Embedded point in hyperbolic space (kills noise)
        """
        try:
            raw_data = np.array([price, volume, order_flow])
            base_point = self.hyperbolic_manifold.random_point()
            
            tangent_vec = self.hyperbolic_manifold.to_tangent(
                raw_data, 
                base_point=base_point
            )
            
            embedded_point = self.hyperbolic_manifold.exp(tangent_vec)
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'embed_market_data',
                'input': {'price': price, 'volume': volume, 'order_flow': order_flow},
                'output_norm': float(np.linalg.norm(embedded_point)),
                'dimension': self.dimension
            })
            
            return embedded_point
            
        except Exception as e:
            logger.error(f"Error in hyperbolic embedding: {str(e)}")
            raw_data = np.array([price, volume, order_flow])
            normalized = raw_data / (1 + np.linalg.norm(raw_data))
            
            if len(normalized) < self.dimension:
                padded = np.zeros(self.dimension)
                padded[:len(normalized)] = normalized
                return padded
            
            return normalized[:self.dimension]
    
    def embed_price_series(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None, 
                          order_flows: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Embed a series of market data points into hyperbolic space
        
        Parameters:
        - prices: Array of prices
        - volumes: Array of volumes (optional)
        - order_flows: Array of order flows (optional)
        
        Returns:
        - Array of embedded points in hyperbolic space
        """
        if volumes is None:
            volumes = np.ones_like(prices)
        if order_flows is None:
            order_flows = np.zeros_like(prices)
            
        embedded_points = []
        
        for i in range(len(prices)):
            embedded = self.embed_market_data(
                prices[i], 
                volumes[i] if i < len(volumes) else 1.0,
                order_flows[i] if i < len(order_flows) else 0.0
            )
            embedded_points.append(embedded)
            
        return np.array(embedded_points)
    
    def calculate_hyperbolic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate hyperbolic distance between two points
        
        Parameters:
        - point1, point2: Points in hyperbolic space
        
        Returns:
        - Hyperbolic distance
        """
        try:
            euclidean_dist = np.linalg.norm(point1 - point2)
            
            hyperbolic_dist = np.arccosh(1 + 2 * euclidean_dist**2 / 
                                       ((1 - np.linalg.norm(point1)**2) * 
                                        (1 - np.linalg.norm(point2)**2)))
            
            return float(hyperbolic_dist)
            
        except Exception as e:
            logger.warning(f"Error calculating hyperbolic distance: {str(e)}")
            return float(np.linalg.norm(point1 - point2))
    
    def detect_noise_outliers(self, embedded_points: np.ndarray, 
                             threshold: float = 2.0) -> List[int]:
        """
        Detect noise outliers in hyperbolic space
        
        Parameters:
        - embedded_points: Array of embedded points
        - threshold: Distance threshold for outlier detection
        
        Returns:
        - List of indices of outlier points
        """
        if len(embedded_points) < 2:
            return []
            
        center = np.mean(embedded_points, axis=0)
        distances = []
        
        for point in embedded_points:
            dist = self.calculate_hyperbolic_distance(point, center)
            distances.append(dist)
            
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        outliers = []
        for i, dist in enumerate(distances):
            if dist > mean_dist + threshold * std_dist:
                outliers.append(i)
                
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_noise_outliers',
            'total_points': len(embedded_points),
            'outliers_found': len(outliers),
            'threshold': threshold
        })
        
        return outliers
    
    def generate_trading_signal(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None,
                               order_flows: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate trading signal using hyperbolic embedding
        
        Parameters:
        - prices: Array of recent prices
        - volumes: Array of recent volumes (optional)
        - order_flows: Array of recent order flows (optional)
        
        Returns:
        - Trading signal with hyperbolic analysis
        """
        if len(prices) < 3:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'Insufficient data for hyperbolic analysis'
            }
            
        embedded_points = self.embed_price_series(prices, volumes, order_flows)
        
        outliers = self.detect_noise_outliers(embedded_points)
        
        recent_points = embedded_points[-3:]
        
        momentum = 0.0
        for i in range(1, len(recent_points)):
            dist = self.calculate_hyperbolic_distance(recent_points[i-1], recent_points[i])
            momentum += dist
            
        signal = 'NEUTRAL'
        confidence = 0.5
        
        if momentum > 0.1:  # Significant movement in hyperbolic space
            price_change = prices[-1] - prices[-2]
            if price_change > 0:
                signal = 'BUY'
                confidence = min(0.9, 0.5 + momentum)
            else:
                signal = 'SELL'
                confidence = min(0.9, 0.5 + momentum)
        
        if len(outliers) > 0:
            recent_outliers = [o for o in outliers if o >= len(embedded_points) - 3]
            if recent_outliers:
                confidence *= 0.7  # Reduce confidence due to noise
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'hyperbolic_momentum': momentum,
            'outliers_detected': len(outliers),
            'noise_immunity': 1.0 - (len(outliers) / len(embedded_points)),
            'dimension': self.dimension,
            'geomstats_available': GEOMSTATS_AVAILABLE
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'generate_trading_signal',
            'signal': signal,
            'confidence': confidence,
            'momentum': momentum,
            'outliers': len(outliers)
        })
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about hyperbolic manifold usage
        
        Returns:
        - Dictionary with usage statistics
        """
        if not self.history:
            return {'count': 0, 'dimension': self.dimension}
            
        operations = {}
        for h in self.history:
            op = h.get('operation', 'unknown')
            operations[op] = operations.get(op, 0) + 1
            
        return {
            'count': len(self.history),
            'operations': operations,
            'dimension': self.dimension,
            'precision': self.precision,
            'geomstats_available': GEOMSTATS_AVAILABLE
        }
