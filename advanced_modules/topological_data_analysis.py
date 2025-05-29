#!/usr/bin/env python3
"""
Topological Data Analysis (TDA) Module

Implements topological data analysis techniques for market regime detection:
- Persistent homology for detecting market phase shifts
- Simplicial complex construction from financial data
- Mapper algorithm for visualizing high-dimensional market structures
- Betti numbers and persistence diagrams for market topology
- Topological features for trading signal generation

This module provides rigorous mathematical tools to detect hidden structures
in market data using algebraic topology.
"""

import numpy as np
import pandas as pd
from scipy import stats, spatial
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TopologicalDataAnalysis")

class TopologicalDataAnalysis:
    """
    Topological Data Analysis for market regime detection
    
    Implements algebraic topology techniques for financial markets:
    - Persistent homology
    - Simplicial complexes
    - Mapper algorithm
    - Betti numbers
    - Persistence diagrams
    
    Provides rigorous mathematical tools to detect hidden structures
    in market data using algebraic topology.
    """
    
    def __init__(self, precision: int = 64, confidence_level: float = 0.99,
                max_dimension: int = 2, filtration_steps: int = 50):
        """
        Initialize Topological Data Analysis
        
        Parameters:
        - precision: Numerical precision for calculations (default: 64 bits)
        - confidence_level: Statistical confidence level (default: 0.99)
        - max_dimension: Maximum homology dimension to compute (default: 2)
        - filtration_steps: Number of steps in filtration (default: 50)
        """
        self.precision = precision
        self.confidence_level = confidence_level
        self.max_dimension = max_dimension
        self.filtration_steps = filtration_steps
        self.history = []
        
        np.random.seed(42)  # For reproducibility
        
        logger.info(f"Initialized TopologicalDataAnalysis with precision={precision}, "
                   f"confidence_level={confidence_level}, "
                   f"max_dimension={max_dimension}")
    
    
    def construct_vietoris_rips_complex(self, data: np.ndarray, 
                                       epsilon: float) -> Dict[str, List]:
        """
        Construct Vietoris-Rips complex from data points
        
        Parameters:
        - data: Data points as array of shape (n_samples, n_features)
        - epsilon: Distance threshold for edge creation
        
        Returns:
        - Dictionary with simplices of different dimensions
        """
        distances = spatial.distance.pdist(data)
        distance_matrix = spatial.distance.squareform(distances)
        
        n_points = data.shape[0]
        complex_dict = {
            '0-simplices': [[i] for i in range(n_points)],  # Vertices
            '1-simplices': [],  # Edges
            '2-simplices': [],  # Triangles
            '3-simplices': []   # Tetrahedra
        }
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                if distance_matrix[i, j] <= epsilon:
                    complex_dict['1-simplices'].append([i, j])
        
        for i, j, k in self._generate_triplets(n_points):
            if (distance_matrix[i, j] <= epsilon and 
                distance_matrix[j, k] <= epsilon and 
                distance_matrix[i, k] <= epsilon):
                complex_dict['2-simplices'].append([i, j, k])
        
        if self.max_dimension >= 3:
            for i, j, k, l in self._generate_quadruplets(n_points):
                if (distance_matrix[i, j] <= epsilon and 
                    distance_matrix[i, k] <= epsilon and 
                    distance_matrix[i, l] <= epsilon and 
                    distance_matrix[j, k] <= epsilon and 
                    distance_matrix[j, l] <= epsilon and 
                    distance_matrix[k, l] <= epsilon):
                    complex_dict['3-simplices'].append([i, j, k, l])
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'construct_vietoris_rips_complex',
            'data_shape': data.shape,
            'epsilon': epsilon,
            'n_0_simplices': len(complex_dict['0-simplices']),
            'n_1_simplices': len(complex_dict['1-simplices']),
            'n_2_simplices': len(complex_dict['2-simplices']),
            'n_3_simplices': len(complex_dict['3-simplices'])
        })
        
        return complex_dict
    
    def _generate_triplets(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate all triplets of indices from 0 to n-1"""
        return [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def _generate_quadruplets(self, n: int) -> List[Tuple[int, int, int, int]]:
        """Generate all quadruplets of indices from 0 to n-1"""
        return [(i, j, k, l) for i in range(n) for j in range(i+1, n) 
                for k in range(j+1, n) for l in range(k+1, n)]
    
    def construct_alpha_complex(self, data: np.ndarray, 
                               alpha: float) -> Dict[str, List]:
        """
        Construct Alpha complex from data points
        
        Parameters:
        - data: Data points as array of shape (n_samples, n_features)
        - alpha: Alpha parameter for filtration
        
        Returns:
        - Dictionary with simplices of different dimensions
        """
        
        distances = spatial.distance.pdist(data)
        distance_matrix = spatial.distance.squareform(distances)
        
        n_points = data.shape[0]
        complex_dict = {
            '0-simplices': [[i] for i in range(n_points)],  # Vertices
            '1-simplices': [],  # Edges
            '2-simplices': [],  # Triangles
            '3-simplices': []   # Tetrahedra
        }
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                if distance_matrix[i, j] <= 2 * alpha:  # Alpha complex criterion
                    complex_dict['1-simplices'].append([i, j])
        
        for i, j, k in self._generate_triplets(n_points):
            if (distance_matrix[i, j] <= 2 * alpha and 
                distance_matrix[j, k] <= 2 * alpha and 
                distance_matrix[i, k] <= 2 * alpha):
                a, b, c = distance_matrix[i, j], distance_matrix[j, k], distance_matrix[i, k]
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area > 0:
                    circumradius = (a * b * c) / (4 * area)
                    if circumradius <= alpha:
                        complex_dict['2-simplices'].append([i, j, k])
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'construct_alpha_complex',
            'data_shape': data.shape,
            'alpha': alpha,
            'n_0_simplices': len(complex_dict['0-simplices']),
            'n_1_simplices': len(complex_dict['1-simplices']),
            'n_2_simplices': len(complex_dict['2-simplices'])
        })
        
        return complex_dict
    
    
    def compute_betti_numbers(self, complex_dict: Dict[str, List]) -> Dict[str, int]:
        """
        Compute Betti numbers from simplicial complex
        
        Parameters:
        - complex_dict: Dictionary with simplices of different dimensions
        
        Returns:
        - Dictionary with Betti numbers
        """
        
        n_0_simplices = len(complex_dict['0-simplices'])
        n_1_simplices = len(complex_dict['1-simplices'])
        n_2_simplices = len(complex_dict['2-simplices'])
        n_3_simplices = len(complex_dict['3-simplices'])
        
        edges = complex_dict['1-simplices']
        vertices = list(range(n_0_simplices))
        
        parent = list(range(n_0_simplices))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        for edge in edges:
            union(edge[0], edge[1])
        
        components = set(find(v) for v in vertices)
        betti_0 = len(components)
        
        betti_1 = max(0, n_1_simplices - n_0_simplices + betti_0 - n_2_simplices)
        
        betti_2 = max(0, n_2_simplices - n_1_simplices + betti_1 - n_3_simplices)
        
        result = {
            'betti_0': betti_0,  # Connected components
            'betti_1': betti_1,  # Cycles/holes
            'betti_2': betti_2   # Voids/cavities
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'compute_betti_numbers',
            'n_0_simplices': n_0_simplices,
            'n_1_simplices': n_1_simplices,
            'n_2_simplices': n_2_simplices,
            'result': result
        })
        
        return result
    
    def compute_persistence_diagram(self, data: np.ndarray, 
                                   max_epsilon: float) -> Dict[str, List]:
        """
        Compute persistence diagram from data
        
        Parameters:
        - data: Data points as array of shape (n_samples, n_features)
        - max_epsilon: Maximum distance threshold for filtration
        
        Returns:
        - Dictionary with persistence pairs for different dimensions
        """
        
        persistence_diagram = {
            'dim_0': [],  # (birth, death) pairs for H0
            'dim_1': [],  # (birth, death) pairs for H1
            'dim_2': []   # (birth, death) pairs for H2
        }
        
        epsilon_values = np.linspace(0, max_epsilon, self.filtration_steps)
        
        prev_betti = {'betti_0': 0, 'betti_1': 0, 'betti_2': 0}
        betti_births = {'betti_0': [], 'betti_1': [], 'betti_2': []}
        
        for epsilon in epsilon_values:
            complex_dict = self.construct_vietoris_rips_complex(data, epsilon)
            
            betti = self.compute_betti_numbers(complex_dict)
            
            for dim in range(3):
                betti_key = f'betti_{dim}'
                
                new_births = max(0, betti[betti_key] - prev_betti[betti_key])
                for _ in range(new_births):
                    betti_births[betti_key].append(epsilon)
                
                new_deaths = max(0, prev_betti[betti_key] - betti[betti_key])
                for _ in range(new_deaths):
                    if betti_births[betti_key]:
                        birth = betti_births[betti_key].pop(0)
                        persistence_diagram[f'dim_{dim}'].append((birth, epsilon))
            
            prev_betti = betti
        
        for dim in range(3):
            betti_key = f'betti_{dim}'
            for birth in betti_births[betti_key]:
                persistence_diagram[f'dim_{dim}'].append((birth, max_epsilon))
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'compute_persistence_diagram',
            'data_shape': data.shape,
            'max_epsilon': max_epsilon,
            'filtration_steps': self.filtration_steps,
            'n_dim_0_pairs': len(persistence_diagram['dim_0']),
            'n_dim_1_pairs': len(persistence_diagram['dim_1']),
            'n_dim_2_pairs': len(persistence_diagram['dim_2'])
        })
        
        return persistence_diagram
    
    def persistence_statistics(self, persistence_diagram: Dict[str, List]) -> Dict:
        """
        Compute statistics from persistence diagram
        
        Parameters:
        - persistence_diagram: Dictionary with persistence pairs
        
        Returns:
        - Dictionary with persistence statistics
        """
        stats = {}
        
        for dim in range(3):
            dim_key = f'dim_{dim}'
            pairs = persistence_diagram[dim_key]
            
            if not pairs:
                stats[f'{dim_key}_count'] = 0
                stats[f'{dim_key}_avg_persistence'] = 0
                stats[f'{dim_key}_max_persistence'] = 0
                continue
                
            persistence_values = [death - birth for birth, death in pairs]
            
            stats[f'{dim_key}_count'] = len(pairs)
            stats[f'{dim_key}_avg_persistence'] = float(np.mean(persistence_values))
            stats[f'{dim_key}_max_persistence'] = float(np.max(persistence_values))
            stats[f'{dim_key}_total_persistence'] = float(np.sum(persistence_values))
            
            if persistence_values:
                total = np.sum(persistence_values)
                if total > 0:
                    probs = np.array(persistence_values) / total
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    stats[f'{dim_key}_persistence_entropy'] = float(entropy)
                else:
                    stats[f'{dim_key}_persistence_entropy'] = 0
            else:
                stats[f'{dim_key}_persistence_entropy'] = 0
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'persistence_statistics',
            'result': stats
        })
        
        return stats
    
    
    def mapper_algorithm(self, data: np.ndarray, filter_function: str = 'pca',
                        n_intervals: int = 10, overlap: float = 0.5,
                        clustering_method: str = 'dbscan') -> Dict:
        """
        Apply Mapper algorithm to visualize high-dimensional data
        
        Parameters:
        - data: Data points as array of shape (n_samples, n_features)
        - filter_function: Function to reduce dimensionality ('pca', 'distance', 'eccentricity')
        - n_intervals: Number of intervals for filter function range
        - overlap: Overlap percentage between intervals
        - clustering_method: Clustering method for points in intervals ('dbscan', 'kmeans')
        
        Returns:
        - Dictionary with Mapper graph structure
        """
        filter_values = self._apply_filter_function(data, filter_function)
        
        intervals = self._create_overlapping_intervals(filter_values, n_intervals, overlap)
        
        nodes = []
        node_points = []
        
        for i, (interval_min, interval_max) in enumerate(intervals):
            interval_mask = (filter_values >= interval_min) & (filter_values <= interval_max)
            interval_points_idx = np.where(interval_mask)[0]
            
            if len(interval_points_idx) < 2:
                continue
                
            interval_points = data[interval_points_idx]
            
            clusters = self._cluster_points(interval_points, clustering_method)
            
            for j, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    cluster_points = interval_points_idx[cluster]
                    node_id = f"interval_{i}_cluster_{j}"
                    nodes.append(node_id)
                    node_points.append(cluster_points.tolist())
        
        edges = []
        
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                shared_points = set(node_points[i]) & set(node_points[j])
                if shared_points:
                    edges.append((nodes[i], nodes[j], len(shared_points)))
        
        mapper_graph = {
            'nodes': nodes,
            'node_points': node_points,
            'edges': edges,
            'n_nodes': len(nodes),
            'n_edges': len(edges)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'mapper_algorithm',
            'data_shape': data.shape,
            'filter_function': filter_function,
            'n_intervals': n_intervals,
            'overlap': overlap,
            'clustering_method': clustering_method,
            'n_nodes': len(nodes),
            'n_edges': len(edges)
        })
        
        return mapper_graph
    
    def _apply_filter_function(self, data: np.ndarray, filter_function: str) -> np.ndarray:
        """Apply filter function to reduce dimensionality"""
        if filter_function == 'pca':
            pca = PCA(n_components=1)
            return pca.fit_transform(data).flatten()
        elif filter_function == 'distance':
            mean = np.mean(data, axis=0)
            return np.linalg.norm(data - mean, axis=1)
        elif filter_function == 'eccentricity':
            distances = spatial.distance.pdist(data)
            distance_matrix = spatial.distance.squareform(distances)
            return np.mean(distance_matrix, axis=1)
        else:
            logger.warning(f"Unknown filter function: {filter_function}, using PCA")
            pca = PCA(n_components=1)
            return pca.fit_transform(data).flatten()
    
    def _create_overlapping_intervals(self, values: np.ndarray, 
                                     n_intervals: int, overlap: float) -> List[Tuple[float, float]]:
        """Create overlapping intervals for filter function range"""
        min_val, max_val = np.min(values), np.max(values)
        interval_size = (max_val - min_val) / (n_intervals - (n_intervals - 1) * overlap)
        
        intervals = []
        for i in range(n_intervals):
            interval_min = min_val + i * interval_size * (1 - overlap)
            interval_max = interval_min + interval_size
            intervals.append((interval_min, interval_max))
            
        return intervals
    
    def _cluster_points(self, points: np.ndarray, clustering_method: str) -> List[np.ndarray]:
        """Cluster points in an interval"""
        if len(points) < 2:
            return [np.array([0])]
            
        if clustering_method == 'dbscan':
            n_neighbors = min(len(points) - 1, 5)
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(points)
            distances, _ = nn.kneighbors(points)
            epsilon = np.mean(distances[:, -1]) * 1.5
            
            dbscan = DBSCAN(eps=epsilon, min_samples=2)
            labels = dbscan.fit_predict(points)
            
            max_label = np.max(labels)
            for i, label in enumerate(labels):
                if label == -1:
                    max_label += 1
                    labels[i] = max_label
            
            unique_labels = np.unique(labels)
            return [np.where(labels == label)[0] for label in unique_labels]
        elif clustering_method == 'kmeans':
            n_clusters = min(len(points) // 5 + 1, len(points))
            n_clusters = max(1, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(points)
            
            unique_labels = np.unique(labels)
            return [np.where(labels == label)[0] for label in unique_labels]
        else:
            logger.warning(f"Unknown clustering method: {clustering_method}, using DBSCAN")
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(points)
            
            max_label = np.max(labels)
            for i, label in enumerate(labels):
                if label == -1:
                    max_label += 1
                    labels[i] = max_label
            
            unique_labels = np.unique(labels)
            return [np.where(labels == label)[0] for label in unique_labels]
    
    
    def detect_market_regimes(self, returns: np.ndarray, 
                             window_size: int = 50) -> Dict:
        """
        Detect market regimes using topological features
        
        Parameters:
        - returns: Array of asset returns with shape (time_steps, assets)
        - window_size: Window size for rolling analysis
        
        Returns:
        - Dictionary with market regime detection results
        """
        returns = np.asarray(returns)
        
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        time_steps, assets = returns.shape
        
        regime_changes = []
        betti_0_series = []
        betti_1_series = []
        
        for t in range(window_size, time_steps):
            window_data = returns[t-window_size:t]
            
            scaler = StandardScaler()
            window_data_scaled = scaler.fit_transform(window_data)
            
            persistence_diagram = self.compute_persistence_diagram(
                window_data_scaled, max_epsilon=2.0)
            
            stats = self.persistence_statistics(persistence_diagram)
            
            betti_0_series.append(stats['dim_0_count'])
            betti_1_series.append(stats['dim_1_count'])
            
            if len(betti_1_series) >= 2:
                if (abs(betti_1_series[-1] - betti_1_series[-2]) > 2 or
                    abs(betti_0_series[-1] - betti_0_series[-2]) > 2):
                    regime_changes.append(t)
        
        regime_labels = np.zeros(time_steps, dtype=int)
        current_regime = 0
        
        for t in range(window_size):
            regime_labels[t] = current_regime
            
        for t in range(window_size, time_steps):
            if t in regime_changes:
                current_regime += 1
            regime_labels[t] = current_regime
        
        unique_regimes = np.unique(regime_labels)
        regime_stats = []
        
        for regime in unique_regimes:
            regime_mask = (regime_labels == regime)
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats.append({
                    'regime_id': int(regime),
                    'start_idx': int(np.where(regime_mask)[0][0]),
                    'end_idx': int(np.where(regime_mask)[0][-1]),
                    'duration': int(np.sum(regime_mask)),
                    'mean_return': float(np.mean(regime_returns)),
                    'volatility': float(np.std(np.mean(regime_returns, axis=1))),
                    'sharpe': float(np.mean(np.mean(regime_returns, axis=1)) / 
                                  np.std(np.mean(regime_returns, axis=1)) 
                                  if np.std(np.mean(regime_returns, axis=1)) > 0 else 0)
                })
        
        result = {
            'regime_changes': regime_changes,
            'regime_labels': regime_labels.tolist(),
            'regime_stats': regime_stats,
            'n_regimes': len(unique_regimes),
            'betti_0_series': betti_0_series,
            'betti_1_series': betti_1_series
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_market_regimes',
            'returns_shape': returns.shape,
            'window_size': window_size,
            'n_regime_changes': len(regime_changes),
            'n_regimes': len(unique_regimes)
        })
        
        return result
    
    def topological_trading_signal(self, returns: np.ndarray, 
                                  lookback: int = 50) -> Dict:
        """
        Generate trading signals based on topological features
        
        Parameters:
        - returns: Array of asset returns with shape (time_steps, assets)
        - lookback: Lookback period for analysis
        
        Returns:
        - Dictionary with trading signals
        """
        returns = np.asarray(returns)
        
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        time_steps, assets = returns.shape
        
        if time_steps < lookback:
            logger.warning(f"Not enough data points: {time_steps} < {lookback}")
            return {'signals': [], 'confidence': 0.0}
        
        recent_returns = returns[-lookback:]
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(recent_returns)
        
        persistence_diagram = self.compute_persistence_diagram(
            data_scaled, max_epsilon=2.0)
        
        stats = self.persistence_statistics(persistence_diagram)
        
        signals = np.zeros(assets)
        confidence = 0.0
        
        if stats['dim_1_count'] > 0:
            reversal_signal = -np.sign(np.mean(recent_returns[-5:], axis=0))
            reversal_strength = min(1.0, stats['dim_1_avg_persistence'] * 2)
            
            if stats['dim_0_count'] > 1:
                kmeans = KMeans(n_clusters=min(stats['dim_0_count'], assets), random_state=42)
                clusters = kmeans.fit_predict(recent_returns.T)
                
                cluster_returns = {}
                for i in range(assets):
                    cluster = clusters[i]
                    if cluster not in cluster_returns:
                        cluster_returns[cluster] = []
                    cluster_returns[cluster].append(np.mean(recent_returns[-5:, i]))
                
                for i in range(assets):
                    cluster = clusters[i]
                    cluster_mean = np.mean(cluster_returns[cluster])
                    if abs(cluster_mean) > 0.001:  # Significant cluster trend
                        signals[i] = np.sign(cluster_mean)
                    else:
                        signals[i] = reversal_signal[i] * reversal_strength
            else:
                signals = reversal_signal * reversal_strength
            
            confidence = min(0.95, 0.5 + stats['dim_1_avg_persistence'] + 
                           stats['dim_0_avg_persistence'])
        else:
            signals = np.sign(np.mean(recent_returns[-5:], axis=0))
            confidence = 0.5
        
        result = {
            'signals': signals.tolist(),
            'confidence': float(confidence),
            'betti_0': stats['dim_0_count'],
            'betti_1': stats['dim_1_count'],
            'persistence_0': stats.get('dim_0_avg_persistence', 0),
            'persistence_1': stats.get('dim_1_avg_persistence', 0)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'topological_trading_signal',
            'returns_shape': returns.shape,
            'lookback': lookback,
            'confidence': float(confidence)
        })
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about topological data analysis usage
        
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
            'max_dimension': self.max_dimension
        }
