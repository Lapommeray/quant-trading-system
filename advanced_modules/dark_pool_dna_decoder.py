#!/usr/bin/env python3
"""
Dark Pool DNA Decoder Module

Implements algorithmic trading pattern detection.
Based on the extracted dark pool DNA decoder code with proper integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.cluster import DBSCAN

class DarkPoolDNADecoder:
    """Algorithmic trading pattern detection using clustering"""
    
    def __init__(self, algorithm):
        """Initialize with algorithm reference following existing pattern"""
        self.algo = algorithm
        self.eps = 0.5
        self.min_samples = 3
        self.clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
    def analyze(self, symbol, history_data):
        """Main analysis method that returns trading signal"""
        try:
            if '1m' not in history_data or len(history_data['1m']) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            df = history_data['1m']
            trades = self._extract_trade_features(df)
            
            algo_clusters = self._decode_algo_dna(trades)
            pattern_analysis = self._analyze_algo_patterns(algo_clusters, trades)
            
            if pattern_analysis['pattern_strength'] > 0.7:
                signal = 'BUY' if pattern_analysis['dominant_direction'] > 0 else 'SELL'
                confidence = min(0.9, pattern_analysis['pattern_strength'])
            else:
                signal = 'NEUTRAL'
                confidence = 0.4
            
            return {
                'signal': signal,
                'confidence': confidence,
                'algo_clusters': algo_clusters.tolist(),
                'pattern_analysis': pattern_analysis,
                'num_algorithms': len(set(algo_clusters[algo_clusters >= 0]))
            }
        except Exception as e:
            self.algo.Debug(f"DarkPoolDNADecoder error: {str(e)}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'error': str(e)}
    
    def _extract_trade_features(self, df):
        """Extract trade features for clustering"""
        try:
            prices = df['Close'].values
            volumes = df['Volume'].values
            
            trades = []
            for i in range(1, len(prices)):
                time_gap = 1.0
                price_impact = abs(prices[i] - prices[i-1]) / prices[i-1]
                size = volumes[i] / np.mean(volumes)
                
                trades.append({
                    'size': size,
                    'time_gap': time_gap,
                    'price_impact': price_impact
                })
            
            return trades
        except Exception:
            return [{'size': 1.0, 'time_gap': 1.0, 'price_impact': 0.001} for _ in range(10)]
    
    def _decode_algo_dna(self, trades):
        """Decode algorithmic DNA using clustering"""
        try:
            if len(trades) < self.min_samples:
                return np.array([-1] * len(trades))
            
            X = np.array([[t['size'], t['time_gap'], t['price_impact']] for t in trades])
            
            labels = self.clustering.fit_predict(X)
            
            return labels
        except Exception:
            return np.array([-1] * len(trades))
    
    def _analyze_algo_patterns(self, clusters, trades):
        """Analyze patterns in algorithmic clusters"""
        try:
            unique_clusters = set(clusters[clusters >= 0])
            
            if not unique_clusters:
                return {
                    'pattern_strength': 0.0,
                    'dominant_direction': 0,
                    'cluster_consistency': 0.0
                }
            
            cluster_directions = []
            cluster_sizes = []
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_trades = [trades[i] for i in range(len(trades)) if cluster_mask[i]]
                
                if cluster_trades:
                    avg_impact = np.mean([t['price_impact'] for t in cluster_trades])
                    direction = 1 if avg_impact > np.median([t['price_impact'] for t in trades]) else -1
                    
                    cluster_directions.append(direction)
                    cluster_sizes.append(len(cluster_trades))
            
            if cluster_directions:
                weighted_direction = np.average(cluster_directions, weights=cluster_sizes)
                pattern_strength = 1.0 - np.std(cluster_directions) / 2.0 if len(cluster_directions) > 1 else 0.8
                cluster_consistency = len(unique_clusters) / len(trades) if len(trades) > 0 else 0.0
            else:
                weighted_direction = 0
                pattern_strength = 0.0
                cluster_consistency = 0.0
            
            return {
                'pattern_strength': max(0.0, min(1.0, float(pattern_strength))),
                'dominant_direction': int(np.sign(weighted_direction)),
                'cluster_consistency': cluster_consistency,
                'num_clusters': len(unique_clusters)
            }
        except Exception:
            return {
                'pattern_strength': 0.0,
                'dominant_direction': 0,
                'cluster_consistency': 0.0,
                'num_clusters': 0
            }
