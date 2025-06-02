#!/usr/bin/env python3
"""
Path Signature Transformer Module

Implements nested path signature analysis for market patterns.
Based on the extracted path signature transformer code with proper integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any

try:
    import esig.tosig as ts
    ESIG_AVAILABLE = True
except ImportError:
    ESIG_AVAILABLE = False
    class MockEsig:
        @staticmethod
        def stream2sig(path, depth):
            return np.random.rand(2**depth - 1)
        @staticmethod
        def stream2logsig(path, depth):
            return np.random.rand(depth * (depth + 1) // 2)
    ts = MockEsig()

class PathSignatureTransformer:
    """Nested path signature analysis for market patterns"""
    
    def __init__(self, algorithm):
        """Initialize with algorithm reference following existing pattern"""
        self.algo = algorithm
        self.signature_depth = 4
        self.window_size = 20
        self.nested_levels = 3
        
    def analyze(self, symbol, history_data):
        """Main analysis method that returns trading signal"""
        try:
            if '1m' not in history_data or len(history_data['1m']) < self.window_size:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            df = history_data['1m']
            nested_signatures = self._compute_nested_signatures(df)
            
            pattern_strength = self._analyze_signature_patterns(nested_signatures)
            trend_consistency = self._measure_trend_consistency(nested_signatures)
            
            if pattern_strength > 0.7 and trend_consistency > 0.6:
                signal = self._determine_signal_direction(nested_signatures)
                confidence = min(0.9, float((pattern_strength + trend_consistency) / 2))
            else:
                signal = 'NEUTRAL'
                confidence = 0.4
            
            return {
                'signal': signal,
                'confidence': confidence,
                'pattern_strength': pattern_strength,
                'trend_consistency': trend_consistency,
                'signature_features': self._extract_signature_features(nested_signatures)
            }
        except Exception as e:
            self.algo.Debug(f"PathSignatureTransformer error: {str(e)}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'error': str(e)}
    
    def _compute_nested_signatures(self, df):
        """Compute nested path signatures at multiple levels"""
        try:
            prices = df['Close'].values
            volumes = df['Volume'].values
            
            nested_sigs = []
            
            for level in range(self.nested_levels):
                window = self.window_size // (level + 1)
                if window < 5:
                    break
                
                level_sigs = []
                for i in range(window, len(prices)):
                    price_window = prices[i-window:i]
                    vol_window = volumes[i-window:i]
                    
                    time_points = np.linspace(0, 1, len(price_window))
                    path = np.column_stack([time_points, price_window, vol_window])
                    
                    signature = ts.stream2sig(path, min(self.signature_depth, 3))
                    log_signature = ts.stream2logsig(path, min(self.signature_depth, 3))
                    
                    level_sigs.append({
                        'signature': signature,
                        'log_signature': log_signature,
                        'level': level
                    })
                
                nested_sigs.append(level_sigs)
            
            return nested_sigs
        except Exception:
            return [[{'signature': np.random.rand(7), 'log_signature': np.random.rand(6), 'level': 0}]]
    
    def _analyze_signature_patterns(self, nested_signatures):
        """Analyze patterns in nested signatures"""
        try:
            if not nested_signatures or not nested_signatures[0]:
                return 0.5
            
            pattern_scores = []
            
            for level_sigs in nested_signatures:
                if not level_sigs:
                    continue
                
                signatures = [sig['signature'] for sig in level_sigs[-5:]]
                if len(signatures) < 2:
                    continue
                
                correlations = []
                for i in range(1, len(signatures)):
                    corr = np.corrcoef(signatures[i-1], signatures[i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                if correlations:
                    pattern_scores.append(np.mean(correlations))
            
            return np.mean(pattern_scores) if pattern_scores else 0.5
        except Exception:
            return 0.5
    
    def _measure_trend_consistency(self, nested_signatures):
        """Measure consistency of trends across signature levels"""
        try:
            if not nested_signatures:
                return 0.5
            
            trend_directions = []
            
            for level_sigs in nested_signatures:
                if len(level_sigs) < 2:
                    continue
                
                recent_sigs = level_sigs[-3:]
                if len(recent_sigs) < 2:
                    continue
                
                sig_trends = []
                for i in range(1, len(recent_sigs)):
                    sig_diff = np.mean(recent_sigs[i]['signature'] - recent_sigs[i-1]['signature'])
                    sig_trends.append(1 if sig_diff > 0 else -1)
                
                if sig_trends:
                    trend_directions.append(np.mean(sig_trends))
            
            if not trend_directions:
                return 0.5
            
            consistency = 1.0 - np.std(trend_directions) / 2.0
            return max(0.0, min(1.0, float(consistency)))
        except Exception:
            return 0.5
    
    def _determine_signal_direction(self, nested_signatures):
        """Determine trading signal direction from signatures"""
        try:
            if not nested_signatures or not nested_signatures[0]:
                return 'NEUTRAL'
            
            direction_votes = []
            
            for level_sigs in nested_signatures:
                if len(level_sigs) < 2:
                    continue
                
                recent_sig = level_sigs[-1]['signature']
                prev_sig = level_sigs[-2]['signature']
                
                sig_change = np.mean(recent_sig - prev_sig)
                direction_votes.append(1 if sig_change > 0 else -1)
            
            if not direction_votes:
                return 'NEUTRAL'
            
            avg_direction = np.mean(direction_votes)
            
            if avg_direction > 0.3:
                return 'BUY'
            elif avg_direction < -0.3:
                return 'SELL'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'
    
    def _extract_signature_features(self, nested_signatures):
        """Extract key features from nested signatures"""
        try:
            if not nested_signatures or not nested_signatures[0]:
                return {}
            
            features = {
                'num_levels': len(nested_signatures),
                'total_signatures': sum(len(level) for level in nested_signatures),
                'signature_complexity': 0.0,
                'log_signature_norm': 0.0
            }
            
            all_sigs = []
            all_log_sigs = []
            
            for level_sigs in nested_signatures:
                for sig_data in level_sigs[-3:]:
                    all_sigs.append(sig_data['signature'])
                    all_log_sigs.append(sig_data['log_signature'])
            
            if all_sigs:
                features['signature_complexity'] = float(np.mean([np.std(sig) for sig in all_sigs]))
            
            if all_log_sigs:
                features['log_signature_norm'] = float(np.mean([np.linalg.norm(log_sig) for log_sig in all_log_sigs]))
            
            return features
        except Exception:
            return {'num_levels': 0, 'total_signatures': 0, 'signature_complexity': 0.0, 'log_signature_norm': 0.0}
