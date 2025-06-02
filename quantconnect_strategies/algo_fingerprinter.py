"""
Algo Fingerprinter Strategy for QuantConnect
Adapted from Dark Pool DNA Decoder - keeps sklearn DBSCAN
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available - using simplified clustering")

class AlgoFingerprinter:
    """QuantConnect-compatible Algo Fingerprinter (adapted from Dark Pool DNA Decoder)"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.vix_threshold = 15.0
        self.features = []
        self.max_features = 100
        self.min_cluster_size = 5
        
        if SKLEARN_AVAILABLE:
            self.dbscan = DBSCAN(eps=0.3, min_samples=self.min_cluster_size)
        else:
            self.dbscan = None
    
    def analyze(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze order flow patterns and detect algo fingerprints"""
        try:
            if not market_data:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'No market data'}
            
            volatility = self._calculate_market_volatility(market_data)
            if volatility >= self.vix_threshold:
                return {
                    'signal': 'NEUTRAL', 
                    'confidence': 0.0, 
                    'reason': f'High volatility: {volatility:.2f}',
                    'volatility': volatility
                }
            
            order_features = self._extract_order_features(market_data)
            if len(order_features) < self.min_cluster_size:
                return {
                    'signal': 'NEUTRAL', 
                    'confidence': 0.0, 
                    'reason': 'Insufficient order data',
                    'volatility': volatility
                }
            
            pattern_result = self._detect_algo_patterns(order_features)
            
            if pattern_result['pattern_detected']:
                signal = pattern_result['signal']
                confidence = min(0.9, pattern_result['confidence'])
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'pattern': 'algo_detected',
                    'volatility': volatility,
                    'reason': f'Algo pattern detected with confidence {confidence:.3f}'
                }
            
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.4,
                'volatility': volatility,
                'reason': 'No significant algo patterns detected'
            }
            
        except Exception as e:
            self.algorithm.Error(f"AlgoFingerprinter error: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _extract_order_features(self, market_data: Dict[str, Any]) -> List[List[float]]:
        """Extract features from market data for clustering"""
        features = []
        
        if 'volumes' in market_data and 'prices' in market_data:
            volumes = market_data['volumes']
            prices = market_data['prices']
            
            if len(volumes) > 0 and len(prices) > 0:
                current_price = prices[-1] if prices else 100.0
                
                for i in range(min(len(volumes), len(prices), self.max_features)):
                    volume = volumes[i] if i < len(volumes) else 1000
                    price = prices[i] if i < len(prices) else current_price
                    price_deviation = (price - current_price) / current_price
                    
                    features.append([volume, price_deviation])
        
        return features[-self.max_features:] if len(features) > self.max_features else features
    
    def _detect_algo_patterns(self, features: List[List[float]]) -> Dict[str, Any]:
        """Detect algorithmic trading patterns using clustering"""
        if not features or len(features) < self.min_cluster_size:
            return {'pattern_detected': False, 'signal': 'NEUTRAL', 'confidence': 0.0}
        
        try:
            if self.dbscan and SKLEARN_AVAILABLE:
                labels = self.dbscan.fit_predict(np.array(features))
                
                if len(labels) > 0 and labels[-1] != -1:
                    unique_labels = len(set(labels[labels != -1]))
                    cluster_ratio = unique_labels / len(labels) if len(labels) > 0 else 0
                    
                    last_feature = features[-1]
                    signal = 'SELL' if last_feature[0] > np.median([f[0] for f in features]) else 'BUY'
                    confidence = min(0.9, cluster_ratio * 2)
                    
                    return {
                        'pattern_detected': True,
                        'signal': signal,
                        'confidence': confidence
                    }
            else:
                return self._simple_pattern_detection(features)
                
        except Exception as e:
            self.algorithm.Error(f"Pattern detection error: {e}")
            return self._simple_pattern_detection(features)
        
        return {'pattern_detected': False, 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _simple_pattern_detection(self, features: List[List[float]]) -> Dict[str, Any]:
        """Simple pattern detection when sklearn is not available"""
        if len(features) < 3:
            return {'pattern_detected': False, 'signal': 'NEUTRAL', 'confidence': 0.0}
        
        volumes = [f[0] for f in features]
        price_devs = [f[1] for f in features]
        
        volume_trend = np.mean(volumes[-3:]) - np.mean(volumes[:-3])
        price_trend = np.mean(price_devs[-3:]) - np.mean(price_devs[:-3])
        
        if abs(volume_trend) > np.std(volumes) and abs(price_trend) > np.std(price_devs):
            signal = 'BUY' if price_trend > 0 else 'SELL'
            confidence = min(0.8, float((abs(volume_trend) + abs(price_trend)) / 2))
            return {
                'pattern_detected': True,
                'signal': signal,
                'confidence': confidence
            }
        
        return {'pattern_detected': False, 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility equivalent to VIX"""
        try:
            if 'returns' in market_data and market_data['returns']:
                returns = market_data['returns']
                return float(np.std(returns) * 100) if len(returns) > 0 else 0.0
            elif 'prices' in market_data and market_data['prices']:
                prices = market_data['prices']
                if len(prices) > 1:
                    returns = np.diff(prices) / prices[:-1]
                    return float(np.std(returns) * 100)
            return 0.0
        except Exception:
            return 0.0
    
    def generate_signal(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate signal for AI consensus integration"""
        return self.analyze(market_data, symbol)
