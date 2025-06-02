"""
LSTM Liquidity Predictor Strategy for QuantConnect
Adapted from Neural Market Holography - removes torch dependency
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

class LSTMLiquidityPredictor:
    """QuantConnect-compatible LSTM Liquidity Predictor (adapted from Neural Market Holography)"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.sequence_length = 10
        self.vix_threshold = 15.0
        self.prediction_threshold = 0.7
        self.min_data_points = 20
        
    def analyze(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze liquidity patterns and predict future movements"""
        try:
            if not market_data or 'prices' not in market_data:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'No market data'}
            
            prices = market_data['prices']
            if len(prices) < self.min_data_points:
                return {
                    'signal': 'NEUTRAL', 
                    'confidence': 0.0, 
                    'reason': f'Insufficient data: {len(prices)} < {self.min_data_points}'
                }
            
            volatility = self._calculate_volatility(prices)
            vix_equivalent = volatility * 100
            
            if vix_equivalent >= self.vix_threshold:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'reason': f'High volatility: {vix_equivalent:.2f}',
                    'volatility': volatility
                }
            
            prediction = self._simplified_lstm_prediction(market_data)
            
            if prediction > self.prediction_threshold:
                signal = 'BUY'
                confidence = min(0.9, prediction)
            elif prediction < (1 - self.prediction_threshold):
                signal = 'SELL'
                confidence = min(0.9, 1 - prediction)
            else:
                signal = 'NEUTRAL'
                confidence = 0.4
            
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'volatility': volatility,
                'reason': f'LSTM prediction: {prediction:.3f}'
            }
            
        except Exception as e:
            self.algorithm.Error(f"LSTMLiquidityPredictor error: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _simplified_lstm_prediction(self, market_data: Dict[str, Any]) -> float:
        """Simplified LSTM-like prediction using statistical methods"""
        try:
            prices = market_data['prices']
            volumes = market_data.get('volumes', [1000] * len(prices))
            
            if len(prices) < self.sequence_length:
                return 0.5
            
            closes = np.array(prices[-self.sequence_length:])
            vols = np.array(volumes[-self.sequence_length:]) if len(volumes) >= self.sequence_length else np.ones(self.sequence_length)
            
            short_ma = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            long_ma = np.mean(closes[-self.sequence_length:])
            
            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
            
            volume_ratio = (np.mean(vols[-3:]) / np.mean(vols[-self.sequence_length:])) if len(vols) >= 3 else 1.0
            
            trend_strength = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            prediction = 0.5 + (momentum * 0.3) + (trend_strength * 0.2) + ((volume_ratio - 1) * 0.1)
            
            prediction = max(0.0, min(1.0, float(prediction)))
            
            return prediction
            
        except Exception as e:
            self.algorithm.Error(f"LSTM prediction error: {e}")
            return 0.5
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate volatility using existing pattern"""
        try:
            if len(prices) < 2:
                return 0.001
            
            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return 0.001
                
            return max(float(np.std(returns)), 0.0005)
            
        except Exception:
            return 0.001
    
    def generate_signal(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate signal for AI consensus integration"""
        return self.analyze(market_data, symbol)
