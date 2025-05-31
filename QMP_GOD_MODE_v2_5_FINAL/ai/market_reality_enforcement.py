"""
Market Reality Enforcement Engine - Ensures signals align with market reality
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

class MarketRealityEnforcement:
    """
    Enforces market reality constraints to prevent impossible trades
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.logger = logging.getLogger("MarketRealityEnforcement")
        self.reality_checks = {
            'liquidity_check': True,
            'volatility_check': True,
            'fundamental_check': True,
            'technical_check': True,
            'sentiment_check': True
        }
        
    def enforce_reality(self, signal, confidence, market_data, symbol):
        """
        Enforce market reality constraints on trading signals
        """
        reality_score = 1.0
        enforcement_results = {}
        
        liquidity_result = self._check_liquidity_reality(market_data, symbol)
        enforcement_results['liquidity'] = liquidity_result
        reality_score *= liquidity_result['multiplier']
        
        volatility_result = self._check_volatility_reality(market_data, confidence)
        enforcement_results['volatility'] = volatility_result
        reality_score *= volatility_result['multiplier']
        
        technical_result = self._check_technical_reality(signal, market_data)
        enforcement_results['technical'] = technical_result
        reality_score *= technical_result['multiplier']
        
        fundamental_result = self._check_fundamental_reality(signal, symbol)
        enforcement_results['fundamental'] = fundamental_result
        reality_score *= fundamental_result['multiplier']
        
        market_hours_result = self._check_market_hours_reality(symbol)
        enforcement_results['market_hours'] = market_hours_result
        reality_score *= market_hours_result['multiplier']
        
        enforced_confidence = confidence * reality_score
        enforced_signal = signal if reality_score > 0.5 else "NEUTRAL"
        
        return {
            'original_signal': signal,
            'original_confidence': confidence,
            'enforced_signal': enforced_signal,
            'enforced_confidence': enforced_confidence,
            'reality_score': reality_score,
            'enforcement_results': enforcement_results,
            'reality_compliant': reality_score > 0.8
        }
    
    def _check_liquidity_reality(self, market_data, symbol):
        """Check if signal is realistic given market liquidity"""
        volume = market_data.get('volume', [])
        
        if not volume or len(volume) < 5:
            return {'passed': False, 'multiplier': 0.5, 'reason': 'insufficient_volume_data'}
        
        recent_volume = np.mean(volume[-5:])
        avg_volume = np.mean(volume) if len(volume) > 10 else recent_volume
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio < 0.1:  # Very low volume
            return {'passed': False, 'multiplier': 0.3, 'reason': 'extremely_low_volume'}
        elif volume_ratio < 0.5:  # Low volume
            return {'passed': True, 'multiplier': 0.7, 'reason': 'low_volume'}
        elif volume_ratio > 3.0:  # Very high volume
            return {'passed': True, 'multiplier': 1.2, 'reason': 'high_volume_boost'}
        else:
            return {'passed': True, 'multiplier': 1.0, 'reason': 'normal_volume'}
    
    def _check_volatility_reality(self, market_data, confidence):
        """Check if confidence is realistic given market volatility"""
        returns = market_data.get('returns', [])
        
        if not returns or len(returns) < 10:
            return {'passed': True, 'multiplier': 0.8, 'reason': 'insufficient_data'}
        
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        if volatility > 0.05:  # Very high volatility
            multiplier = max(0.4, float(1.0 - (volatility - 0.05) * 10))
            return {'passed': True, 'multiplier': multiplier, 'reason': 'high_volatility_adjustment'}
        elif volatility > 0.02:  # Moderate volatility
            multiplier = max(0.7, float(1.0 - (volatility - 0.02) * 5))
            return {'passed': True, 'multiplier': multiplier, 'reason': 'moderate_volatility_adjustment'}
        else:  # Low volatility
            return {'passed': True, 'multiplier': 1.1, 'reason': 'low_volatility_boost'}
    
    def _check_technical_reality(self, signal, market_data):
        """Check if signal aligns with technical reality"""
        returns = market_data.get('returns', [])
        
        if not returns or len(returns) < 5:
            return {'passed': True, 'multiplier': 0.9, 'reason': 'insufficient_data'}
        
        recent_trend = np.mean(returns[-5:])
        
        if signal == "BUY" and recent_trend < -0.02:  # Buying in strong downtrend
            return {'passed': False, 'multiplier': 0.5, 'reason': 'signal_trend_misalignment'}
        elif signal == "SELL" and recent_trend > 0.02:  # Selling in strong uptrend
            return {'passed': False, 'multiplier': 0.5, 'reason': 'signal_trend_misalignment'}
        else:
            return {'passed': True, 'multiplier': 1.0, 'reason': 'signal_trend_aligned'}
    
    def _check_fundamental_reality(self, signal, symbol):
        """Check if signal aligns with fundamental reality"""
        current_time = datetime.now()
        
        if current_time.hour in [8, 9, 14, 15]:  # Major announcement hours
            return {'passed': True, 'multiplier': 0.8, 'reason': 'major_event_hours'}
        
        if current_time.hour >= 15 and current_time.weekday() == 4:  # Friday afternoon
            return {'passed': True, 'multiplier': 0.9, 'reason': 'end_of_week_effect'}
        
        return {'passed': True, 'multiplier': 1.0, 'reason': 'normal_conditions'}
    
    def _check_market_hours_reality(self, symbol):
        """Check if trading during appropriate market hours"""
        current_time = datetime.now()
        
        if current_time.weekday() >= 5:  # Weekend
            return {'passed': False, 'multiplier': 0.0, 'reason': 'market_closed_weekend'}
        
        hour = current_time.hour
        if hour < 9 or hour > 16:  # Outside market hours
            return {'passed': False, 'multiplier': 0.2, 'reason': 'outside_market_hours'}
        
        return {'passed': True, 'multiplier': 1.0, 'reason': 'market_hours_active'}
