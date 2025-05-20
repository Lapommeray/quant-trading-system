# Detects destabilizing market shifts from rare unseen conditions.

import numpy as np
from scipy import stats
import math

class BlackSwanProtector:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 100  # Number of candles to analyze
        self.volatility_window = 20  # Window for volatility calculation
        self.tail_risk_threshold = 3.0  # Z-score threshold for tail risk
        self.correlation_window = 30  # Window for correlation analysis
        
    def decode(self, symbol, history_window):
        """
        Analyzes market data for potential black swan events.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - Dictionary with black swan detection results
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"BlackSwan: Insufficient history for {symbol}")
            return {"black_swan_risk": 0.0, "protection_active": False, "risk_type": None}
            
        closes = np.array([bar.Close for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        volumes = np.array([bar.Volume for bar in history_window])
        
        volatility_risk = self._calculate_volatility_risk(closes)
        tail_risk = self._calculate_tail_risk(closes, volumes)
        correlation_risk = self._calculate_correlation_breakdown(closes)
        liquidity_risk = self._calculate_liquidity_risk(volumes)
        
        risk_score = (
            0.3 * volatility_risk["score"] + 
            0.3 * tail_risk["score"] + 
            0.2 * correlation_risk["score"] + 
            0.2 * liquidity_risk["score"]
        )
        
        if risk_score > 0.7:
            if volatility_risk["score"] > max(tail_risk["score"], correlation_risk["score"], liquidity_risk["score"]):
                risk_type = "VOLATILITY_EXPLOSION"
            elif tail_risk["score"] > max(correlation_risk["score"], liquidity_risk["score"]):
                risk_type = "TAIL_EVENT"
            elif correlation_risk["score"] > liquidity_risk["score"]:
                risk_type = "CORRELATION_BREAKDOWN"
            else:
                risk_type = "LIQUIDITY_CRISIS"
                
            self.algo.Debug(f"BlackSwan: {symbol} - High risk detected! Score: {risk_score:.2f}, Type: {risk_type}")
            protection_active = True
        else:
            risk_type = None
            protection_active = False
            
        return {
            "black_swan_risk": risk_score,
            "protection_active": protection_active,
            "risk_type": risk_type,
            "volatility_risk": volatility_risk["score"],
            "tail_risk": tail_risk["score"],
            "correlation_risk": correlation_risk["score"],
            "liquidity_risk": liquidity_risk["score"]
        }
    
    def _calculate_volatility_risk(self, closes):
        """Calculate volatility explosion risk"""
        if len(closes) < self.volatility_window * 2:
            return {"score": 0.0}
        
        returns = np.diff(closes) / closes[:-1]
        vol_windows = []
        
        for i in range(len(returns) - self.volatility_window + 1):
            window_vol = np.std(returns[i:i+self.volatility_window])
            vol_windows.append(window_vol)
        
        recent_vol = vol_windows[-1]
        historical_vol = np.mean(vol_windows[:-5])
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        vol_acceleration = 0
        if len(vol_windows) >= 3:
            vol_acceleration = (vol_windows[-1] - vol_windows[-2]) - (vol_windows[-2] - vol_windows[-3])
        
        score = min(vol_ratio / 3.0, 1.0)  # Cap at 1.0
        if vol_acceleration > 0:
            score = min(score + vol_acceleration * 10, 1.0)  # Add acceleration component
            
        return {"score": score}
    
    def _calculate_tail_risk(self, closes, volumes):
        """Calculate tail risk based on extreme price movements"""
        if len(closes) < 30:
            return {"score": 0.0}
        
        returns = np.diff(closes) / closes[:-1]
        
        mu, sigma = stats.norm.fit(returns)
        
        z_scores = [(r - mu) / sigma for r in returns[-10:]]
        max_z_score = max(abs(z) for z in z_scores)
        
        vol_mean = np.mean(volumes[:-10])
        vol_std = np.std(volumes[:-10])
        
        recent_vol_z_scores = [(v - vol_mean) / vol_std for v in volumes[-10:]]
        
        combined_extremes = sum(1 for i in range(len(z_scores)) 
                               if abs(z_scores[i]) > 2.0 and recent_vol_z_scores[i] > 2.0)
        
        score = min(max_z_score / self.tail_risk_threshold, 1.0)
        if combined_extremes > 0:
            score = min(score + 0.1 * combined_extremes, 1.0)
            
        return {"score": score}
    
    def _calculate_correlation_breakdown(self, closes):
        """Calculate risk of correlation breakdown"""
        if len(closes) < self.correlation_window * 2:
            return {"score": 0.0}
        
        correlation_changes = []
        
        for i in range(len(closes) - self.correlation_window * 2):
            segment1 = closes[i:i+self.correlation_window]
            segment2 = closes[i+self.correlation_window:i+self.correlation_window*2]
            
            time_array1 = np.arange(len(segment1))
            time_array2 = np.arange(len(segment2))
            
            corr1 = np.corrcoef(segment1, time_array1)[0, 1]
            corr2 = np.corrcoef(segment2, time_array2)[0, 1]
            
            correlation_changes.append(abs(corr2 - corr1))
        
        recent_changes = np.mean(correlation_changes[-5:]) if len(correlation_changes) >= 5 else 0
        historical_changes = np.mean(correlation_changes[:-5]) if len(correlation_changes) >= 10 else 0
        
        score = min(float(recent_changes / 0.5), 1.0)  # Cap at 1.0
        if recent_changes > historical_changes * 2:
            score = min(score * 1.5, 1.0)  # Amplify if recent changes are much larger
            
        return {"score": score}
    
    def _calculate_liquidity_risk(self, volumes):
        """Calculate liquidity crisis risk"""
        if len(volumes) < 30:
            return {"score": 0.0}
        
        recent_vol = np.mean(volumes[-5:])
        historical_vol = np.mean(volumes[-30:-5])
        
        vol_ratio = historical_vol / recent_vol if recent_vol > 0 else 1.0
        
        vol_std = np.std(volumes[-5:])
        vol_mean = np.mean(volumes[-5:])
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0  # Coefficient of variation
        
        score = min(vol_ratio / 3.0, 1.0)  # Cap at 1.0
        if vol_cv > 1.0:  # High variation in recent volumes
            score = min(score + vol_cv * 0.2, 1.0)
            
        return {"score": score}
