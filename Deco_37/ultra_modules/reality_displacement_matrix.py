# Compares alternate timeline paths to catch manipulation signatures early.

import numpy as np
from scipy import stats
import math
from datetime import datetime, timedelta

class RealityDisplacementMatrix:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 100  # Number of candles to analyze
        self.timeline_count = 5  # Number of alternate timelines to generate
        self.displacement_threshold = 0.7
        self.manipulation_threshold = 0.75
        
    def decode(self, symbol, history_window):
        """
        Analyzes market data to detect reality displacement and manipulation.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - Dictionary with reality displacement detection results
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"RealityMatrix: Insufficient history for {symbol}")
            return {"manipulation_detected": False, "confidence": 0.0, "true_direction": "NEUTRAL"}
            
        closes = np.array([bar.Close for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        volumes = np.array([bar.Volume for bar in history_window])
        timestamps = [bar.EndTime for bar in history_window]
        
        timelines = self._generate_alternate_timelines(closes, highs, lows, volumes, timestamps)
        
        displacement_score = self._calculate_displacement_score(closes, timelines)
        
        manipulation_result = self._detect_manipulation(closes, volumes, timelines)
        
        true_direction = self._determine_true_direction(closes, timelines, manipulation_result["is_manipulated"])
        
        confidence = (displacement_score + manipulation_result["score"]) / 2
        
        if confidence > self.displacement_threshold:
            self.algo.Debug(f"RealityMatrix: {symbol} - Reality displacement detected! Score: {confidence:.2f}, True direction: {true_direction}")
            if manipulation_result["is_manipulated"]:
                self.algo.Debug(f"RealityMatrix: {symbol} - Manipulation pattern: {manipulation_result['pattern']}")
        
        return {
            "manipulation_detected": manipulation_result["is_manipulated"],
            "confidence": confidence,
            "true_direction": true_direction,
            "displacement_score": displacement_score,
            "manipulation_score": manipulation_result["score"],
            "manipulation_pattern": manipulation_result["pattern"] if manipulation_result["is_manipulated"] else None
        }
    
    def _generate_alternate_timelines(self, closes, highs, lows, volumes, timestamps):
        """Generate alternate price timelines based on different models"""
        timelines = []
        
        mean_reversion = self._generate_mean_reversion_timeline(closes)
        timelines.append(mean_reversion)
        
        momentum = self._generate_momentum_timeline(closes)
        timelines.append(momentum)
        
        volume_weighted = self._generate_volume_weighted_timeline(closes, volumes)
        timelines.append(volume_weighted)
        
        volatility_adjusted = self._generate_volatility_timeline(closes, highs, lows)
        timelines.append(volatility_adjusted)
        
        time_cycle = self._generate_time_cycle_timeline(closes, timestamps)
        timelines.append(time_cycle)
        
        return timelines
    
    def _generate_mean_reversion_timeline(self, closes):
        """Generate a mean reversion timeline"""
        window = 20
        timeline = np.copy(closes)
        
        for i in range(window, len(closes)):
            mean = np.mean(closes[i-window:i])
            std = np.std(closes[i-window:i])
            
            z_score = (closes[i] - mean) / std if std > 0 else 0
            reversion_strength = min(abs(z_score) / 2, 0.3)
            
            if closes[i] > mean:
                timeline[i] = closes[i] * (1 - reversion_strength)
            else:
                timeline[i] = closes[i] * (1 + reversion_strength)
                
        return timeline
    
    def _generate_momentum_timeline(self, closes):
        """Generate a momentum continuation timeline"""
        window = 10
        timeline = np.copy(closes)
        
        for i in range(window, len(closes)):
            momentum = (closes[i] / closes[i-window]) - 1
            
            momentum_strength = min(abs(momentum) * 5, 0.2)
            
            if momentum > 0:
                timeline[i] = closes[i] * (1 + momentum_strength)
            else:
                timeline[i] = closes[i] * (1 - momentum_strength)
                
        return timeline
    
    def _generate_volume_weighted_timeline(self, closes, volumes):
        """Generate a volume-weighted timeline"""
        window = 15
        timeline = np.copy(closes)
        
        for i in range(window, len(closes)):
            vwap = np.sum(closes[i-window:i] * volumes[i-window:i]) / np.sum(volumes[i-window:i])
            
            vol_mean = np.mean(volumes[i-window:i])
            vol_std = np.std(volumes[i-window:i])
            vol_z = (volumes[i] - vol_mean) / vol_std if vol_std > 0 else 0
            
            vol_impact = min(abs(vol_z) / 3, 0.25)
            
            if vol_z > 1.5 and closes[i] > vwap:
                timeline[i] = closes[i] * (1 + vol_impact)
            elif vol_z > 1.5 and closes[i] < vwap:
                timeline[i] = closes[i] * (1 - vol_impact)
            elif vol_z < -1.5:
                timeline[i] = closes[i] * (1 - (closes[i] - vwap) / closes[i] * 0.1)
                
        return timeline
    
    def _generate_volatility_timeline(self, closes, highs, lows):
        """Generate a volatility-adjusted timeline"""
        window = 20
        timeline = np.copy(closes)
        
        for i in range(window, len(closes)):
            ranges = highs[i-window:i] - lows[i-window:i]
            avg_range = np.mean(ranges)
            
            current_range = highs[i] - lows[i]
            
            vol_ratio = current_range / avg_range if avg_range > 0 else 1.0
            
            vol_impact = min((vol_ratio - 1) * 0.1, 0.2) if vol_ratio > 1 else max((vol_ratio - 1) * 0.1, -0.2)
            
            timeline[i] = closes[i] * (1 + vol_impact)
                
        return timeline
    
    def _generate_time_cycle_timeline(self, closes, timestamps):
        """Generate a time-cycle based timeline"""
        timeline = np.copy(closes)
        
        hours = [ts.hour for ts in timestamps]
        days = [ts.weekday() for ts in timestamps]
        
        for i in range(len(closes)):
            hour_effect = 0
            if hours[i] in [9, 10]:  # Opening hours often trend
                hour_effect = 0.05
            elif hours[i] in [15, 16]:  # Closing hours often reverse
                hour_effect = -0.05
                
            day_effect = 0
            if days[i] == 0:  # Monday effect
                day_effect = 0.02
            elif days[i] == 4:  # Friday effect
                day_effect = -0.02
                
            timeline[i] = closes[i] * (1 + hour_effect + day_effect)
                
        return timeline
    
    def _calculate_displacement_score(self, actual_prices, timelines):
        """Calculate how much actual prices deviate from alternate timelines"""
        recent_actual = actual_prices[-20:]
        
        deviations = []
        for timeline in timelines:
            recent_timeline = timeline[-20:]
            
            deviation = np.mean(np.abs(recent_actual - recent_timeline) / recent_actual)
            deviations.append(deviation)
            
        avg_deviation = np.mean(deviations)
        
        displacement_score = min(float(avg_deviation * 10), 1.0)
        
        return displacement_score
    
    def _detect_manipulation(self, closes, volumes, timelines):
        """Detect manipulation patterns by comparing actual to alternate timelines"""
        recent_closes = closes[-30:]
        recent_volumes = volumes[-30:]
        
        price_changes = np.diff(recent_closes) / recent_closes[:-1]
        
        patterns = {
            "PUMP_AND_DUMP": 0.0,
            "STOP_HUNTING": 0.0,
            "SPOOFING": 0.0,
            "LAYERING": 0.0
        }
        
        if len(price_changes) >= 10:
            pump_phase = price_changes[:5]
            dump_phase = price_changes[5:10]
            
            if np.mean(pump_phase) > 0.01 and np.mean(dump_phase) < -0.01:
                pump_vol = np.mean(recent_volumes[:5])
                dump_vol = np.mean(recent_volumes[5:10])
                
                if dump_vol > pump_vol * 1.5:
                    patterns["PUMP_AND_DUMP"] = 0.8
        
        if len(recent_closes) >= 15:
            for i in range(5, len(recent_closes) - 5):
                pre_move = (recent_closes[i] - recent_closes[i-5]) / recent_closes[i-5]
                post_move = (recent_closes[i+5] - recent_closes[i]) / recent_closes[i]
                
                if (pre_move > 0.02 and post_move < -0.02) or (pre_move < -0.02 and post_move > 0.02):
                    patterns["STOP_HUNTING"] = 0.7
        
        vol_changes = np.diff(recent_volumes) / recent_volumes[:-1]
        if len(vol_changes) >= 10:
            for i in range(len(vol_changes) - 5):
                if vol_changes[i] > 1.0 and abs(price_changes[i]) > 0.01:
                    if np.mean(vol_changes[i+1:i+5]) < 0:
                        patterns["SPOOFING"] = 0.6
        
        timeline_deviations = []
        for timeline in timelines:
            recent_timeline = timeline[-30:]
            
            if len(recent_closes) == len(recent_timeline):
                correlation = np.corrcoef(recent_closes, recent_timeline)[0, 1]
                timeline_deviations.append(1 - correlation)  # Higher value = more deviation
        
        avg_timeline_deviation = np.mean(timeline_deviations) if timeline_deviations else 0
        
        max_pattern_score = max(patterns.values())
        manipulation_score = max(float(max_pattern_score), float(avg_timeline_deviation))
        
        is_manipulated = manipulation_score > self.manipulation_threshold
        pattern = max(patterns.items(), key=lambda x: x[1])[0] if max_pattern_score > 0 else None
        
        return {
            "is_manipulated": is_manipulated,
            "score": manipulation_score,
            "pattern": pattern
        }
    
    def _determine_true_direction(self, closes, timelines, is_manipulated):
        """Determine the true market direction without manipulation"""
        if not is_manipulated:
            recent_trend = closes[-1] > closes[-10]
            return "BUY" if recent_trend else "SELL"
        
        buy_votes = 0
        sell_votes = 0
        
        for timeline in timelines:
            timeline_trend = timeline[-1] > timeline[-10]
            if timeline_trend:
                buy_votes += 1
            else:
                sell_votes += 1
        
        if buy_votes > sell_votes:
            return "BUY"
        elif sell_votes > buy_votes:
            return "SELL"
        else:
            return "NEUTRAL"
