# Finds microscopic price shifts (quantum tremors) before larger moves.

import numpy as np
from scipy import stats

class QuantumTremorScanner:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 60  # Number of candles to analyze
        self.tremor_threshold = 0.65
        self.micro_window = 5  # Window for micro-pattern detection
        
    def decode(self, symbol, history_window):
        """
        Analyzes price action for quantum tremors - microscopic patterns that precede larger moves.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - Dictionary with tremor detection results
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"QuantumTremor: Insufficient history for {symbol}")
            return {"tremors_detected": False, "direction": None, "confidence": 0.0}
            
        closes = np.array([bar.Close for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        volumes = np.array([bar.Volume for bar in history_window])
        
        micro_volatility = self._calculate_micro_volatility(closes)
        volume_anomalies = self._detect_volume_anomalies(volumes)
        price_microstructures = self._analyze_price_microstructure(highs, lows, closes)
        
        tremor_score = (
            0.4 * micro_volatility["score"] + 
            0.3 * volume_anomalies["score"] + 
            0.3 * price_microstructures["score"]
        )
        
        if tremor_score > self.tremor_threshold:
            direction = self._determine_likely_direction(micro_volatility, volume_anomalies, price_microstructures)
            self.algo.Debug(f"QuantumTremor: {symbol} - Tremors detected! Score: {tremor_score:.2f}, Direction: {direction}")
            return {
                "tremors_detected": True,
                "direction": direction,
                "confidence": tremor_score,
                "micro_volatility": micro_volatility["value"],
                "volume_anomaly": volume_anomalies["value"],
                "microstructure": price_microstructures["value"],
                "5d_scanning_activated": True  # Confirm 5D scanning activation
            }
        else:
            self.algo.Debug(f"QuantumTremor: {symbol} - No significant tremors. Score: {tremor_score:.2f}")
            return {"tremors_detected": False, "direction": None, "confidence": tremor_score, "5d_scanning_activated": True}
    
    def _calculate_micro_volatility(self, closes):
        """Detect changes in micro-volatility patterns"""
        micro_std = np.zeros(len(closes) - self.micro_window + 1)
        for i in range(len(micro_std)):
            micro_std[i] = np.std(closes[i:i+self.micro_window])
        
        micro_std_change = np.diff(micro_std)
        
        recent_changes = micro_std_change[-10:]
        baseline_changes = micro_std_change[:-10]
        
        if len(baseline_changes) == 0:
            return {"score": 0.0, "value": 0.0, "direction": None}
        
        mean_baseline = np.mean(baseline_changes)
        std_baseline = np.std(baseline_changes) if np.std(baseline_changes) > 0 else 1e-9
        
        z_scores = [(x - mean_baseline) / std_baseline for x in recent_changes]
        max_z_score = max(abs(z) for z in z_scores)
        
        direction = "BUY" if np.mean(recent_changes) > 0 else "SELL"
        
        score = min(max_z_score / 3.0, 1.0)  # Cap at 1.0, 3 sigma is considered significant
        
        return {"score": score, "value": max_z_score, "direction": direction}
    
    def _detect_volume_anomalies(self, volumes):
        """Detect unusual volume patterns at the micro level"""
        if len(volumes) < 20:
            return {"score": 0.0, "value": 0.0, "direction": None}
        
        window = 10
        rolling_avg = np.zeros(len(volumes) - window + 1)
        for i in range(len(rolling_avg)):
            rolling_avg[i] = np.mean(volumes[i:i+window])
        
        recent_volumes = volumes[-5:]
        recent_avg = rolling_avg[-5] if len(rolling_avg) >= 5 else np.mean(volumes[-15:-5])
        
        volume_ratio = np.mean(recent_volumes) / recent_avg if recent_avg > 0 else 1.0
        
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        direction = "BUY" if volume_trend > 0 else "SELL"
        
        score = min(abs(volume_ratio - 1.0), 1.0)
        
        return {"score": score, "value": volume_ratio, "direction": direction}
    
    def _analyze_price_microstructure(self, highs, lows, closes):
        """Analyze price microstructure for patterns"""
        if len(closes) < 20:
            return {"score": 0.0, "value": 0.0, "direction": None}
        
        ranges = highs - lows
        
        recent_ranges = ranges[-10:]
        baseline_ranges = ranges[:-10]
        
        if len(baseline_ranges) == 0:
            return {"score": 0.0, "value": 0.0, "direction": None}
        
        compression_ratio = np.mean(baseline_ranges) / np.mean(recent_ranges) if np.mean(recent_ranges) > 0 else 1.0
        
        recent_moves = np.diff(closes[-11:])
        positive_moves = sum(1 for move in recent_moves if move > 0)
        negative_moves = sum(1 for move in recent_moves if move < 0)
        
        direction = "BUY" if positive_moves > negative_moves else "SELL"
        
        compression_score = min(abs(compression_ratio - 1.0), 1.0)
        consistency_score = max(positive_moves, negative_moves) / len(recent_moves)
        
        score = 0.6 * compression_score + 0.4 * consistency_score
        
        return {"score": score, "value": compression_ratio, "direction": direction}
    
    def _determine_likely_direction(self, micro_volatility, volume_anomalies, price_microstructures):
        """Determine the most likely direction based on all tremor indicators"""
        directions = [
            (micro_volatility["direction"], 0.4),
            (volume_anomalies["direction"], 0.3),
            (price_microstructures["direction"], 0.3)
        ]
        
        buy_votes = sum(weight for direction, weight in directions if direction == "BUY")
        sell_votes = sum(weight for direction, weight in directions if direction == "SELL")
        
        return "BUY" if buy_votes >= sell_votes else "SELL"
