"""
Fractal Pattern Analyzer

A realistic implementation of fractal pattern matching and historical comparison
instead of fictional time fractal prediction with attosecond precision.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime
import logging
import json

class FractalPatternAnalyzer:
    """
    A realistic implementation of fractal pattern matching and historical comparison.
    Replaces the fictional TimeFractalPredictor with practical techniques.
    """
    
    def __init__(self, lookback_periods=200, pattern_length=50, min_similarity=0.7, max_patterns=10):
        """
        Initialize the FractalPatternAnalyzer
        
        Parameters:
        - lookback_periods: Number of periods to look back for pattern matching
        - pattern_length: Length of patterns to match
        - min_similarity: Minimum similarity threshold for pattern matching
        - max_patterns: Maximum number of patterns to store
        """
        self.lookback_periods = lookback_periods
        self.pattern_length = pattern_length
        self.min_similarity = min_similarity
        self.max_patterns = max_patterns
        
        self.historical_patterns = {}
        self.current_matches = {}
        self.last_match = None
        self.match_history = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("FractalPatternAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _normalize_pattern(self, pattern):
        """
        Normalize a price pattern to 0-1 range
        
        Parameters:
        - pattern: Array of prices
        
        Returns:
        - Normalized pattern
        """
        if len(pattern) < 2:
            return pattern
        
        pattern_reshaped = pattern.reshape(-1, 1)
        
        normalized = self.scaler.fit_transform(pattern_reshaped).flatten()
        
        return normalized
    
    def _calculate_similarity(self, pattern1, pattern2):
        """
        Calculate similarity between two patterns
        
        Parameters:
        - pattern1: First pattern
        - pattern2: Second pattern
        
        Returns:
        - Similarity score (0-1)
        """
        if len(pattern1) != len(pattern2):
            min_length = min(len(pattern1), len(pattern2))
            pattern1 = pattern1[:min_length]
            pattern2 = pattern2[:min_length]
        
        try:
            correlation, _ = pearsonr(pattern1, pattern2)
            similarity = (correlation + 1) / 2
        except:
            similarity = 0
        
        rmse = np.sqrt(mean_squared_error(pattern1, pattern2))
        rmse_similarity = 1 / (1 + rmse)
        
        combined_similarity = 0.7 * similarity + 0.3 * rmse_similarity
        
        return combined_similarity
    
    def _extract_pattern(self, price_data, start_idx, length=None):
        """
        Extract a price pattern from data
        
        Parameters:
        - price_data: Array of prices
        - start_idx: Starting index
        - length: Pattern length
        
        Returns:
        - Extracted pattern
        """
        if length is None:
            length = self.pattern_length
        
        if start_idx + length > len(price_data):
            length = len(price_data) - start_idx
        
        pattern = price_data[start_idx:start_idx + length]
        
        normalized_pattern = self._normalize_pattern(pattern)
        
        return normalized_pattern
    
    def _find_pivot_points(self, price_data, prominence=0.1):
        """
        Find pivot points in price data
        
        Parameters:
        - price_data: Array of prices
        - prominence: Peak prominence threshold
        
        Returns:
        - Indices of pivot points
        """
        normalized = self._normalize_pattern(price_data)
        
        peaks, _ = find_peaks(normalized, prominence=prominence)
        troughs, _ = find_peaks(-normalized, prominence=prominence)
        
        pivots = np.sort(np.concatenate([peaks, troughs]))
        
        return pivots
    
    def add_historical_pattern(self, symbol, price_data, outcome_data=None, label=None):
        """
        Add a historical pattern to the database
        
        Parameters:
        - symbol: Symbol for the pattern
        - price_data: Array of prices
        - outcome_data: Data following the pattern (optional)
        - label: Pattern label (optional)
        
        Returns:
        - Pattern ID
        """
        if symbol not in self.historical_patterns:
            self.historical_patterns[symbol] = []
        
        if isinstance(price_data, pd.Series):
            price_data = price_data.values
        
        normalized_pattern = self._normalize_pattern(price_data)
        
        pattern_id = f"{symbol}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.historical_patterns[symbol])}"
        
        pattern = {
            "id": pattern_id,
            "symbol": symbol,
            "pattern": normalized_pattern,
            "timestamp": datetime.datetime.now(),
            "label": label
        }
        
        if outcome_data is not None:
            if isinstance(outcome_data, pd.Series):
                outcome_data = outcome_data.values
            
            if len(outcome_data) > 0:
                direction = "UP" if outcome_data[-1] > outcome_data[0] else "DOWN"
                
                magnitude = (outcome_data[-1] / outcome_data[0] - 1) * 100
                
                volatility = np.std(np.diff(outcome_data) / outcome_data[:-1]) * 100
                
                max_drawdown = 0
                peak = outcome_data[0]
                
                for price in outcome_data:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                max_drawdown *= 100
                
                pattern["outcome"] = {
                    "direction": direction,
                    "magnitude": magnitude,
                    "volatility": volatility,
                    "max_drawdown": max_drawdown,
                    "data": self._normalize_pattern(outcome_data)
                }
        
        self.historical_patterns[symbol].append(pattern)
        
        if len(self.historical_patterns[symbol]) > self.max_patterns:
            self.historical_patterns[symbol] = self.historical_patterns[symbol][-self.max_patterns:]
        
        return pattern_id
    
    def find_matching_fractal(self, symbol, current_data, outcome_length=20):
        """
        Find matching historical patterns
        
        Parameters:
        - symbol: Symbol to analyze
        - current_data: Current price data
        - outcome_length: Length of outcome to predict
        
        Returns:
        - Dictionary with matching pattern details
        """
        if symbol not in self.historical_patterns or not self.historical_patterns[symbol]:
            return {
                "match_found": False,
                "confidence": 0.0,
                "next_move": "UNKNOWN",
                "time_left": None,
                "details": f"No historical patterns for {symbol}"
            }
        
        if isinstance(current_data, pd.Series):
            current_data = current_data.values
        
        current_pattern = self._normalize_pattern(current_data[-self.pattern_length:])
        
        best_match = None
        best_similarity = 0
        
        for pattern in self.historical_patterns[symbol]:
            historical_pattern = pattern["pattern"]
            
            if len(historical_pattern) < len(current_pattern):
                continue
            
            for i in range(len(historical_pattern) - len(current_pattern) + 1):
                segment = historical_pattern[i:i+len(current_pattern)]
                similarity = self._calculate_similarity(current_pattern, segment)
                
                if similarity > best_similarity and similarity >= self.min_similarity:
                    best_similarity = similarity
                    best_match = {
                        "pattern": pattern,
                        "alignment_offset": i,
                        "similarity": similarity
                    }
        
        if best_match is None:
            return {
                "match_found": False,
                "confidence": 0.0,
                "next_move": "UNKNOWN",
                "time_left": None,
                "details": f"No matching pattern found for {symbol}"
            }
        
        pattern = best_match["pattern"]
        alignment_offset = best_match["alignment_offset"]
        
        next_move = "UNKNOWN"
        time_left = None
        magnitude = 0
        volatility = 0
        max_drawdown = 0
        
        if "outcome" in pattern:
            outcome = pattern["outcome"]
            next_move = outcome["direction"]
            magnitude = outcome["magnitude"]
            volatility = outcome["volatility"]
            max_drawdown = outcome["max_drawdown"]
            
            remaining_pattern = len(pattern["pattern"]) - (alignment_offset + len(current_pattern))
            time_left = remaining_pattern  # In bars/periods
        
        result = {
            "match_found": True,
            "pattern_id": pattern["id"],
            "similarity": best_similarity,
            "confidence": best_similarity,
            "next_move": next_move,
            "time_left": time_left,
            "magnitude": magnitude,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "label": pattern.get("label"),
            "timestamp": datetime.datetime.now()
        }
        
        self.logger.info(f"Pattern match found for {symbol}: {result['pattern_id']} with {result['similarity']:.2f} similarity")
        
        self.last_match = result
        self.match_history.append(result)
        
        if symbol not in self.current_matches:
            self.current_matches[symbol] = []
        
        self.current_matches[symbol].append(result)
        
        if len(self.match_history) > 100:
            self.match_history = self.match_history[-100:]
        
        if len(self.current_matches[symbol]) > 10:
            self.current_matches[symbol] = self.current_matches[symbol][-10:]
        
        return result
    
    def get_match_statistics(self, symbol=None, lookback_days=7):
        """
        Get match statistics
        
        Parameters:
        - symbol: Symbol to analyze (optional)
        - lookback_days: Number of days to look back
        
        Returns:
        - Dictionary with match statistics
        """
        if symbol:
            matches = [m for m in self.match_history if m.get("symbol") == symbol]
        else:
            matches = self.match_history
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
        recent_matches = [m for m in matches if m.get("timestamp", datetime.datetime.now()) > cutoff_time]
        
        total_matches = len(recent_matches)
        
        if total_matches == 0:
            return {
                "total_matches": 0,
                "avg_similarity": 0.0,
                "up_predictions": 0,
                "down_predictions": 0,
                "unknown_predictions": 0,
                "lookback_days": lookback_days,
                "symbol": symbol
            }
        
        avg_similarity = sum(m.get("similarity", 0) for m in recent_matches) / total_matches
        
        up_predictions = sum(1 for m in recent_matches if m.get("next_move") == "UP")
        down_predictions = sum(1 for m in recent_matches if m.get("next_move") == "DOWN")
        unknown_predictions = sum(1 for m in recent_matches if m.get("next_move") == "UNKNOWN")
        
        return {
            "total_matches": total_matches,
            "avg_similarity": avg_similarity,
            "up_predictions": up_predictions,
            "down_predictions": down_predictions,
            "unknown_predictions": unknown_predictions,
            "lookback_days": lookback_days,
            "symbol": symbol
        }
    
    def load_patterns_from_file(self, filepath):
        """
        Load patterns from a JSON file
        
        Parameters:
        - filepath: Path to JSON file
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.historical_patterns = data
            return True
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return False
    
    def save_patterns_to_file(self, filepath):
        """
        Save patterns to a JSON file
        
        Parameters:
        - filepath: Path to JSON file
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            serializable_patterns = {}
            
            for symbol, patterns in self.historical_patterns.items():
                serializable_patterns[symbol] = []
                
                for pattern in patterns:
                    serializable_pattern = pattern.copy()
                    
                    if "pattern" in serializable_pattern and isinstance(serializable_pattern["pattern"], np.ndarray):
                        serializable_pattern["pattern"] = serializable_pattern["pattern"].tolist()
                    
                    if "outcome" in serializable_pattern and "data" in serializable_pattern["outcome"] and isinstance(serializable_pattern["outcome"]["data"], np.ndarray):
                        serializable_pattern["outcome"]["data"] = serializable_pattern["outcome"]["data"].tolist()
                    
                    if "timestamp" in serializable_pattern and isinstance(serializable_pattern["timestamp"], datetime.datetime):
                        serializable_pattern["timestamp"] = serializable_pattern["timestamp"].isoformat()
                    
                    serializable_patterns[symbol].append(serializable_pattern)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_patterns, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")
            return False
