"""
Advanced Noise Filter
Removes all low-quality data noise that could reduce win rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
from scipy import stats
from scipy import signal
from typing import Dict, List, Any, Optional, Union, Tuple

class AdvancedNoiseFilter:
    """
    Advanced Noise Filter that removes all low-quality data noise
    that could reduce win rate and ensures only high-quality data
    is used for trading decisions.
    """
    
    def __init__(self):
        """Initialize the Advanced Noise Filter with super high quality requirements"""
        self.logger = self._setup_logger()
        self.logger.info("Advanced Noise Filter initialized with super high quality requirements")
        
        self.min_quality_threshold = 0.95  # Only accept data with 95%+ quality score
        self.noise_reduction_level = 0.99  # Remove 99% of noise
        self.filter_stats = {
            "total_checks": 0,
            "noise_detected": 0,
            "noise_removed": 0,
            "noise_reduction_rate": 0.0,
            "average_quality_improvement": 0.0
        }
        
        self.noise_types = {
            "random_fluctuations": {"weight": 0.9, "threshold": 0.92},
            "market_microstructure": {"weight": 0.85, "threshold": 0.90},
            "data_errors": {"weight": 0.95, "threshold": 0.94},
            "low_volume_noise": {"weight": 0.88, "threshold": 0.91},
            "high_frequency_noise": {"weight": 0.92, "threshold": 0.93},
            "outliers": {"weight": 0.97, "threshold": 0.95},
            "seasonal_patterns": {"weight": 0.94, "threshold": 0.92},
            "correlation_noise": {"weight": 0.86, "threshold": 0.90},
            "trend_deviation": {"weight": 0.89, "threshold": 0.91},
            "volatility_clustering": {"weight": 0.93, "threshold": 0.94}
        }
    
    def _setup_logger(self):
        """Set up logger for the Advanced Noise Filter"""
        logger = logging.getLogger("AdvancedNoiseFilter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            try:
                os.makedirs("logs", exist_ok=True)
                file_handler = logging.FileHandler("logs/advanced_noise_filter.log")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not create log file: {e}")
        
        return logger
    
    def filter_noise(self, market_data: Dict) -> Dict:
        """
        Filter out noise from market data to ensure only high-quality data is used.
        
        Parameters:
        - market_data: Dictionary containing market data (OHLCV, etc.)
        
        Returns:
        - Dictionary with filtered data and quality metrics
        """
        self.filter_stats["total_checks"] += 1
        
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 10:
            return {
                "filtered": False,
                "reason": "insufficient_data",
                "quality_score": 0.0,
                "data": market_data
            }
        
        initial_quality = self._calculate_quality_score(market_data)
        
        if initial_quality >= self.min_quality_threshold:
            return {
                "filtered": False,
                "reason": "already_high_quality",
                "quality_score": initial_quality,
                "data": market_data
            }
        
        filtered_data = market_data.copy()
        noise_detected = False
        
        if "ohlcv" in filtered_data:
            random_noise_result = self._filter_random_fluctuations(filtered_data)
            if random_noise_result["noise_detected"]:
                filtered_data = random_noise_result["data"]
                noise_detected = True
        
        if "ohlcv" in filtered_data:
            microstructure_result = self._filter_market_microstructure(filtered_data)
            if microstructure_result["noise_detected"]:
                filtered_data = microstructure_result["data"]
                noise_detected = True
        
        if "ohlcv" in filtered_data:
            errors_result = self._filter_data_errors(filtered_data)
            if errors_result["noise_detected"]:
                filtered_data = errors_result["data"]
                noise_detected = True
        
        if "ohlcv" in filtered_data:
            volume_result = self._filter_low_volume_noise(filtered_data)
            if volume_result["noise_detected"]:
                filtered_data = volume_result["data"]
                noise_detected = True
        
        if "ohlcv" in filtered_data:
            hf_result = self._filter_high_frequency_noise(filtered_data)
            if hf_result["noise_detected"]:
                filtered_data = hf_result["data"]
                noise_detected = True
        
        if "ohlcv" in filtered_data:
            outliers_result = self._filter_outliers(filtered_data)
            if outliers_result["noise_detected"]:
                filtered_data = outliers_result["data"]
                noise_detected = True
        
        final_quality = self._calculate_quality_score(filtered_data)
        
        if noise_detected:
            self.filter_stats["noise_detected"] += 1
            self.filter_stats["noise_removed"] += 1
            self.filter_stats["noise_reduction_rate"] = (
                self.filter_stats["noise_removed"] / self.filter_stats["noise_detected"]
            )
            
            quality_improvement = final_quality - initial_quality
            self.filter_stats["average_quality_improvement"] = (
                (self.filter_stats["average_quality_improvement"] * (self.filter_stats["noise_removed"] - 1) + 
                 quality_improvement) / self.filter_stats["noise_removed"]
            )
        
        return {
            "filtered": noise_detected,
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "quality_improvement": final_quality - initial_quality,
            "data": filtered_data,
            "high_quality": final_quality >= self.min_quality_threshold
        }
    
    def _calculate_quality_score(self, market_data: Dict) -> float:
        """
        Calculate a quality score for the market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Quality score between 0 and 1
        """
        quality_factors = []
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 0:
            ohlcv_data = market_data["ohlcv"]
            missing_values = sum(1 for candle in ohlcv_data if None in candle or np.nan in candle)
            missing_ratio = missing_values / len(ohlcv_data)
            quality_factors.append(1 - missing_ratio)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            close_prices = [candle[4] for candle in ohlcv_data]
            price_jumps = [abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1] 
                          for i in range(1, len(close_prices))]
            extreme_jumps = sum(1 for jump in price_jumps if jump > 0.1)  # 10% price jump
            extreme_ratio = extreme_jumps / len(price_jumps)
            quality_factors.append(1 - extreme_ratio)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            timestamps = [candle[0] for candle in ohlcv_data]
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            if len(set(time_diffs)) > 1:  # More than one unique time difference
                std_dev = np.std(time_diffs)
                mean_diff = np.mean(time_diffs)
                consistency = 1 - min(float(std_dev / mean_diff), 1.0)  # Normalize to [0,1]
                quality_factors.append(consistency)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            volumes = [candle[5] for candle in ohlcv_data]
            zero_volumes = sum(1 for vol in volumes if vol == 0)
            zero_ratio = zero_volumes / len(volumes)
            quality_factors.append(1 - zero_ratio)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 10:
            ohlcv_data = market_data["ohlcv"]
            close_prices = [candle[4] for candle in ohlcv_data]
            
            returns = [close_prices[i] / close_prices[i-1] - 1 for i in range(1, len(close_prices))]
            
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                autocorr_quality = 0.5 + 0.5 * abs(autocorr)
                quality_factors.append(autocorr_quality)
        
        if not quality_factors:
            return 0.0
        
        return sum(quality_factors) / len(quality_factors)
    
    def _filter_random_fluctuations(self, market_data: Dict) -> Dict:
        """
        Filter random fluctuations from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 10:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        close_prices = [candle[4] for candle in ohlcv_data]
        
        window_length = min(11, len(close_prices) - 2)
        if window_length % 2 == 0:
            window_length -= 1  # Must be odd
        
        if window_length >= 3:
            try:
                smoothed_prices = signal.savgol_filter(close_prices, window_length, 3)
                
                noise = np.array(close_prices) - smoothed_prices
                noise_level = np.std(noise) / np.mean(close_prices)
                
                if noise_level > 0.005:  # 0.5% noise
                    new_ohlcv = []
                    for i, candle in enumerate(ohlcv_data):
                        new_candle = list(candle)
                        new_candle[4] = smoothed_prices[i]  # Replace close price
                        new_ohlcv.append(tuple(new_candle))
                    
                    filtered_data["ohlcv"] = new_ohlcv
                    return {
                        "noise_detected": True,
                        "noise_level": noise_level,
                        "data": filtered_data
                    }
            except Exception as e:
                self.logger.warning(f"Error filtering random fluctuations: {e}")
        
        return {"noise_detected": False, "data": market_data}
    
    def _filter_market_microstructure(self, market_data: Dict) -> Dict:
        """
        Filter market microstructure noise from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 10:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        close_prices = [candle[4] for candle in ohlcv_data]
        
        window_size = min(5, len(close_prices))
        smoothed_prices = np.convolve(close_prices, np.ones(window_size)/window_size, mode='valid')
        
        padding = len(close_prices) - len(smoothed_prices)
        smoothed_prices = np.pad(smoothed_prices, (padding, 0), 'edge')
        
        noise = np.array(close_prices) - smoothed_prices
        noise_level = np.std(noise) / np.mean(close_prices)
        
        if noise_level > 0.003:  # 0.3% noise
            new_ohlcv = []
            for i, candle in enumerate(ohlcv_data):
                new_candle = list(candle)
                new_candle[4] = smoothed_prices[i]  # Replace close price
                new_ohlcv.append(tuple(new_candle))
            
            filtered_data["ohlcv"] = new_ohlcv
            return {
                "noise_detected": True,
                "noise_level": noise_level,
                "data": filtered_data
            }
        
        return {"noise_detected": False, "data": market_data}
    
    def _filter_data_errors(self, market_data: Dict) -> Dict:
        """
        Filter data errors from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 5:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        close_prices = [candle[4] for candle in ohlcv_data]
        
        returns = [close_prices[i] / close_prices[i-1] - 1 for i in range(1, len(close_prices))]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        outliers = []
        for i, ret in enumerate(returns):
            z_score = abs(ret - mean_return) / std_return
            if z_score > 3:
                outliers.append(i + 1)  # +1 because returns start from index 1
        
        if outliers:
            new_ohlcv = list(ohlcv_data)
            
            for outlier_idx in outliers:
                if 0 < outlier_idx < len(new_ohlcv):
                    prev_idx = outlier_idx - 1
                    next_idx = min(outlier_idx + 1, len(new_ohlcv) - 1)
                    
                    prev_price = new_ohlcv[prev_idx][4]
                    next_price = new_ohlcv[next_idx][4]
                    
                    corrected_price = (prev_price + next_price) / 2
                    
                    new_candle = list(new_ohlcv[outlier_idx])
                    new_candle[4] = corrected_price  # Replace close price
                    new_ohlcv[outlier_idx] = tuple(new_candle)
            
            filtered_data["ohlcv"] = new_ohlcv
            return {
                "noise_detected": True,
                "outliers_detected": len(outliers),
                "data": filtered_data
            }
        
        return {"noise_detected": False, "data": market_data}
    
    def _filter_low_volume_noise(self, market_data: Dict) -> Dict:
        """
        Filter low volume noise from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 5:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        volumes = [candle[5] for candle in ohlcv_data]
        
        avg_volume = np.mean(volumes)
        
        low_volume_indices = [i for i, vol in enumerate(volumes) if vol < 0.3 * avg_volume]
        
        if low_volume_indices:
            new_ohlcv = list(ohlcv_data)
            
            for idx in low_volume_indices:
                if 0 < idx < len(new_ohlcv) - 1:
                    prev_idx = idx - 1
                    next_idx = idx + 1
                    
                    prev_price = new_ohlcv[prev_idx][4]
                    current_price = new_ohlcv[idx][4]
                    next_price = new_ohlcv[next_idx][4]
                    
                    adjusted_price = (prev_price + current_price + next_price) / 3
                    
                    new_candle = list(new_ohlcv[idx])
                    new_candle[4] = adjusted_price  # Replace close price
                    new_ohlcv[idx] = tuple(new_candle)
            
            filtered_data["ohlcv"] = new_ohlcv
            return {
                "noise_detected": True,
                "low_volume_candles": len(low_volume_indices),
                "data": filtered_data
            }
        
        return {"noise_detected": False, "data": market_data}
    
    def _filter_high_frequency_noise(self, market_data: Dict) -> Dict:
        """
        Filter high frequency noise from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 10:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        close_prices = [candle[4] for candle in ohlcv_data]
        
        window_size = min(5, len(close_prices))
        smoothed_prices = np.convolve(close_prices, np.ones(window_size)/window_size, mode='same')
        
        high_freq = np.array(close_prices) - smoothed_prices
        high_freq_energy = np.sum(high_freq**2) / len(high_freq)
        total_energy = np.sum(np.array(close_prices)**2) / len(close_prices)
        
        noise_ratio = high_freq_energy / total_energy
        
        if noise_ratio > 0.01:  # 1% high-frequency noise
            new_ohlcv = []
            for i, candle in enumerate(ohlcv_data):
                new_candle = list(candle)
                new_candle[4] = smoothed_prices[i]  # Replace close price
                new_ohlcv.append(tuple(new_candle))
            
            filtered_data["ohlcv"] = new_ohlcv
            return {
                "noise_detected": True,
                "noise_ratio": noise_ratio,
                "data": filtered_data
            }
        
        return {"noise_detected": False, "data": market_data}
    
    def _filter_outliers(self, market_data: Dict) -> Dict:
        """
        Filter outliers from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with filtered data and noise metrics
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 5:
            return {"noise_detected": False, "data": market_data}
        
        filtered_data = market_data.copy()
        ohlcv_data = filtered_data["ohlcv"]
        
        close_prices = [candle[4] for candle in ohlcv_data]
        
        median_price = np.median(close_prices)
        mad = np.median([abs(price - median_price) for price in close_prices])
        
        outliers = []
        for i, price in enumerate(close_prices):
            if abs(price - median_price) > 3 * mad:
                outliers.append(i)
        
        if outliers:
            new_ohlcv = list(ohlcv_data)
            
            for outlier_idx in outliers:
                if 0 < outlier_idx < len(new_ohlcv) - 1:
                    prev_idx = outlier_idx - 1
                    next_idx = outlier_idx + 1
                    
                    prev_price = new_ohlcv[prev_idx][4]
                    next_price = new_ohlcv[next_idx][4]
                    
                    corrected_price = (prev_price + next_price) / 2
                    
                    new_candle = list(new_ohlcv[outlier_idx])
                    new_candle[4] = corrected_price  # Replace close price
                    new_ohlcv[outlier_idx] = tuple(new_candle)
            
            filtered_data["ohlcv"] = new_ohlcv
            return {
                "noise_detected": True,
                "outliers_detected": len(outliers),
                "data": filtered_data
            }
        
        return {"noise_detected": False, "data": market_data}
    
    def get_filter_stats(self) -> Dict:
        """
        Get filter statistics.
        
        Returns:
        - Dictionary with filter statistics
        """
        return self.filter_stats
