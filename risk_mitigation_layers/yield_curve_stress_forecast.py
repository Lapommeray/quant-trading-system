"""
Yield Curve Stress Forecast

Exits bonds if inversion lasts > 5 days for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class YieldCurveStressForecast:
    """
    Exits bonds if inversion lasts > 5 days.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Yield Curve Stress Forecast.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("YieldCurveStressForecast")
        self.logger.setLevel(logging.INFO)
        
        self.inversion_threshold = 0.0  # Flat or inverted
        self.deep_inversion_threshold = -0.1  # 10bps inversion
        self.severe_inversion_threshold = -0.25  # 25bps inversion
        
        self.warning_duration = 3
        self.exit_duration = 5
        self.severe_duration = 10
        
        self.stress_levels = {
            "normal": 0.0,
            "elevated": 0.25,
            "high": 0.5,
            "severe": 0.75,
            "extreme": 1.0
        }
        
        self.current_stress_level = "normal"
        self.current_stress_score = self.stress_levels[self.current_stress_level]
        
        self.yield_curve_data = {
            "2y": 0.0,
            "5y": 0.0,
            "10y": 0.0,
            "30y": 0.0,
            "10y2y_spread": 0.0,
            "30y5y_spread": 0.0,
            "inversion_days": 0,
            "deep_inversion_days": 0,
            "severe_inversion_days": 0,
            "historical": []
        }
        
        self.stress_forecast = {
            "current_level": self.current_stress_level,
            "current_score": self.current_stress_score,
            "forecast_1w": 0.0,
            "forecast_2w": 0.0,
            "forecast_1m": 0.0,
            "exit_signals": {}
        }
        
        self.stress_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=6)  # Less frequent updates for yield curve data
        
    def update(self, current_time, yield_data=None):
        """
        Update the yield curve stress forecast with latest data.
        
        Parameters:
        - current_time: Current datetime
        - yield_data: Yield curve data (optional)
        
        Returns:
        - Dictionary containing stress forecast results
        """
        if current_time - self.last_update < self.update_frequency and yield_data is None:
            return {
                "stress_forecast": self.stress_forecast,
                "yield_curve_data": self.yield_curve_data
            }
            
        if yield_data is not None:
            self._update_yield_curve_data(yield_data)
        else:
            self._update_yield_curve_data_internal()
        
        self._calculate_stress_level()
        
        self._generate_stress_forecast()
        
        self.last_update = current_time
        
        return {
            "stress_forecast": self.stress_forecast,
            "yield_curve_data": self.yield_curve_data
        }
        
    def _update_yield_curve_data(self, yield_data):
        """
        Update yield curve data.
        
        Parameters:
        - yield_data: Yield curve data
        """
        for key, value in yield_data.items():
            if key in self.yield_curve_data:
                self.yield_curve_data[key] = value
        
        self._calculate_spreads()
        
    def _update_yield_curve_data_internal(self):
        """
        Update yield curve data internally.
        """
        
        self.yield_curve_data["2y"] = 4.8
        self.yield_curve_data["5y"] = 4.5
        self.yield_curve_data["10y"] = 4.3
        self.yield_curve_data["30y"] = 4.4
        
        self._calculate_spreads()
        
        self.yield_curve_data["historical"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "2y": self.yield_curve_data["2y"],
            "5y": self.yield_curve_data["5y"],
            "10y": self.yield_curve_data["10y"],
            "30y": self.yield_curve_data["30y"],
            "10y2y_spread": self.yield_curve_data["10y2y_spread"],
            "30y5y_spread": self.yield_curve_data["30y5y_spread"]
        })
        
        if len(self.yield_curve_data["historical"]) > 30:
            self.yield_curve_data["historical"] = self.yield_curve_data["historical"][-30:]
        
    def _calculate_spreads(self):
        """
        Calculate yield curve spreads.
        """
        self.yield_curve_data["10y2y_spread"] = self.yield_curve_data["10y"] - self.yield_curve_data["2y"]
        
        self.yield_curve_data["30y5y_spread"] = self.yield_curve_data["30y"] - self.yield_curve_data["5y"]
        
        if self.yield_curve_data["10y2y_spread"] <= self.inversion_threshold:
            self.yield_curve_data["inversion_days"] += 1
        else:
            self.yield_curve_data["inversion_days"] = 0
        
        if self.yield_curve_data["10y2y_spread"] <= self.deep_inversion_threshold:
            self.yield_curve_data["deep_inversion_days"] += 1
        else:
            self.yield_curve_data["deep_inversion_days"] = 0
        
        if self.yield_curve_data["10y2y_spread"] <= self.severe_inversion_threshold:
            self.yield_curve_data["severe_inversion_days"] += 1
        else:
            self.yield_curve_data["severe_inversion_days"] = 0
        
    def _calculate_stress_level(self):
        """
        Calculate stress level.
        """
        spread_10y2y = self.yield_curve_data["10y2y_spread"]
        spread_30y5y = self.yield_curve_data["30y5y_spread"]
        inversion_days = self.yield_curve_data["inversion_days"]
        deep_inversion_days = self.yield_curve_data["deep_inversion_days"]
        severe_inversion_days = self.yield_curve_data["severe_inversion_days"]
        
        if severe_inversion_days >= self.severe_duration:
            new_stress_level = "extreme"
        elif deep_inversion_days >= self.exit_duration:
            new_stress_level = "severe"
        elif inversion_days >= self.exit_duration:
            new_stress_level = "high"
        elif inversion_days >= self.warning_duration:
            new_stress_level = "elevated"
        else:
            new_stress_level = "normal"
        
        if spread_10y2y <= self.inversion_threshold and spread_30y5y <= self.inversion_threshold:
            stress_levels = list(self.stress_levels.keys())
            current_index = stress_levels.index(new_stress_level)
            if current_index < len(stress_levels) - 1:
                new_stress_level = stress_levels[current_index + 1]
        
        if new_stress_level != self.current_stress_level:
            self.stress_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "from_level": self.current_stress_level,
                "to_level": new_stress_level,
                "10y2y_spread": spread_10y2y,
                "30y5y_spread": spread_30y5y,
                "inversion_days": inversion_days
            })
            
            self.current_stress_level = new_stress_level
            self.current_stress_score = self.stress_levels[self.current_stress_level]
            
            self.logger.info(f"Stress level changed from {self.stress_history[-1]['from_level']} to {self.current_stress_level}")
            self.logger.info(f"Stress score: {self.current_stress_score}")
        
    def _generate_stress_forecast(self):
        """
        Generate stress forecast.
        """
        spread_10y2y = self.yield_curve_data["10y2y_spread"]
        inversion_days = self.yield_curve_data["inversion_days"]
        
        trend = 0.0
        if len(self.yield_curve_data["historical"]) >= 5:
            recent_spreads = [data["10y2y_spread"] for data in self.yield_curve_data["historical"][-5:]]
            if len(recent_spreads) >= 2:
                trend = recent_spreads[-1] - recent_spreads[0]
        
        forecast_1w = self.current_stress_score
        forecast_2w = self.current_stress_score
        forecast_1m = self.current_stress_score
        
        if trend < -0.05:  # Spread narrowing/inverting
            forecast_1w = min(1.0, forecast_1w + 0.1)
            forecast_2w = min(1.0, forecast_2w + 0.2)
            forecast_1m = min(1.0, forecast_1m + 0.3)
        elif trend > 0.05:  # Spread widening
            forecast_1w = max(0.0, forecast_1w - 0.1)
            forecast_2w = max(0.0, forecast_2w - 0.2)
            forecast_1m = max(0.0, forecast_1m - 0.3)
        
        exit_signals = {}
        
        if inversion_days >= self.exit_duration:
            exit_signals["bonds"] = {
                "action": "EXIT",
                "reason": f"Yield curve inversion for {inversion_days} days",
                "severity": self.current_stress_score
            }
        elif inversion_days >= self.warning_duration:
            exit_signals["bonds"] = {
                "action": "REDUCE",
                "reason": f"Yield curve inversion for {inversion_days} days",
                "severity": self.current_stress_score
            }
        
        if spread_10y2y < 0:
            exit_signals["long_duration_bonds"] = {
                "action": "EXIT",
                "reason": "Negative 10y-2y spread",
                "severity": self.current_stress_score
            }
        
        self.stress_forecast = {
            "current_level": self.current_stress_level,
            "current_score": self.current_stress_score,
            "forecast_1w": forecast_1w,
            "forecast_2w": forecast_2w,
            "forecast_1m": forecast_1m,
            "exit_signals": exit_signals
        }
        
    def get_current_stress_level(self):
        """
        Get current stress level.
        
        Returns:
        - Current stress level
        """
        return self.current_stress_level
        
    def get_current_stress_score(self):
        """
        Get current stress score.
        
        Returns:
        - Current stress score
        """
        return self.current_stress_score
        
    def get_stress_forecast(self):
        """
        Get stress forecast.
        
        Returns:
        - Stress forecast
        """
        return self.stress_forecast
        
    def get_exit_signals(self):
        """
        Get exit signals.
        
        Returns:
        - Exit signals
        """
        return self.stress_forecast["exit_signals"]
        
    def get_stress_history(self):
        """
        Get stress history.
        
        Returns:
        - Stress history
        """
        return self.stress_history
        
    def calculate_position_size(self, base_size, asset_type=None, duration=None):
        """
        Calculate position size based on stress forecast.
        
        Parameters:
        - base_size: Base position size
        - asset_type: Asset type (optional)
        - duration: Bond duration in years (optional)
        
        Returns:
        - Adjusted position size
        """
        stress_score = self.current_stress_score
        
        multiplier = 1.0 - stress_score
        
        if asset_type == "bonds":
            if "bonds" in self.stress_forecast["exit_signals"]:
                exit_signal = self.stress_forecast["exit_signals"]["bonds"]
                if exit_signal["action"] == "EXIT":
                    multiplier = 0.0
                elif exit_signal["action"] == "REDUCE":
                    multiplier = 0.5
            
            if duration is not None:
                duration_factor = min(1.0, duration / 10.0)
                multiplier *= (1.0 - duration_factor * stress_score)
        
        adjusted_size = base_size * multiplier
        
        return adjusted_size
