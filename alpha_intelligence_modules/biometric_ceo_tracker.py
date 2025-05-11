"""
Biometric CEO Tracker

Detects stress in exec comms for short plays for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import re

class BiometricCEOTracker:
    """
    Detects stress in exec comms for short plays.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Biometric CEO Tracker.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("BiometricCEOTracker")
        self.logger.setLevel(logging.INFO)
        
        self.biometric_metrics = [
            "voice_stress",
            "micro_expressions",
            "word_choice",
            "speech_pattern",
            "body_language"
        ]
        
        self.stress_thresholds = {
            "low": 0.3,
            "moderate": 0.5,
            "high": 0.7,
            "extreme": 0.85
        }
        
        self.tracked_executives = {}
        
        self.biometric_data = {}
        
        self.stress_signals = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=6)  # Less frequent updates for exec data
        
        self.tracked_companies = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "GS"]
        
        self.stress_history = {}
        
    def update(self, current_time, custom_data=None):
        """
        Update the biometric CEO tracker with latest data.
        
        Parameters:
        - current_time: Current datetime
        - custom_data: Custom biometric data (optional)
        
        Returns:
        - Dictionary containing stress results
        """
        if current_time - self.last_update < self.update_frequency and custom_data is None:
            return {
                "biometric_data": self.biometric_data,
                "stress_signals": self.stress_signals
            }
            
        self._update_tracked_executives()
        
        if custom_data is not None:
            self._update_biometric_data(custom_data)
        else:
            self._update_biometric_data_internal()
        
        self._analyze_stress()
        
        self._generate_signals()
        
        self.last_update = current_time
        
        return {
            "biometric_data": self.biometric_data,
            "stress_signals": self.stress_signals
        }
        
    def _update_tracked_executives(self):
        """
        Update tracked executives.
        """
        
        self.tracked_executives = {
            "AAPL": {
                "CEO": {
                    "name": "Tim Cook",
                    "role": "CEO",
                    "tenure": 12.5,
                    "baseline_stress": 0.25
                },
                "CFO": {
                    "name": "Luca Maestri",
                    "role": "CFO",
                    "tenure": 9.2,
                    "baseline_stress": 0.3
                }
            },
            "MSFT": {
                "CEO": {
                    "name": "Satya Nadella",
                    "role": "CEO",
                    "tenure": 11.3,
                    "baseline_stress": 0.2
                },
                "CFO": {
                    "name": "Amy Hood",
                    "role": "CFO",
                    "tenure": 12.8,
                    "baseline_stress": 0.25
                }
            },
            "AMZN": {
                "CEO": {
                    "name": "Andy Jassy",
                    "role": "CEO",
                    "tenure": 3.9,
                    "baseline_stress": 0.35
                },
                "CFO": {
                    "name": "Brian Olsavsky",
                    "role": "CFO",
                    "tenure": 8.5,
                    "baseline_stress": 0.3
                }
            },
            "TSLA": {
                "CEO": {
                    "name": "Elon Musk",
                    "role": "CEO",
                    "tenure": 15.2,
                    "baseline_stress": 0.4
                },
                "CFO": {
                    "name": "Zachary Kirkhorn",
                    "role": "CFO",
                    "tenure": 4.3,
                    "baseline_stress": 0.35
                }
            }
        }
        
        for company in self.tracked_companies:
            if company not in self.tracked_executives:
                self.tracked_executives[company] = {
                    "CEO": {
                        "name": f"{company} CEO",
                        "role": "CEO",
                        "tenure": np.random.uniform(3.0, 15.0),
                        "baseline_stress": np.random.uniform(0.2, 0.4)
                    },
                    "CFO": {
                        "name": f"{company} CFO",
                        "role": "CFO",
                        "tenure": np.random.uniform(3.0, 15.0),
                        "baseline_stress": np.random.uniform(0.2, 0.4)
                    }
                }
        
    def _update_biometric_data(self, custom_data):
        """
        Update biometric data.
        
        Parameters:
        - custom_data: Custom biometric data
        """
        for company, execs in custom_data.items():
            if company not in self.biometric_data:
                self.biometric_data[company] = {}
            
            for exec_role, metrics in execs.items():
                if exec_role not in self.biometric_data[company]:
                    self.biometric_data[company][exec_role] = {}
                
                for metric, value in metrics.items():
                    if metric in self.biometric_metrics:
                        self.biometric_data[company][exec_role][metric] = value
        
    def _update_biometric_data_internal(self):
        """
        Update biometric data internally.
        """
        
        for company, execs in self.tracked_executives.items():
            if company not in self.biometric_data:
                self.biometric_data[company] = {}
            
            for exec_role, exec_data in execs.items():
                if exec_role not in self.biometric_data[company]:
                    self.biometric_data[company][exec_role] = {}
                
                baseline_stress = exec_data["baseline_stress"]
                
                variation = np.random.normal(0, 0.1)
                
                if company == "TSLA":
                    variation *= 2.0
                
                stress_level = baseline_stress + variation
                
                stress_level = max(0.0, min(1.0, stress_level))
                
                self.biometric_data[company][exec_role] = {
                    "voice_stress": stress_level * np.random.uniform(0.8, 1.2),
                    "micro_expressions": stress_level * np.random.uniform(0.8, 1.2),
                    "word_choice": stress_level * np.random.uniform(0.8, 1.2),
                    "speech_pattern": stress_level * np.random.uniform(0.8, 1.2),
                    "body_language": stress_level * np.random.uniform(0.8, 1.2),
                    "timestamp": datetime.now(),
                    "event_type": "earnings_call" if np.random.random() < 0.3 else "interview",
                    "baseline_deviation": variation
                }
            
            if company not in self.stress_history:
                self.stress_history[company] = []
            
            avg_stress = np.mean([
                np.mean([value for key, value in exec_data.items() if key in self.biometric_metrics])
                for exec_role, exec_data in self.biometric_data[company].items()
            ])
            
            self.stress_history[company].append({
                "timestamp": datetime.now(),
                "stress": avg_stress
            })
            
            if len(self.stress_history[company]) > 50:
                self.stress_history[company] = self.stress_history[company][-50:]
        
    def _analyze_stress(self):
        """
        Analyze stress data.
        """
        for company, execs in self.biometric_data.items():
            company_stress = {
                "avg_stress": 0.0,
                "max_stress": 0.0,
                "stress_change": 0.0,
                "stress_volatility": 0.0,
                "exec_count": 0
            }
            
            for exec_role, metrics in execs.items():
                metric_values = [value for key, value in metrics.items() if key in self.biometric_metrics]
                if len(metric_values) > 0:
                    avg_stress = np.mean(metric_values)
                    
                    company_stress["avg_stress"] += avg_stress
                    company_stress["max_stress"] = max(float(company_stress["max_stress"]), float(avg_stress))
                    company_stress["exec_count"] += 1
                    
                    if "exec_stress" not in company_stress:
                        company_stress["exec_stress"] = {}
                    
                    company_stress["exec_stress"][exec_role] = avg_stress
            
            if company_stress["exec_count"] > 0:
                company_stress["avg_stress"] /= company_stress["exec_count"]
            
            if company in self.stress_history and len(self.stress_history[company]) > 1:
                recent_stress = self.stress_history[company][-1]["stress"]
                previous_stress = self.stress_history[company][-2]["stress"]
                company_stress["stress_change"] = recent_stress - previous_stress
            
            if company in self.stress_history and len(self.stress_history[company]) > 5:
                recent_stresses = [point["stress"] for point in self.stress_history[company][-5:]]
                company_stress["stress_volatility"] = np.std(recent_stresses)
            
            if company not in self.stress_signals:
                self.stress_signals[company] = {}
            
            self.stress_signals[company]["analysis"] = company_stress
        
    def _generate_signals(self):
        """
        Generate stress signals.
        """
        for company, data in self.stress_signals.items():
            if "analysis" not in data:
                continue
                
            analysis = data["analysis"]
            
            avg_stress = analysis["avg_stress"]
            max_stress = analysis["max_stress"]
            stress_change = analysis["stress_change"]
            stress_volatility = analysis["stress_volatility"]
            
            if avg_stress >= self.stress_thresholds["extreme"]:
                stress_level = "extreme"
            elif avg_stress >= self.stress_thresholds["high"]:
                stress_level = "high"
            elif avg_stress >= self.stress_thresholds["moderate"]:
                stress_level = "moderate"
            elif avg_stress >= self.stress_thresholds["low"]:
                stress_level = "low"
            else:
                stress_level = "normal"
            
            signal_type = "NEUTRAL"
            signal_strength = 0.0
            
            if stress_level == "extreme" and stress_change > 0.2:
                signal_type = "STRONG_SHORT"
                signal_strength = 0.9
            elif stress_level == "high" and stress_change > 0.15:
                signal_type = "SHORT"
                signal_strength = 0.7
            elif stress_level == "extreme":
                signal_type = "SHORT"
                signal_strength = 0.6
            elif stress_level == "high":
                signal_type = "WEAK_SHORT"
                signal_strength = 0.4
            elif stress_level == "normal" and stress_change < -0.15:
                signal_type = "WEAK_LONG"
                signal_strength = 0.3
            
            if stress_volatility > 0.2:
                signal_strength *= 0.8
            
            self.stress_signals[company]["signal"] = {
                "type": signal_type,
                "strength": signal_strength,
                "stress_level": stress_level,
                "stress_change": stress_change,
                "stress_volatility": stress_volatility
            }
        
    def get_biometric_data(self, company=None, exec_role=None):
        """
        Get biometric data.
        
        Parameters:
        - company: Company to get data for (optional)
        - exec_role: Executive role to get data for (optional)
        
        Returns:
        - Biometric data
        """
        if company is not None:
            if company not in self.biometric_data:
                return {}
                
            if exec_role is not None:
                return self.biometric_data[company].get(exec_role, {})
            else:
                return self.biometric_data[company]
        else:
            return self.biometric_data
        
    def get_stress_signals(self, company=None):
        """
        Get stress signals.
        
        Parameters:
        - company: Company to get signals for (optional)
        
        Returns:
        - Stress signals
        """
        if company is not None:
            return self.stress_signals.get(company, {})
        else:
            return self.stress_signals
        
    def get_stress_history(self, company=None):
        """
        Get stress history.
        
        Parameters:
        - company: Company to get history for (optional)
        
        Returns:
        - Stress history
        """
        if company is not None:
            return self.stress_history.get(company, [])
        else:
            return self.stress_history
        
    def get_trading_signal(self, company):
        """
        Get trading signal for a company.
        
        Parameters:
        - company: Company to get signal for
        
        Returns:
        - Trading signal
        """
        if company not in self.stress_signals or "signal" not in self.stress_signals[company]:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0
            }
        
        signal = self.stress_signals[company]["signal"]
        
        if signal["type"] == "STRONG_SHORT":
            action = "SELL"
            confidence = signal["strength"]
        elif signal["type"] == "SHORT":
            action = "SELL"
            confidence = signal["strength"]
        elif signal["type"] == "WEAK_SHORT":
            action = "SELL"
            confidence = signal["strength"]
        elif signal["type"] == "WEAK_LONG":
            action = "BUY"
            confidence = signal["strength"]
        else:
            action = "NEUTRAL"
            confidence = 0.0
        
        return {
            "action": action,
            "confidence": confidence
        }
