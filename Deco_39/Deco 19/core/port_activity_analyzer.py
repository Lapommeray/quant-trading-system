"""
Port Activity Analyzer

Python wrapper for the port activity analysis module that uses audio samples
to detect patterns in shipping container movements. This module can predict
supply chain disruptions based on port activity patterns.

Original implementation uses:
- librosa for audio analysis
- numpy for signal processing
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PortActivityAnalyzer:
    def __init__(self, algorithm):
        self.algo = algorithm
        
        self.spectral_threshold = 11.0  # Hz
        
        self.monitored_ports = {
            'SHANGHAI': {'importance': 1.0, 'threshold': 11.0},
            'SINGAPORE': {'importance': 0.9, 'threshold': 10.5},
            'ROTTERDAM': {'importance': 0.8, 'threshold': 10.0},
            'LOS_ANGELES': {'importance': 0.7, 'threshold': 9.5},
            'BUSAN': {'importance': 0.6, 'threshold': 9.0}
        }
        
        self.detection_history = {}
        self.max_history_size = 30
        
        self.last_check_time = {}
        self.check_interval = timedelta(hours=12)  # Check twice daily
        
        self.algo.Debug("PortActivityAnalyzer initialized")
    
    def analyze(self, current_time=None):
        """
        Analyze port activity for supply chain disruption signals
        
        Parameters:
        - current_time: Current algorithm time (optional)
        
        Returns:
        - Dictionary with analysis results
        """
        if current_time is None:
            current_time = self.algo.Time
            
        should_analyze = False
        for port in self.monitored_ports:
            if port not in self.last_check_time or current_time - self.last_check_time[port] >= self.check_interval:
                should_analyze = True
                self.last_check_time[port] = current_time
                
        if not should_analyze:
            return None
            
        port_results = {}
        overall_disruption = False
        highest_confidence = 0.0
        
        for port, info in self.monitored_ports.items():
            spectral_value = self._simulate_spectral_data(port, current_time)
            
            disruption_detected = spectral_value > info['threshold']
            
            if disruption_detected:
                confidence = min(info['importance'] * (spectral_value - info['threshold']) / 5 + 0.5, 1.0)
            else:
                confidence = 0.0
                
            if disruption_detected and confidence > highest_confidence:
                highest_confidence = confidence
                overall_disruption = True
                
            if port not in self.detection_history:
                self.detection_history[port] = []
                
            self.detection_history[port].append({
                'time': current_time,
                'spectral_value': spectral_value,
                'disruption_detected': disruption_detected,
                'confidence': confidence
            })
            
            if len(self.detection_history[port]) > self.max_history_size:
                self.detection_history[port] = self.detection_history[port][-self.max_history_size:]
                
            port_results[port] = {
                'disruption_detected': disruption_detected,
                'spectral_value': spectral_value,
                'confidence': confidence
            }
            
            if disruption_detected:
                self.algo.Debug(f"PortActivityAnalyzer: Disruption detected at {port}! " +
                              f"Spectral value: {spectral_value:.2f}, Confidence: {confidence:.2f}")
        
        return {
            'disruption_detected': overall_disruption,
            'port_results': port_results,
            'confidence': highest_confidence,
            'direction': 'BUY' if overall_disruption else None  # Supply chain disruptions often lead to price increases
        }
    
    def _simulate_spectral_data(self, port, current_time):
        """
        Simulate spectral analysis data for testing
        In production, this would be replaced with actual audio analysis
        
        Parameters:
        - port: Port name
        - current_time: Current algorithm time
        
        Returns:
        - Simulated spectral centroid value
        """
        port_hash = sum(ord(c) for c in port) % 100
        day_of_month = current_time.day
        
        base_value = 8.0 + (port_hash / 50)  # 8.0 to 10.0
        
        time_factor = (day_of_month % 7) / 7 * 2  # 0 to 2
        
        random_factor = np.random.normal(0, 0.5)
        
        if np.random.random() < 0.08:  # 8% chance of spike
            spike_factor = np.random.uniform(1.5, 3.0)
        else:
            spike_factor = 0.0
            
        value = base_value + time_factor + random_factor + spike_factor
        
        return value
    
    def get_port_status_summary(self):
        """
        Generate a summary of current port status
        
        Returns:
        - Dictionary with port status summary
        """
        summary = {}
        
        for port in self.monitored_ports:
            if port in self.detection_history and self.detection_history[port]:
                latest = self.detection_history[port][-1]
                
                recent = self.detection_history[port][-5:]
                if len(recent) >= 3:
                    spectral_values = [entry['spectral_value'] for entry in recent]
                    trend = np.polyfit(range(len(spectral_values)), spectral_values, 1)[0]
                    trend_direction = "increasing" if trend > 0.1 else ("decreasing" if trend < -0.1 else "stable")
                else:
                    trend_direction = "unknown"
                
                summary[port] = {
                    'current_value': latest['spectral_value'],
                    'disruption_status': latest['disruption_detected'],
                    'trend': trend_direction,
                    'last_updated': latest['time']
                }
        
        return summary
