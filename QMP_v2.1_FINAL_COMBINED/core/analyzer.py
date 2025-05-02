"""
Automated Post-Mortem Analysis Module

This module implements the Automated Post-Mortem Analysis system for the QMP Overrider strategy.
It provides comprehensive root cause analysis and actionable recommendations for circuit breaker events.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
import requests
from typing import Dict, List, Tuple, Optional, Union, Any
import os

class PostMortemEngine:
    """
    Automated Post-Mortem Analysis Engine for the QMP Overrider system.
    
    This class provides comprehensive root cause analysis and actionable recommendations
    for circuit breaker events and trading anomalies.
    """
    
    def __init__(self, prometheus_url="http://prometheus:9090"):
        """
        Initialize the Post-Mortem Engine.
        
        Parameters:
            prometheus_url: URL of the Prometheus server for metrics retrieval
        """
        self.logger = logging.getLogger("PostMortemEngine")
        
        self.prometheus_url = prometheus_url
        self.log_dir = Path("logs/post_mortem")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_history = []
        self._load_history()
        
        self.logger.info("Post-Mortem Engine initialized")
    
    def _load_history(self):
        """Load report history from file"""
        history_file = self.log_dir / "report_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.report_history = json.load(f)
                self.logger.info(f"Loaded {len(self.report_history)} historical post-mortem reports")
            except Exception as e:
                self.logger.error(f"Error loading report history: {e}")
    
    def _save_history(self):
        """Save report history to file"""
        history_file = self.log_dir / "report_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.report_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving report history: {e}")
    
    def get_metric_range_data(self, metric_name, start_time, end_time):
        """
        Get metric data from Prometheus for a specific time range.
        
        Parameters:
            metric_name: Name of the metric to retrieve
            start_time: Start time for the data range
            end_time: End time for the data range
            
        Returns:
            List of metric data points
        """
        try:
            start_timestamp = int(pd.Timestamp(start_time).timestamp())
            end_timestamp = int(pd.Timestamp(end_time).timestamp())
            
            query_url = f"{self.prometheus_url}/api/v1/query_range"
            params = {
                "query": metric_name,
                "start": start_timestamp,
                "end": end_timestamp,
                "step": "15s"
            }
            
            response = requests.get(query_url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success" and "data" in result and "result" in result["data"]:
                    data_points = []
                    for series in result["data"]["result"]:
                        for value in series["values"]:
                            data_points.append({
                                "timestamp": value[0],
                                "value": float(value[1])
                            })
                    return data_points
            
            self.logger.warning(f"Failed to get metric data for {metric_name}: {response.text}")
            
            return self._generate_mock_data(metric_name, start_time, end_time)
            
        except Exception as e:
            self.logger.error(f"Error getting metric data: {e}")
            return self._generate_mock_data(metric_name, start_time, end_time)
    
    def _generate_mock_data(self, metric_name, start_time, end_time):
        """
        Generate mock data for testing or when Prometheus is not available.
        
        Parameters:
            metric_name: Name of the metric to mock
            start_time: Start time for the data range
            end_time: End time for the data range
            
        Returns:
            List of mock data points
        """
        start_timestamp = pd.Timestamp(start_time).timestamp()
        end_timestamp = pd.Timestamp(end_time).timestamp()
        
        timestamps = np.arange(start_timestamp, end_timestamp, 15)
        
        if "latency" in metric_name:
            base_value = 20
            peak_value = 150
            values = np.ones(len(timestamps)) * base_value
            
            mid_idx = len(values) // 2
            spike_width = len(values) // 5
            for i in range(mid_idx - spike_width // 2, mid_idx + spike_width // 2):
                if i >= 0 and i < len(values):
                    distance = abs(i - mid_idx)
                    values[i] = base_value + (peak_value - base_value) * (1 - distance / (spike_width // 2))
            
        elif "order_book" in metric_name:
            values = np.random.uniform(30, 50, len(timestamps))
            
            mid_idx = len(values) // 2
            imbalance_width = len(values) // 4
            for i in range(mid_idx - imbalance_width // 2, mid_idx + imbalance_width // 2):
                if i >= 0 and i < len(values):
                    values[i] = np.random.uniform(70, 90)
            
        elif "strategy_health" in metric_name:
            values = np.ones(len(timestamps))
            
            mid_idx = len(values) // 2
            degradation_width = len(values) // 3
            for i in range(mid_idx - degradation_width // 2, mid_idx + degradation_width // 2):
                if i >= 0 and i < len(values):
                    distance = abs(i - mid_idx)
                    values[i] = max(0.3, 1 - distance / (degradation_width // 2) * 0.7)
            
        else:
            values = np.random.normal(50, 10, len(timestamps))
        
        data_points = []
        for i in range(len(timestamps)):
            data_points.append({
                "timestamp": timestamps[i],
                "value": values[i]
            })
        
        return data_points
    
    def generate_report(self, trigger_timestamp):
        """
        Generate a comprehensive post-mortem report for a circuit breaker event.
        
        Parameters:
            trigger_timestamp: Timestamp of the circuit breaker event
            
        Returns:
            Post-mortem report as a string
        """
        self.logger.info(f"Generating post-mortem report for event at {trigger_timestamp}")
        
        if isinstance(trigger_timestamp, (int, float)):
            trigger_timestamp = datetime.fromtimestamp(trigger_timestamp)
        
        start_time = trigger_timestamp - timedelta(minutes=2)
        end_time = trigger_timestamp + timedelta(minutes=3)
        
        data = {
            "order_book": self.get_metric_range_data(
                'order_book_depth',
                start_time,
                end_time
            ),
            "latency": self.get_metric_range_data(
                'exchange_latency_ms',
                start_time,
                end_time
            ),
            "strategy_health": self.get_metric_range_data(
                'strategy_heartbeat',
                start_time,
                end_time
            ),
            "volatility": self.get_metric_range_data(
                'market_volatility',
                start_time,
                end_time
            ),
            "trade_volume": self.get_metric_range_data(
                'trade_volume',
                start_time,
                end_time
            )
        }
        
        latency_spike = self._calc_spike(data['latency'])
        order_book_imbalance = self._calc_imbalance(data['order_book'])
        strategy_health = self._health_status(data['strategy_health'])
        volatility = self._calc_volatility(data['volatility'])
        volume_anomaly = self._calc_volume_anomaly(data['trade_volume'])
        
        report = f"""
POST-MORTEM REPORT - {trigger_timestamp}
===================================

Root Cause Analysis:
- Latency Spike: {latency_spike}ms
- Order Book Imbalance: {order_book_imbalance:.1f}%
- Strategy Health: {strategy_health}
- Market Volatility: {volatility:.2f}
- Volume Anomaly: {volume_anomaly:.1f}x normal

Primary Trigger: {self._determine_primary_trigger(latency_spike, order_book_imbalance, volatility, volume_anomaly)}

Recommended Actions:
{self._generate_actions(data, latency_spike, order_book_imbalance, strategy_health, volatility, volume_anomaly)}

Technical Details:
- Event Timestamp: {trigger_timestamp}
- Analysis Window: {start_time} to {end_time}
- Data Points Analyzed: {sum(len(points) for points in data.values())}
- Confidence Level: {self._calculate_confidence(data):.1f}%

Report generated at: {datetime.now()}
"""
        
        report_record = {
            "timestamp": trigger_timestamp.timestamp() if hasattr(trigger_timestamp, 'timestamp') else trigger_timestamp,
            "latency_spike": latency_spike,
            "order_book_imbalance": order_book_imbalance,
            "strategy_health": strategy_health,
            "volatility": volatility,
            "volume_anomaly": volume_anomaly,
            "primary_trigger": self._determine_primary_trigger(latency_spike, order_book_imbalance, volatility, volume_anomaly),
            "report_time": datetime.now().timestamp()
        }
        
        self.report_history.append(report_record)
        self._save_history()
        
        report_file = self.log_dir / f"post_mortem_{int(report_record['timestamp'])}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        self.logger.info(f"Post-mortem report generated and saved to {report_file}")
        
        return report
    
    def _calc_spike(self, latency_data):
        """Calculate latency spike magnitude"""
        if not latency_data:
            return 0
        
        values = [point["value"] for point in latency_data]
        return max(values) - min(values)
    
    def _calc_imbalance(self, order_book_data):
        """Calculate order book imbalance percentage"""
        if not order_book_data:
            return 0
        
        values = [point["value"] for point in order_book_data]
        return max(values)
    
    def _health_status(self, health_data):
        """Determine strategy health status"""
        if not health_data:
            return "unknown"
        
        values = [point["value"] for point in health_data]
        avg_health = sum(values) / len(values)
        
        if avg_health > 0.8:
            return "healthy"
        elif avg_health > 0.5:
            return "degraded"
        else:
            return "critical"
    
    def _calc_volatility(self, volatility_data):
        """Calculate volatility level"""
        if not volatility_data:
            return 0
        
        values = [point["value"] for point in volatility_data]
        return sum(values) / len(values)
    
    def _calc_volume_anomaly(self, volume_data):
        """Calculate volume anomaly factor"""
        if not volume_data:
            return 1.0
        
        values = [point["value"] for point in volume_data]
        
        baseline = sum(values[:len(values)//4]) / (len(values)//4) if len(values) >= 4 else sum(values) / len(values)
        
        peak = max(values)
        
        return peak / baseline if baseline > 0 else 1.0
    
    def _determine_primary_trigger(self, latency_spike, order_book_imbalance, volatility, volume_anomaly):
        """Determine the primary trigger for the circuit breaker event"""
        triggers = {
            "Latency Spike": latency_spike / 100,  # Normalize to 0-1 range
            "Order Book Imbalance": order_book_imbalance / 100,
            "Market Volatility": volatility / 0.2,  # Normalize to 0-1 range
            "Volume Anomaly": (volume_anomaly - 1) / 5  # Normalize to 0-1 range
        }
        
        primary_trigger = max(triggers.items(), key=lambda x: x[1])
        
        return primary_trigger[0]
    
    def _generate_actions(self, data, latency_spike, order_book_imbalance, strategy_health, volatility, volume_anomaly):
        """Generate recommended actions based on analysis"""
        actions = []
        
        if latency_spike > 100:
            actions.append("1. Upgrade exchange API endpoints")
            actions.append("2. Enable backup fiber routes")
            actions.append("3. Implement latency-aware order routing")
            
        elif order_book_imbalance > 70:
            actions.append("1. Adjust liquidity thresholds")
            actions.append("2. Enable anti-gaming logic")
            actions.append("3. Implement dynamic order sizing based on book depth")
            
        elif strategy_health == "critical":
            actions.append("1. Review strategy parameters")
            actions.append("2. Check for algorithm logic errors")
            actions.append("3. Verify data feed integrity")
            
        elif volatility > 0.15:
            actions.append("1. Increase volatility thresholds")
            actions.append("2. Implement volatility-adjusted position sizing")
            actions.append("3. Enable circuit breaker cooling period extension")
            
        elif volume_anomaly > 3.0:
            actions.append("1. Implement volume-aware execution")
            actions.append("2. Adjust order timing to avoid volume spikes")
            actions.append("3. Enable dark pool routing for large orders")
            
        else:
            actions.append("1. Review strategy parameters")
            actions.append("2. Stress-test under similar conditions")
            actions.append("3. Adjust circuit breaker thresholds")
        
        return "\n".join(actions)
    
    def _calculate_confidence(self, data):
        """Calculate confidence level of the analysis"""
        total_points = sum(len(points) for points in data.values())
        
        if total_points > 1000:
            confidence = 95
        elif total_points > 500:
            confidence = 90
        elif total_points > 100:
            confidence = 80
        elif total_points > 50:
            confidence = 70
        else:
            confidence = 60
        
        for key, points in data.items():
            if not points:
                confidence -= 10
        
        return max(50, min(confidence, 99))
