"""
Model Confidence Tracker Module

This module implements the Model Confidence Tracker for the QMP Overrider system.
It tracks the confidence of AI models over time and helps identify when models need retraining.
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

class ModelConfidenceTracker:
    """
    Model Confidence Tracker for the QMP Overrider system.
    
    This class tracks the confidence of AI models over time and helps identify when
    models need retraining based on confidence trends and performance metrics.
    """
    
    def __init__(self, log_dir=None, models=None):
        """
        Initialize the Model Confidence Tracker.
        
        Parameters:
        - log_dir: Directory to store confidence logs (or None for default)
        - models: List of model names to track (or None for auto-detection)
        """
        self.logger = logging.getLogger("ModelConfidenceTracker")
        
        if log_dir is None:
            self.log_dir = Path("logs/model_confidence")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = models or []
        
        self.confidence_data = defaultdict(list)
        self.performance_data = defaultdict(list)
        
        self._load_data()
        
        self.logger.info(f"Model Confidence Tracker initialized for {len(self.models)} models")
    
    def _load_data(self):
        """Load existing confidence data"""
        confidence_file = self.log_dir / "confidence_data.json"
        performance_file = self.log_dir / "performance_data.json"
        
        if confidence_file.exists():
            try:
                with open(confidence_file, "r") as f:
                    data = json.load(f)
                    
                    for model, entries in data.items():
                        self.confidence_data[model] = entries
                        
                        if model not in self.models:
                            self.models.append(model)
                
                self.logger.info(f"Loaded confidence data for {len(self.confidence_data)} models")
            except Exception as e:
                self.logger.error(f"Error loading confidence data: {e}")
        
        if performance_file.exists():
            try:
                with open(performance_file, "r") as f:
                    data = json.load(f)
                    
                    for model, entries in data.items():
                        self.performance_data[model] = entries
                
                self.logger.info(f"Loaded performance data for {len(self.performance_data)} models")
            except Exception as e:
                self.logger.error(f"Error loading performance data: {e}")
    
    def _save_data(self):
        """Save confidence data to file"""
        confidence_file = self.log_dir / "confidence_data.json"
        performance_file = self.log_dir / "performance_data.json"
        
        try:
            with open(confidence_file, "w") as f:
                json.dump(dict(self.confidence_data), f, indent=2)
            
            with open(performance_file, "w") as f:
                json.dump(dict(self.performance_data), f, indent=2)
            
            self.logger.info("Saved confidence and performance data")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def track_confidence(self, model_name, confidence, timestamp=None, metadata=None):
        """
        Track model confidence.
        
        Parameters:
        - model_name: Name of the model
        - confidence: Confidence score (0.0 to 1.0)
        - timestamp: Timestamp (or None for current time)
        - metadata: Additional metadata about the prediction
        
        Returns:
        - True if successful, False otherwise
        """
        if not 0.0 <= confidence <= 1.0:
            self.logger.warning(f"Invalid confidence value: {confidence}. Must be between 0.0 and 1.0.")
            return False
        
        if model_name not in self.models:
            self.models.append(model_name)
        
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        elif isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.isoformat()
        
        entry = {
            "timestamp": timestamp,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.confidence_data[model_name].append(entry)
        
        self._save_data()
        
        return True
    
    def track_performance(self, model_name, accuracy, timestamp=None, metadata=None):
        """
        Track model performance.
        
        Parameters:
        - model_name: Name of the model
        - accuracy: Accuracy score (0.0 to 1.0)
        - timestamp: Timestamp (or None for current time)
        - metadata: Additional metadata about the performance
        
        Returns:
        - True if successful, False otherwise
        """
        if not 0.0 <= accuracy <= 1.0:
            self.logger.warning(f"Invalid accuracy value: {accuracy}. Must be between 0.0 and 1.0.")
            return False
        
        if model_name not in self.models:
            self.models.append(model_name)
        
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        elif isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.isoformat()
        
        entry = {
            "timestamp": timestamp,
            "accuracy": accuracy,
            "metadata": metadata or {}
        }
        
        self.performance_data[model_name].append(entry)
        
        self._save_data()
        
        return True
    
    def get_confidence_trend(self, model_name, window=10):
        """
        Get confidence trend for a model.
        
        Parameters:
        - model_name: Name of the model
        - window: Window size for moving average
        
        Returns:
        - Dictionary with trend data
        """
        if model_name not in self.confidence_data:
            return {"trend": "unknown", "slope": 0.0, "current": 0.0, "history": []}
        
        entries = self.confidence_data[model_name]
        if not entries:
            return {"trend": "unknown", "slope": 0.0, "current": 0.0, "history": []}
        
        confidences = [entry["confidence"] for entry in entries]
        
        if len(confidences) >= window:
            moving_avg = []
            for i in range(len(confidences) - window + 1):
                avg = sum(confidences[i:i+window]) / window
                moving_avg.append(avg)
            
            if len(moving_avg) >= 2:
                slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
                
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                slope = 0.0
                trend = "unknown"
        else:
            moving_avg = confidences
            slope = 0.0
            trend = "unknown"
        
        return {
            "trend": trend,
            "slope": slope,
            "current": confidences[-1] if confidences else 0.0,
            "history": confidences,
            "moving_avg": moving_avg
        }
    
    def get_performance_trend(self, model_name, window=10):
        """
        Get performance trend for a model.
        
        Parameters:
        - model_name: Name of the model
        - window: Window size for moving average
        
        Returns:
        - Dictionary with trend data
        """
        if model_name not in self.performance_data:
            return {"trend": "unknown", "slope": 0.0, "current": 0.0, "history": []}
        
        entries = self.performance_data[model_name]
        if not entries:
            return {"trend": "unknown", "slope": 0.0, "current": 0.0, "history": []}
        
        accuracies = [entry["accuracy"] for entry in entries]
        
        if len(accuracies) >= window:
            moving_avg = []
            for i in range(len(accuracies) - window + 1):
                avg = sum(accuracies[i:i+window]) / window
                moving_avg.append(avg)
            
            if len(moving_avg) >= 2:
                slope = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
                
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                slope = 0.0
                trend = "unknown"
        else:
            moving_avg = accuracies
            slope = 0.0
            trend = "unknown"
        
        return {
            "trend": trend,
            "slope": slope,
            "current": accuracies[-1] if accuracies else 0.0,
            "history": accuracies,
            "moving_avg": moving_avg
        }
    
    def needs_retraining(self, model_name, confidence_threshold=0.7, performance_threshold=0.6, trend_window=10):
        """
        Check if a model needs retraining.
        
        Parameters:
        - model_name: Name of the model
        - confidence_threshold: Threshold for confidence
        - performance_threshold: Threshold for performance
        - trend_window: Window size for trend analysis
        
        Returns:
        - True if model needs retraining, False otherwise
        """
        confidence_trend = self.get_confidence_trend(model_name, trend_window)
        
        performance_trend = self.get_performance_trend(model_name, trend_window)
        
        if confidence_trend["current"] < confidence_threshold:
            return True
        
        if performance_trend["current"] < performance_threshold:
            return True
        
        if confidence_trend["trend"] == "decreasing" and confidence_trend["slope"] < -0.05:
            return True
        
        if performance_trend["trend"] == "decreasing" and performance_trend["slope"] < -0.05:
            return True
        
        return False
    
    def get_model_health(self, model_name):
        """
        Get health status of a model.
        
        Parameters:
        - model_name: Name of the model
        
        Returns:
        - Dictionary with health status
        """
        confidence_trend = self.get_confidence_trend(model_name)
        
        performance_trend = self.get_performance_trend(model_name)
        
        confidence = confidence_trend["current"]
        performance = performance_trend["current"]
        
        health_score = (confidence + performance) / 2
        
        if health_score >= 0.8:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.6:
            status = "fair"
        elif health_score >= 0.5:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "model": model_name,
            "health_score": health_score,
            "status": status,
            "confidence": confidence,
            "performance": performance,
            "confidence_trend": confidence_trend["trend"],
            "performance_trend": performance_trend["trend"],
            "needs_retraining": self.needs_retraining(model_name)
        }
    
    def get_all_models_health(self):
        """
        Get health status of all models.
        
        Returns:
        - Dictionary with health status for all models
        """
        health = {}
        for model in self.models:
            health[model] = self.get_model_health(model)
        
        return health
    
    def plot_confidence_trend(self, model_name, output_file=None, show=False):
        """
        Plot confidence trend for a model.
        
        Parameters:
        - model_name: Name of the model
        - output_file: Path to save plot (or None to not save)
        - show: Whether to show the plot
        
        Returns:
        - True if successful, False otherwise
        """
        if model_name not in self.confidence_data:
            self.logger.warning(f"No confidence data for model: {model_name}")
            return False
        
        entries = self.confidence_data[model_name]
        if not entries:
            self.logger.warning(f"Empty confidence data for model: {model_name}")
            return False
        
        try:
            timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) for entry in entries]
            confidences = [entry["confidence"] for entry in entries]
            
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, confidences, marker='o', linestyle='-', label='Confidence')
            
            if len(confidences) >= 5:
                window = min(5, len(confidences))
                moving_avg = []
                for i in range(len(confidences) - window + 1):
                    avg = sum(confidences[i:i+window]) / window
                    moving_avg.append(avg)
                
                plt.plot(timestamps[window-1:], moving_avg, linestyle='--', label=f'{window}-point Moving Avg')
            
            plt.xlabel('Timestamp')
            plt.ylabel('Confidence')
            plt.title(f'Confidence Trend for {model_name}')
            plt.legend()
            plt.grid(True)
            
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved confidence trend plot to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error plotting confidence trend: {e}")
            return False
    
    def plot_performance_trend(self, model_name, output_file=None, show=False):
        """
        Plot performance trend for a model.
        
        Parameters:
        - model_name: Name of the model
        - output_file: Path to save plot (or None to not save)
        - show: Whether to show the plot
        
        Returns:
        - True if successful, False otherwise
        """
        if model_name not in self.performance_data:
            self.logger.warning(f"No performance data for model: {model_name}")
            return False
        
        entries = self.performance_data[model_name]
        if not entries:
            self.logger.warning(f"Empty performance data for model: {model_name}")
            return False
        
        try:
            timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) for entry in entries]
            accuracies = [entry["accuracy"] for entry in entries]
            
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, accuracies, marker='o', linestyle='-', label='Accuracy')
            
            if len(accuracies) >= 5:
                window = min(5, len(accuracies))
                moving_avg = []
                for i in range(len(accuracies) - window + 1):
                    avg = sum(accuracies[i:i+window]) / window
                    moving_avg.append(avg)
                
                plt.plot(timestamps[window-1:], moving_avg, linestyle='--', label=f'{window}-point Moving Avg')
            
            plt.xlabel('Timestamp')
            plt.ylabel('Accuracy')
            plt.title(f'Performance Trend for {model_name}')
            plt.legend()
            plt.grid(True)
            
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved performance trend plot to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error plotting performance trend: {e}")
            return False
    
    def export_data(self, output_dir=None, format="csv"):
        """
        Export confidence and performance data.
        
        Parameters:
        - output_dir: Directory to save exported data (or None for default)
        - format: Format to export data ("csv", "json", or "excel")
        
        Returns:
        - True if successful, False otherwise
        """
        if output_dir is None:
            output_dir = self.log_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            confidence_data_flat = []
            for model, entries in self.confidence_data.items():
                for entry in entries:
                    flat_entry = {
                        "model": model,
                        "timestamp": entry["timestamp"],
                        "confidence": entry["confidence"]
                    }
                    
                    for key, value in entry.get("metadata", {}).items():
                        flat_entry[f"metadata_{key}"] = value
                    
                    confidence_data_flat.append(flat_entry)
            
            performance_data_flat = []
            for model, entries in self.performance_data.items():
                for entry in entries:
                    flat_entry = {
                        "model": model,
                        "timestamp": entry["timestamp"],
                        "accuracy": entry["accuracy"]
                    }
                    
                    for key, value in entry.get("metadata", {}).items():
                        flat_entry[f"metadata_{key}"] = value
                    
                    performance_data_flat.append(flat_entry)
            
            if format.lower() == "csv":
                confidence_df = pd.DataFrame(confidence_data_flat)
                confidence_file = output_dir / "confidence_data.csv"
                confidence_df.to_csv(confidence_file, index=False)
                
                performance_df = pd.DataFrame(performance_data_flat)
                performance_file = output_dir / "performance_data.csv"
                performance_df.to_csv(performance_file, index=False)
            elif format.lower() == "json":
                confidence_file = output_dir / "confidence_data.json"
                with open(confidence_file, "w") as f:
                    json.dump(confidence_data_flat, f, indent=2)
                
                performance_file = output_dir / "performance_data.json"
                with open(performance_file, "w") as f:
                    json.dump(performance_data_flat, f, indent=2)
            elif format.lower() == "excel":
                confidence_df = pd.DataFrame(confidence_data_flat)
                
                performance_df = pd.DataFrame(performance_data_flat)
                
                excel_file = output_dir / "model_confidence_data.xlsx"
                with pd.ExcelWriter(excel_file) as writer:
                    confidence_df.to_excel(writer, sheet_name="Confidence", index=False)
                    performance_df.to_excel(writer, sheet_name="Performance", index=False)
            else:
                self.logger.warning(f"Invalid export format: {format}")
                return False
            
            self.logger.info(f"Exported data to {output_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
