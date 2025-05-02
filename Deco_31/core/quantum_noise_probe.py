"""
Quantum Noise Probe Module

This module implements the Quantum Noise Probe for the QMP Overrider system.
It detects external noise deviation in quantum-secure data streams to identify macro-imbalances.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class QuantumNoiseProbe:
    """
    Quantum Noise Probe for the QMP Overrider system.
    
    This class detects external noise deviation in quantum-secure data streams to identify
    macro-imbalances. Quantum noise bursts often precede central bank routing shifts and
    other significant market events.
    """
    
    def __init__(self, log_dir=None, noise_threshold=0.15):
        """
        Initialize the Quantum Noise Probe.
        
        Parameters:
        - log_dir: Directory to store noise logs (or None for default)
        - noise_threshold: Threshold for noise detection (0.0 to 1.0)
        """
        self.logger = logging.getLogger("QuantumNoiseProbe")
        
        if log_dir is None:
            self.log_dir = Path("logs/quantum_noise")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.noise_threshold = noise_threshold
        
        self.reference_distribution = None
        self.noise_history = []
        self.current_noise_level = 0.0
        self.anomaly_predictions = {}
        
        self._load_data()
        
        if self.reference_distribution is None:
            self._initialize_reference_distribution()
        
        self.logger.info(f"Quantum Noise Probe initialized with threshold: {noise_threshold}")
    
    def _load_data(self):
        """Load existing noise data"""
        noise_file = self.log_dir / "noise_history.json"
        reference_file = self.log_dir / "reference_distribution.json"
        
        if noise_file.exists():
            try:
                with open(noise_file, "r") as f:
                    data = json.load(f)
                    
                    if "noise_history" in data:
                        self.noise_history = data["noise_history"]
                    
                    if "current_noise_level" in data:
                        self.current_noise_level = data["current_noise_level"]
                    
                    if "anomaly_predictions" in data:
                        self.anomaly_predictions = data["anomaly_predictions"]
                
                self.logger.info(f"Loaded noise history with {len(self.noise_history)} entries")
            except Exception as e:
                self.logger.error(f"Error loading noise data: {e}")
        
        if reference_file.exists():
            try:
                with open(reference_file, "r") as f:
                    data = json.load(f)
                    
                    if "reference_distribution" in data:
                        self.reference_distribution = np.array(data["reference_distribution"])
                
                self.logger.info("Loaded reference distribution")
            except Exception as e:
                self.logger.error(f"Error loading reference distribution: {e}")
    
    def _save_data(self):
        """Save noise data to file"""
        noise_file = self.log_dir / "noise_history.json"
        reference_file = self.log_dir / "reference_distribution.json"
        
        try:
            noise_data = {
                "noise_history": self.noise_history,
                "current_noise_level": self.current_noise_level,
                "anomaly_predictions": self.anomaly_predictions
            }
            
            with open(noise_file, "w") as f:
                json.dump(noise_data, f, indent=2)
            
            if self.reference_distribution is not None:
                reference_data = {
                    "reference_distribution": self.reference_distribution.tolist()
                }
                
                with open(reference_file, "w") as f:
                    json.dump(reference_data, f, indent=2)
            
            self.logger.info("Saved noise data")
        except Exception as e:
            self.logger.error(f"Error saving noise data: {e}")
    
    def _initialize_reference_distribution(self):
        """Initialize reference distribution for noise detection"""
        self.reference_distribution = np.random.normal(0, 1, 1000)
        self.logger.info("Initialized reference distribution")
    
    def _create_reference_circuit(self):
        """
        Create reference quantum circuit for noise detection.
        
        In a real quantum system, this would create an actual quantum circuit.
        For simulation, we'll just return a placeholder.
        """
        return "reference_circuit"
    
    def _simulate_quantum_measurement(self, circuit, noise_level=0.0):
        """
        Simulate quantum measurement with optional noise.
        
        In a real quantum system, this would perform actual quantum measurements.
        For simulation, we'll generate random data with the specified noise level.
        
        Parameters:
        - circuit: Quantum circuit to measure
        - noise_level: Noise level to add (0.0 to 1.0)
        
        Returns:
        - Simulated measurement results
        """
        base_distribution = np.random.normal(0, 1, 1000)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 5, 1000)
            noisy_distribution = base_distribution + noise
        else:
            noisy_distribution = base_distribution
        
        return noisy_distribution
    
    def _calculate_hellinger_distance(self, dist1, dist2):
        """
        Calculate Hellinger distance between two distributions.
        
        Parameters:
        - dist1: First distribution
        - dist2: Second distribution
        
        Returns:
        - Hellinger distance (0.0 to 1.0)
        """
        hist1, _ = np.histogram(dist1, bins=50, density=True)
        hist2, _ = np.histogram(dist2, bins=50, density=True)
        
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
        
        return hellinger
    
    def measure_noise(self, external_data=None, timestamp=None):
        """
        Measure quantum noise level.
        
        Parameters:
        - external_data: External data to use for noise detection (or None to simulate)
        - timestamp: Timestamp for the measurement (or None for current time)
        
        Returns:
        - Noise level (0.0 to 1.0)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        elif isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        circuit = self._create_reference_circuit()
        
        if external_data is not None:
            measurement_results = external_data
        else:
            random_noise = random.random() * 0.3  # Random noise level up to 0.3
            measurement_results = self._simulate_quantum_measurement(circuit, random_noise)
        
        noise_level = self._calculate_hellinger_distance(self.reference_distribution, measurement_results)
        
        noise_entry = {
            "timestamp": timestamp,
            "noise_level": noise_level,
            "is_anomaly": noise_level > self.noise_threshold
        }
        
        self.noise_history.append(noise_entry)
        
        self.current_noise_level = noise_level
        
        if noise_level > self.noise_threshold:
            self._predict_anomalies(noise_level, timestamp)
        
        self._save_data()
        
        return noise_level
    
    def _predict_anomalies(self, noise_level, timestamp):
        """
        Predict anomalies based on noise level.
        
        Parameters:
        - noise_level: Noise level (0.0 to 1.0)
        - timestamp: Timestamp for the prediction
        
        Returns:
        - Dictionary with anomaly predictions
        """
        anomaly_probability = min(1.0, noise_level * 2)
        
        if noise_level > 0.5:
            anomaly_type = "extreme"
            time_window = "24h"
            description = "Extreme market event imminent"
        elif noise_level > 0.3:
            anomaly_type = "major"
            time_window = "48h"
            description = "Major market shift expected"
        else:
            anomaly_type = "minor"
            time_window = "72h"
            description = "Minor market disruption possible"
        
        anomaly_prediction = {
            "timestamp": timestamp,
            "noise_level": noise_level,
            "anomaly_type": anomaly_type,
            "anomaly_probability": anomaly_probability,
            "time_window": time_window,
            "description": description
        }
        
        self.anomaly_predictions[timestamp] = anomaly_prediction
        
        return anomaly_prediction
    
    def get_current_noise_level(self):
        """
        Get current noise level.
        
        Returns:
        - Current noise level (0.0 to 1.0)
        """
        return self.current_noise_level
    
    def get_noise_history(self, start_date=None, end_date=None, min_level=0.0):
        """
        Get noise history.
        
        Parameters:
        - start_date: Start date for history (ISO format)
        - end_date: End date for history (ISO format)
        - min_level: Minimum noise level to include
        
        Returns:
        - List of noise entries
        """
        filtered_history = self.noise_history
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            filtered_history = [e for e in filtered_history if "timestamp" in e and datetime.fromisoformat(e["timestamp"]) >= start_dt]
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            filtered_history = [e for e in filtered_history if "timestamp" in e and datetime.fromisoformat(e["timestamp"]) <= end_dt]
        
        filtered_history = [e for e in filtered_history if e["noise_level"] >= min_level]
        
        return filtered_history
    
    def get_anomaly_predictions(self, start_date=None, end_date=None, min_probability=0.0):
        """
        Get anomaly predictions.
        
        Parameters:
        - start_date: Start date for predictions (ISO format)
        - end_date: End date for predictions (ISO format)
        - min_probability: Minimum anomaly probability to include
        
        Returns:
        - Dictionary with anomaly predictions
        """
        filtered_predictions = {}
        
        for timestamp, prediction in self.anomaly_predictions.items():
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
                if datetime.fromisoformat(timestamp) < start_dt:
                    continue
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
                if datetime.fromisoformat(timestamp) > end_dt:
                    continue
            
            if prediction["anomaly_probability"] >= min_probability:
                filtered_predictions[timestamp] = prediction
        
        return filtered_predictions
    
    def is_anomaly_predicted(self):
        """
        Check if an anomaly is currently predicted.
        
        Returns:
        - True if an anomaly is predicted, False otherwise
        """
        return self.current_noise_level > self.noise_threshold
    
    def get_latest_anomaly_prediction(self):
        """
        Get the latest anomaly prediction.
        
        Returns:
        - Latest anomaly prediction or None if no anomalies are predicted
        """
        if not self.anomaly_predictions:
            return None
        
        latest_timestamp = max(self.anomaly_predictions.keys())
        return self.anomaly_predictions[latest_timestamp]
    
    def plot_noise_history(self, output_file=None, show=False):
        """
        Plot noise history.
        
        Parameters:
        - output_file: Path to save plot (or None to not save)
        - show: Whether to show the plot
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.noise_history:
            self.logger.warning("No noise history to plot")
            return False
        
        try:
            timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in self.noise_history]
            noise_levels = [entry["noise_level"] for entry in self.noise_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, noise_levels, marker='o', linestyle='-', label='Noise Level')
            
            plt.axhline(y=self.noise_threshold, color='r', linestyle='--', label=f'Threshold ({self.noise_threshold})')
            
            plt.xlabel('Timestamp')
            plt.ylabel('Noise Level')
            plt.title('Quantum Noise History')
            plt.legend()
            plt.grid(True)
            
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved noise history plot to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error plotting noise history: {e}")
            return False
    
    def export_data(self, output_dir=None, format="json"):
        """
        Export noise data.
        
        Parameters:
        - output_dir: Directory to save exported data (or None for default)
        - format: Format to export data ("json", "csv", or "excel")
        
        Returns:
        - True if successful, False otherwise
        """
        if output_dir is None:
            output_dir = self.log_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "json":
                history_file = output_dir / "noise_history.json"
                with open(history_file, "w") as f:
                    json.dump(self.noise_history, f, indent=2)
                
                predictions_file = output_dir / "anomaly_predictions.json"
                with open(predictions_file, "w") as f:
                    json.dump(self.anomaly_predictions, f, indent=2)
            elif format.lower() == "csv":
                history_df = pd.DataFrame(self.noise_history)
                history_file = output_dir / "noise_history.csv"
                history_df.to_csv(history_file, index=False)
                
                predictions_flat = []
                for timestamp, prediction in self.anomaly_predictions.items():
                    predictions_flat.append(prediction)
                
                predictions_df = pd.DataFrame(predictions_flat)
                predictions_file = output_dir / "anomaly_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
            elif format.lower() == "excel":
                excel_file = output_dir / "quantum_noise_data.xlsx"
                with pd.ExcelWriter(excel_file) as writer:
                    history_df = pd.DataFrame(self.noise_history)
                    history_df.to_excel(writer, sheet_name="Noise History", index=False)
                    
                    predictions_flat = []
                    for timestamp, prediction in self.anomaly_predictions.items():
                        predictions_flat.append(prediction)
                    
                    predictions_df = pd.DataFrame(predictions_flat)
                    predictions_df.to_excel(writer, sheet_name="Anomaly Predictions", index=False)
            else:
                self.logger.warning(f"Invalid export format: {format}")
                return False
            
            self.logger.info(f"Exported noise data to {output_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting noise data: {e}")
            return False
    
    def calibrate_reference_distribution(self, calibration_data=None):
        """
        Calibrate reference distribution for noise detection.
        
        Parameters:
        - calibration_data: Data to use for calibration (or None to simulate)
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            if calibration_data is not None:
                self.reference_distribution = calibration_data
            else:
                circuit = self._create_reference_circuit()
                self.reference_distribution = self._simulate_quantum_measurement(circuit)
            
            self._save_data()
            
            self.logger.info("Calibrated reference distribution")
            return True
        except Exception as e:
            self.logger.error(f"Error calibrating reference distribution: {e}")
            return False
    
    def monitor_quantum_anomalies(self, interval=300, callback=None):
        """
        Monitor quantum anomalies in a separate thread.
        
        Parameters:
        - interval: Interval between measurements in seconds
        - callback: Callback function to call when an anomaly is detected
        
        Returns:
        - Thread object
        """
        import threading
        import time
        
        def monitor_thread():
            self.logger.info(f"Starting quantum anomaly monitoring with interval: {interval}s")
            
            while True:
                try:
                    noise_level = self.measure_noise()
                    
                    if noise_level > self.noise_threshold:
                        self.logger.warning(f"Quantum anomaly detected: {noise_level}")
                        
                        if callback:
                            try:
                                callback(noise_level)
                            except Exception as e:
                                self.logger.error(f"Error in callback: {e}")
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Error in monitor thread: {e}")
                    time.sleep(interval)
        
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
        
        return thread
