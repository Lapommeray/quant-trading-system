"""
Anomaly Detector Module

This module implements real-time anomaly detection for market surveillance
using autoencoder and clustering techniques.
"""

import numpy as np
import datetime
import random
from collections import deque

class Autoencoder:
    """
    Autoencoder for anomaly detection.
    
    This class simulates an autoencoder neural network for detecting
    anomalies in market data streams.
    """
    
    def __init__(self, input_dim=20, latent_dim=8):
        """Initialize the Autoencoder"""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder_weights = np.random.randn(input_dim, latent_dim) * 0.1
        self.encoder_bias = np.random.randn(latent_dim) * 0.1
        
        self.decoder_weights = np.random.randn(latent_dim, input_dim) * 0.1
        self.decoder_bias = np.random.randn(input_dim) * 0.1
        
        self.loss_history = []
        
        print(f"Autoencoder initialized with input dimension {input_dim} and latent dimension {latent_dim}")
    
    def encode(self, x):
        """
        Encode input data to latent representation
        
        Parameters:
        - x: Input data
        
        Returns:
        - Latent representation
        """
        z = np.dot(x, self.encoder_weights) + self.encoder_bias
        
        return np.maximum(0, z)
    
    def decode(self, z):
        """
        Decode latent representation to reconstructed input
        
        Parameters:
        - z: Latent representation
        
        Returns:
        - Reconstructed input
        """
        x_hat = np.dot(z, self.decoder_weights) + self.decoder_bias
        
        return 1 / (1 + np.exp(-x_hat))
    
    def forward(self, x):
        """
        Forward pass through the autoencoder
        
        Parameters:
        - x: Input data
        
        Returns:
        - Reconstructed input
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def reconstruction_error(self, x):
        """
        Calculate reconstruction error
        
        Parameters:
        - x: Input data
        
        Returns:
        - Reconstruction error
        """
        x_hat = self.forward(x)
        return np.mean((x - x_hat) ** 2, axis=1)
    
    def train(self, x, epochs=100, learning_rate=0.01, batch_size=32):
        """
        Train the autoencoder
        
        Parameters:
        - x: Training data
        - epochs: Number of training epochs
        - learning_rate: Learning rate
        - batch_size: Batch size
        
        Returns:
        - Training history
        """
        n_samples = len(x)
        history = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                batch_x = x[batch_indices]
                
                z = self.encode(batch_x)
                x_hat = self.decode(z)
                
                loss = np.mean((batch_x - x_hat) ** 2)
                epoch_loss += loss * len(batch_indices)
                
                d_x_hat = 2 * (x_hat - batch_x) / len(batch_indices)
                d_decoder_bias = np.mean(d_x_hat, axis=0)
                d_decoder_weights = np.dot(z.T, d_x_hat)
                
                d_z = np.dot(d_x_hat, self.decoder_weights.T)
                d_z[z <= 0] = 0  # ReLU derivative
                
                d_encoder_bias = np.mean(d_z, axis=0)
                d_encoder_weights = np.dot(batch_x.T, d_z)
                
                self.decoder_bias -= learning_rate * d_decoder_bias
                self.decoder_weights -= learning_rate * d_decoder_weights
                self.encoder_bias -= learning_rate * d_encoder_bias
                self.encoder_weights -= learning_rate * d_encoder_weights
            
            epoch_loss /= n_samples
            history.append(epoch_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        self.loss_history.extend(history)
        return history

class HDBSCAN:
    """
    HDBSCAN clustering algorithm for anomaly detection.
    
    This class simulates the HDBSCAN clustering algorithm for identifying
    anomalies in reconstruction errors.
    """
    
    def __init__(self, min_cluster_size=50, min_samples=5):
        """Initialize HDBSCAN"""
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels_ = None
        self.probabilities_ = None
        
        print(f"HDBSCAN initialized with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels
        
        Parameters:
        - X: Input data
        
        Returns:
        - Cluster labels
        """
        
        n_samples = len(X)
        
        mean = np.mean(X)
        std = np.std(X)
        
        labels = np.zeros(n_samples, dtype=int)
        probabilities = np.zeros(n_samples)
        
        for i in range(n_samples):
            z_score = (X[i] - mean) / std if std > 0 else 0
            
            if z_score > 3.0:  # More than 3 standard deviations
                labels[i] = -1  # Outlier
                probabilities[i] = min(1.0, z_score / 5.0)
            else:
                labels[i] = 0  # Inlier
                probabilities[i] = max(0.0, 1.0 - z_score / 3.0)
        
        self.labels_ = labels
        self.probabilities_ = probabilities
        
        return labels

class AnomalyDetector:
    """
    Anomaly Detector for real-time market surveillance.
    
    This class combines an autoencoder and clustering algorithm to detect
    anomalies in market data streams.
    """
    
    def __init__(self, input_dim=20, latent_dim=8, window_size=100):
        """Initialize the Anomaly Detector"""
        self.autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cluster = HDBSCAN(min_cluster_size=max(5, window_size // 10))
        self.window_size = window_size
        
        self.data_buffer = deque(maxlen=window_size)
        
        self.anomaly_history = []
        
        self.error_threshold = None
        self.z_score_threshold = 3.0
        
        print(f"Anomaly Detector initialized with window size {window_size}")
    
    def update(self, data_point):
        """
        Update the detector with a new data point
        
        Parameters:
        - data_point: New data point
        
        Returns:
        - Dict with anomaly detection results
        """
        self.data_buffer.append(data_point)
        
        if len(self.data_buffer) < self.window_size // 2:
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "confidence": 0.0,
                "buffer_filling": len(self.data_buffer) / self.window_size
            }
        
        data = np.array(list(self.data_buffer))
        
        errors = self.autoencoder.reconstruction_error(data)
        
        if self.error_threshold is None:
            self.error_threshold = np.mean(errors) + self.z_score_threshold * np.std(errors)
        
        latest_error = errors[-1]
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        z_score = (latest_error - mean_error) / std_error if std_error > 0 else 0
        
        is_anomaly = z_score > self.z_score_threshold or latest_error > self.error_threshold
        
        anomaly_score = max(0.0, z_score / self.z_score_threshold)
        
        confidence = min(1.0, len(self.data_buffer) / self.window_size)
        
        if is_anomaly:
            self.anomaly_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "error": float(latest_error),
                "z_score": float(z_score),
                "anomaly_score": float(anomaly_score)
            })
            
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "confidence": float(confidence),
            "z_score": float(z_score),
            "reconstruction_error": float(latest_error)
        }
    
    def monitor(self, stream):
        """
        Monitor a data stream for anomalies
        
        Parameters:
        - stream: Data stream to monitor
        
        Returns:
        - List of anomaly detection results
        """
        results = []
        
        for data_point in stream:
            result = self.update(data_point)
            results.append(result)
        
        return results
    
    def train(self, training_data, epochs=100):
        """
        Train the anomaly detector
        
        Parameters:
        - training_data: Training data
        - epochs: Number of training epochs
        
        Returns:
        - Training history
        """
        history = self.autoencoder.train(training_data, epochs=epochs)
        
        errors = self.autoencoder.reconstruction_error(training_data)
        
        self.error_threshold = np.mean(errors) + self.z_score_threshold * np.std(errors)
        
        return {
            "loss_history": history,
            "error_threshold": float(self.error_threshold)
        }
    
    def get_anomaly_statistics(self):
        """
        Get statistics about detected anomalies
        
        Returns:
        - Dict with anomaly statistics
        """
        if not self.anomaly_history:
            return {
                "anomaly_count": 0,
                "anomaly_rate": 0.0,
                "last_anomaly_time": None
            }
        
        return {
            "anomaly_count": len(self.anomaly_history),
            "anomaly_rate": len(self.anomaly_history) / max(1, len(self.data_buffer)),
            "last_anomaly_time": self.anomaly_history[-1]["timestamp"],
            "average_anomaly_score": np.mean([a["anomaly_score"] for a in self.anomaly_history]),
            "max_anomaly_score": max([a["anomaly_score"] for a in self.anomaly_history])
        }

class MarketAnomalyDetector(AnomalyDetector):
    """
    Market-specific Anomaly Detector.
    
    This class extends the base AnomalyDetector with market-specific
    features and anomaly types.
    """
    
    def __init__(self, input_dim=20, latent_dim=8, window_size=100):
        """Initialize the Market Anomaly Detector"""
        super().__init__(input_dim, latent_dim, window_size)
        
        self.anomaly_types = {
            "price_spike": 0,
            "volume_spike": 0,
            "liquidity_gap": 0,
            "volatility_surge": 0,
            "correlation_break": 0,
            "pattern_break": 0
        }
        
        print("Market Anomaly Detector initialized with market-specific features")
    
    def update(self, market_data):
        """
        Update the detector with new market data
        
        Parameters:
        - market_data: Dict containing market data
        
        Returns:
        - Dict with anomaly detection results
        """
        if not isinstance(market_data, dict):
            raise ValueError("Market data must be a dictionary")
        
        features = self._extract_market_features(market_data)
        
        result = super().update(features)
        
        if result["is_anomaly"]:
            anomaly_type = self._detect_anomaly_type(market_data, features)
            result["anomaly_type"] = anomaly_type
            
            if anomaly_type in self.anomaly_types:
                self.anomaly_types[anomaly_type] += 1
        
        return result
    
    def _extract_market_features(self, market_data):
        """
        Extract market-specific features
        
        Parameters:
        - market_data: Dict containing market data
        
        Returns:
        - Feature vector
        """
        features = []
        
        if "prices" in market_data:
            prices = np.array(market_data["prices"])
            if len(prices) > 0:
                features.append(prices[-1])
                
                if len(prices) > 1:
                    features.append(prices[-1] - prices[-2])  # Absolute change
                    features.append((prices[-1] / prices[-2]) - 1)  # Percentage change
                else:
                    features.extend([0, 0])
                
                if len(prices) > 5:
                    features.append(np.mean(prices[-5:]))  # Mean
                    features.append(np.std(prices[-5:]))  # Standard deviation
                    features.append(np.max(prices[-5:]))  # Max
                    features.append(np.min(prices[-5:]))  # Min
                else:
                    features.extend([prices[-1], 0, prices[-1], prices[-1]])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        if "volumes" in market_data:
            volumes = np.array(market_data["volumes"])
            if len(volumes) > 0:
                features.append(volumes[-1])
                
                if len(volumes) > 1:
                    features.append(volumes[-1] - volumes[-2])  # Absolute change
                    features.append((volumes[-1] / max(1, volumes[-2])) - 1)  # Percentage change
                else:
                    features.extend([0, 0])
                
                if len(volumes) > 5:
                    features.append(np.mean(volumes[-5:]))  # Mean
                    features.append(np.std(volumes[-5:]))  # Standard deviation
                    features.append(np.max(volumes[-5:]))  # Max
                    features.append(np.min(volumes[-5:]))  # Min
                else:
                    features.extend([volumes[-1], 0, volumes[-1], volumes[-1]])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        if "asks" in market_data and "bids" in market_data:
            asks = market_data["asks"]
            bids = market_data["bids"]
            
            if asks and bids:
                best_ask = min(asks.keys()) if isinstance(asks, dict) else asks[0]
                best_bid = max(bids.keys()) if isinstance(bids, dict) else bids[0]
                
                spread = best_ask - best_bid
                spread_pct = spread / best_bid
                
                features.append(spread)
                features.append(spread_pct)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        if len(features) < self.autoencoder.input_dim:
            features.extend([0] * (self.autoencoder.input_dim - len(features)))
        elif len(features) > self.autoencoder.input_dim:
            features = features[:self.autoencoder.input_dim]
        
        return np.array(features)
    
    def _detect_anomaly_type(self, market_data, features):
        """
        Detect the type of market anomaly
        
        Parameters:
        - market_data: Dict containing market data
        - features: Feature vector
        
        Returns:
        - Anomaly type string
        """
        prices = np.array(market_data.get("prices", []))
        volumes = np.array(market_data.get("volumes", []))
        
        if len(prices) > 1:
            price_change = (prices[-1] / prices[-2]) - 1
            if abs(price_change) > 0.05:  # 5% price change
                return "price_spike"
        
        if len(volumes) > 5:
            volume_mean = np.mean(volumes[:-1])
            if volume_mean > 0 and volumes[-1] > volume_mean * 3:
                return "volume_spike"
        
        if "asks" in market_data and "bids" in market_data:
            asks = market_data["asks"]
            bids = market_data["bids"]
            
            if asks and bids:
                best_ask = min(asks.keys()) if isinstance(asks, dict) else asks[0]
                best_bid = max(bids.keys()) if isinstance(bids, dict) else bids[0]
                
                spread = best_ask - best_bid
                mid_price = (best_ask + best_bid) / 2
                
                if spread > mid_price * 0.01:  # Spread > 1% of mid price
                    return "liquidity_gap"
        
        if len(prices) > 10:
            recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            previous_volatility = np.std(prices[-10:-5]) / np.mean(prices[-10:-5])
            
            if previous_volatility > 0 and recent_volatility > previous_volatility * 2:
                return "volatility_surge"
        
        if "correlated_prices" in market_data and len(prices) > 5:
            correlated_prices = np.array(market_data["correlated_prices"])
            
            if len(correlated_prices) > 5:
                price_returns = np.diff(prices[-5:]) / prices[-6:-1]
                corr_returns = np.diff(correlated_prices[-5:]) / correlated_prices[-6:-1]
                
                correlation = np.corrcoef(price_returns, corr_returns)[0, 1]
                
                if abs(correlation) < 0.2:  # Low correlation
                    return "correlation_break"
        
        return "pattern_break"
    
    def get_anomaly_type_statistics(self):
        """
        Get statistics about detected anomaly types
        
        Returns:
        - Dict with anomaly type statistics
        """
        total_anomalies = sum(self.anomaly_types.values())
        
        if total_anomalies == 0:
            return {
                "total_anomalies": 0,
                "type_distribution": self.anomaly_types.copy()
            }
        
        type_distribution = {
            anomaly_type: {
                "count": count,
                "percentage": count / total_anomalies
            }
            for anomaly_type, count in self.anomaly_types.items()
        }
        
        most_common_type = max(self.anomaly_types.items(), key=lambda x: x[1])
        
        return {
            "total_anomalies": total_anomalies,
            "type_distribution": type_distribution,
            "most_common_type": most_common_type[0],
            "most_common_count": most_common_type[1]
        }
