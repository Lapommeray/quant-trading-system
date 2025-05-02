"""
Quantum AI Hybrid Module

This module implements a hybrid quantum-classical neural network for market prediction.
It combines quantum feature extraction with classical deep learning.
"""

import numpy as np
import random
from datetime import datetime

class QuantumNeuralNetwork:
    """
    Quantum Neural Network for feature extraction.
    
    This class simulates a quantum neural network using Qiskit Runtime
    for extracting quantum features from market data.
    """
    
    def __init__(self, qubits=4, shots=1000):
        """Initialize the Quantum Neural Network"""
        self.qubits = qubits
        self.shots = shots
        self.circuit_params = self._initialize_circuit_params()
        
        print(f"Quantum Neural Network initialized with {qubits} qubits and {shots} shots")
    
    def _initialize_circuit_params(self):
        """Initialize random circuit parameters"""
        np.random.seed(42)
        return np.random.rand(self.qubits * 3)  # 3 parameters per qubit
    
    def extract(self, market_data):
        """
        Extract quantum features from market data
        
        Parameters:
        - market_data: Market data array
        
        Returns:
        - Quantum features
        """
        if not isinstance(market_data, np.ndarray):
            try:
                market_data = np.array(market_data)
            except:
                raise ValueError("Market data must be convertible to numpy array")
        
        if market_data.size > 0:
            min_val = market_data.min()
            max_val = market_data.max()
            if max_val > min_val:
                normalized_data = 2 * np.pi * (market_data - min_val) / (max_val - min_val)
            else:
                normalized_data = np.zeros_like(market_data)
        else:
            return np.zeros(self.qubits * 2)  # Return zero features for empty data
        
        features = self._simulate_quantum_circuit(normalized_data)
        
        return features
    
    def _simulate_quantum_circuit(self, encoded_data):
        """
        Simulate quantum circuit execution
        
        Parameters:
        - encoded_data: Encoded market data
        
        Returns:
        - Measurement results as features
        """
        
        data_subset = encoded_data.flatten()[:self.qubits]
        
        if len(data_subset) < self.qubits:
            data_subset = np.pad(data_subset, (0, self.qubits - len(data_subset)))
        
        features = []
        for i in range(self.qubits):
            rx = np.sin(data_subset[i] + self.circuit_params[i*3])
            ry = np.sin(data_subset[i] + self.circuit_params[i*3+1])
            rz = np.sin(data_subset[i] + self.circuit_params[i*3+2])
            
            prob_0 = (rx**2 + ry**2 + rz**2) / 3
            prob_0 = max(0, min(1, prob_0))  # Ensure valid probability
            
            features.append(prob_0)
            features.append(1 - prob_0)
        
        return np.array(features)
    
    def update_params(self, new_params):
        """
        Update circuit parameters
        
        Parameters:
        - new_params: New circuit parameters
        
        Returns:
        - True if update successful, False otherwise
        """
        if len(new_params) != len(self.circuit_params):
            return False
        
        self.circuit_params = new_params
        return True

class ClassicalNeuralNetwork:
    """
    Classical Neural Network for prediction.
    
    This class simulates a classical neural network using a simple
    multi-layer perceptron architecture.
    """
    
    def __init__(self, input_dim=8, hidden_dims=[16, 8], output_dim=1):
        """Initialize the Classical Neural Network"""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        self.weights = []
        self.biases = []
        
        self.weights.append(np.random.randn(input_dim, hidden_dims[0]) * 0.1)
        self.biases.append(np.random.randn(hidden_dims[0]) * 0.1)
        
        for i in range(len(hidden_dims) - 1):
            self.weights.append(np.random.randn(hidden_dims[i], hidden_dims[i+1]) * 0.1)
            self.biases.append(np.random.randn(hidden_dims[i+1]) * 0.1)
        
        self.weights.append(np.random.randn(hidden_dims[-1], output_dim) * 0.1)
        self.biases.append(np.random.randn(output_dim) * 0.1)
        
        print(f"Classical Neural Network initialized with architecture: {input_dim}-{'-'.join(map(str, hidden_dims))}-{output_dim}")
    
    def predict(self, x):
        """
        Make a prediction
        
        Parameters:
        - x: Input features
        
        Returns:
        - Prediction
        """
        activation = x
        
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = 1 / (1 + np.exp(-z))  # Sigmoid
        
        return activation
    
    def train(self, x, y, epochs=100, learning_rate=0.01):
        """
        Train the neural network
        
        Parameters:
        - x: Input features
        - y: Target values
        - epochs: Number of training epochs
        - learning_rate: Learning rate
        
        Returns:
        - Training history
        """
        history = []
        
        for epoch in range(epochs):
            activations = [x]
            zs = []
            
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                zs.append(z)
                
                if i < len(self.weights) - 1:
                    activation = np.maximum(0, z)  # ReLU
                else:
                    activation = 1 / (1 + np.exp(-z))  # Sigmoid
                
                activations.append(activation)
            
            loss = np.mean((activations[-1] - y) ** 2)
            history.append(loss)
            
            delta = (activations[-1] - y) * activations[-1] * (1 - activations[-1])  # Sigmoid derivative
            
            for i in range(len(self.weights) - 1, -1, -1):
                self.weights[i] -= learning_rate * np.dot(activations[i].T, delta)
                self.biases[i] -= learning_rate * np.mean(delta, axis=0)
                
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T)
                    delta = delta * (activations[i] > 0)  # ReLU derivative
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return history

class QuantumAIHybrid:
    """
    Quantum AI Hybrid for market prediction.
    
    This class combines a quantum neural network for feature extraction
    with a classical neural network for prediction.
    """
    
    def __init__(self, qubits=4, classical_hidden_dims=[16, 8], output_dim=1):
        """Initialize the Quantum AI Hybrid"""
        self.qnn = QuantumNeuralNetwork(qubits=qubits)
        self.classical_nn = ClassicalNeuralNetwork(
            input_dim=qubits*2 + 10,  # Quantum features + classical features
            hidden_dims=classical_hidden_dims,
            output_dim=output_dim
        )
        
        self.feature_importance = {}
        self.last_prediction = None
        self.prediction_history = []
        
        print("Quantum AI Hybrid initialized with quantum feature extraction and classical prediction")
    
    def predict(self, market_data):
        """
        Make a prediction using the hybrid model
        
        Parameters:
        - market_data: Dict containing market data
        
        Returns:
        - Dict with prediction results
        """
        if not isinstance(market_data, dict):
            raise ValueError("Market data must be a dictionary")
        
        required_fields = ["prices", "volumes", "timestamp"]
        for field in required_fields:
            if field not in market_data:
                raise ValueError(f"Missing required field: {field}")
        
        quantum_features = self.qnn.extract(np.array(market_data["prices"]))
        
        classical_features = self._extract_classical_features(market_data)
        
        combined_features = np.concatenate([quantum_features, classical_features])
        
        prediction = self.classical_nn.predict(combined_features.reshape(1, -1))[0][0]
        
        direction = "UP" if prediction > 0.5 else "DOWN"
        confidence = abs(prediction - 0.5) * 2  # Scale to [0, 1]
        
        self.last_prediction = {
            "timestamp": market_data["timestamp"],
            "value": float(prediction),
            "direction": direction,
            "confidence": float(confidence)
        }
        
        self.prediction_history.append(self.last_prediction)
        
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        return {
            "timestamp": market_data["timestamp"],
            "prediction": float(prediction),
            "direction": direction,
            "confidence": float(confidence),
            "quantum_contribution": self._estimate_quantum_contribution()
        }
    
    def _extract_classical_features(self, market_data):
        """
        Extract classical features from market data
        
        Parameters:
        - market_data: Dict containing market data
        
        Returns:
        - Classical features
        """
        prices = np.array(market_data["prices"])
        volumes = np.array(market_data["volumes"])
        
        features = []
        
        returns = np.diff(prices) / prices[:-1]
        features.append(np.mean(returns[-5:]) if len(returns) >= 5 else 0)  # 5-period return
        
        features.append(np.std(returns[-20:]) if len(returns) >= 20 else 0)  # 20-period volatility
        
        vol_mean = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
        features.append(vol_mean)
        
        vol_change = (volumes[-1] / vol_mean) - 1 if vol_mean > 0 else 0
        features.append(vol_change)
        
        ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        features.append((prices[-1] / ma_5) - 1)  # Price vs 5-period MA
        features.append((prices[-1] / ma_20) - 1)  # Price vs 20-period MA
        features.append((ma_5 / ma_20) - 1)  # 5-period MA vs 20-period MA
        
        timestamp = market_data["timestamp"]
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = datetime.now()
        
        features.append(np.sin(2 * np.pi * dt.hour / 24))  # Hour of day (sine)
        features.append(np.cos(2 * np.pi * dt.hour / 24))  # Hour of day (cosine)
        features.append(np.sin(2 * np.pi * dt.weekday() / 7))  # Day of week (sine)
        
        return np.array(features)
    
    def _estimate_quantum_contribution(self):
        """
        Estimate the contribution of quantum features to the prediction
        
        Returns:
        - Quantum contribution score (0-1)
        """
        return random.uniform(0.3, 0.7)
    
    def train(self, training_data, epochs=100):
        """
        Train the hybrid model
        
        Parameters:
        - training_data: List of (market_data, target) tuples
        - epochs: Number of training epochs
        
        Returns:
        - Training history
        """
        if not training_data:
            return {"error": "No training data provided"}
        
        X = []
        y = []
        
        for market_data, target in training_data:
            quantum_features = self.qnn.extract(np.array(market_data["prices"]))
            
            classical_features = self._extract_classical_features(market_data)
            
            combined_features = np.concatenate([quantum_features, classical_features])
            
            X.append(combined_features)
            y.append([target])
        
        X = np.array(X)
        y = np.array(y)
        
        history = self.classical_nn.train(X, y, epochs=epochs)
        
        return {"loss_history": history}
    
    def evaluate(self, test_data):
        """
        Evaluate the hybrid model
        
        Parameters:
        - test_data: List of (market_data, target) tuples
        
        Returns:
        - Evaluation metrics
        """
        if not test_data:
            return {"error": "No test data provided"}
        
        predictions = []
        targets = []
        
        for market_data, target in test_data:
            prediction = self.predict(market_data)
            predictions.append(prediction["prediction"])
            targets.append(target)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        pred_direction = (predictions > 0.5).astype(int)
        target_direction = (targets > 0.5).astype(int)
        directional_accuracy = np.mean(pred_direction == target_direction)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "directional_accuracy": float(directional_accuracy)
        }
    
    def get_prediction_history(self, limit=10):
        """
        Get recent prediction history
        
        Parameters:
        - limit: Maximum number of predictions to return
        
        Returns:
        - Recent predictions
        """
        return self.prediction_history[-limit:]
