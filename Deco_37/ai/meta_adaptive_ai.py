"""
Meta-Adaptive AI Module

This module implements a self-evolving neural architecture that adapts to market conditions
and continuously improves its prediction capabilities through reinforcement learning.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

class MetaAdaptiveAI:
    """
    Self-evolving neural architecture that adapts to market conditions.
    
    This module continuously improves its prediction capabilities through reinforcement
    learning and dynamically adjusts its architecture based on performance metrics.
    """
    
    def __init__(self, algorithm, symbol=None):
        """
        Initialize the Meta-Adaptive AI module.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        - symbol: Optional symbol to create a symbol-specific instance
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.logger = logging.getLogger(f"MetaAdaptiveAI_{symbol}" if symbol else "MetaAdaptiveAI")
        self.logger.setLevel(logging.INFO)
        
        self.models = {
            "forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "neural": MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', random_state=42)
        }
        
        self.active_model = "forest"  # Default model
        self.scaler = StandardScaler()
        
        self.performance_history = {model_name: [] for model_name in self.models.keys()}
        self.feature_importance = {}
        self.evolution_stage = 1  # Starts at stage 1 (basic)
        self.confidence_threshold = 0.65
        self.min_samples_for_training = 30
        self.last_evolution_check = None
        self.evolution_check_interval = timedelta(days=1)
        
        self.training_data = pd.DataFrame()
        self.is_trained = False
        
        self.feature_sets = {
            "basic": [
                "rsi", "macd", "bb_width", "atr", "volume_change",
                "price_change", "ma_cross", "support_resistance"
            ],
            "advanced": [
                "rsi", "macd", "bb_width", "atr", "volume_change",
                "price_change", "ma_cross", "support_resistance",
                "fractal_dimension", "hurst_exponent", "entropy",
                "correlation_matrix", "volatility_regime"
            ],
            "quantum": [
                "rsi", "macd", "bb_width", "atr", "volume_change",
                "price_change", "ma_cross", "support_resistance",
                "fractal_dimension", "hurst_exponent", "entropy",
                "correlation_matrix", "volatility_regime",
                "quantum_probability", "timeline_convergence",
                "emotional_resonance", "intention_field"
            ]
        }
        
        self.current_feature_set = "basic"
        
        self.model_path = os.path.join(
            algorithm.DataFolder, 
            "data", 
            f"meta_adaptive_model_{symbol}.pkl" if symbol else "meta_adaptive_model.pkl"
        )
        
        self._load_model()
        
        algorithm.Debug(f"Meta-Adaptive AI module initialized for {symbol}" if symbol else "Meta-Adaptive AI module initialized")
    
    def predict(self, features):
        """
        Generate predictions using the current best model.
        
        Parameters:
        - features: Dictionary of feature values
        
        Returns:
        - Dictionary containing prediction results
        """
        if not self.is_trained:
            return {"signal": "NEUTRAL", "confidence": 0.0, "model": None}
        
        feature_list = self.feature_sets[self.current_feature_set]
        available_features = [f for f in feature_list if f in features]
        
        if len(available_features) < len(feature_list) * 0.7:  # Need at least 70% of features
            return {"signal": "NEUTRAL", "confidence": 0.0, "model": None}
        
        X = np.array([[features[f] for f in available_features]])
        X_scaled = self.scaler.transform(X)
        
        model = self.models[self.active_model]
        
        try:
            probas = model.predict_proba(X_scaled)[0]
            
            if len(probas) >= 3:  # Multi-class (BUY, SELL, NEUTRAL)
                max_idx = np.argmax(probas)
                confidence = probas[max_idx]
                
                if max_idx == 0:
                    signal = "BUY"
                elif max_idx == 1:
                    signal = "SELL"
                else:
                    signal = "NEUTRAL"
            else:  # Binary (BUY vs SELL)
                confidence = probas[1]  # Probability of positive class
                
                if confidence > 0.5 + (self.confidence_threshold - 0.5) / 2:
                    signal = "BUY"
                elif confidence < 0.5 - (self.confidence_threshold - 0.5) / 2:
                    signal = "SELL"
                else:
                    signal = "NEUTRAL"
                    confidence = 1.0 - abs(confidence - 0.5) * 2  # Rescale confidence for NEUTRAL
            
            if confidence < self.confidence_threshold:
                signal = "NEUTRAL"
            
            return {
                "signal": signal,
                "confidence": confidence,
                "model": self.active_model,
                "evolution_stage": self.evolution_stage,
                "feature_set": self.current_feature_set
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return {"signal": "NEUTRAL", "confidence": 0.0, "model": None}
    
    def train(self, training_data=None):
        """
        Train all models and select the best performing one.
        
        Parameters:
        - training_data: Optional DataFrame with features and target
        
        Returns:
        - Boolean indicating if training was successful
        """
        df = training_data if training_data is not None else self.training_data
        
        if df is None or len(df) < self.min_samples_for_training:
            return False
        
        feature_list = self.feature_sets[self.current_feature_set]
        available_features = [f for f in feature_list if f in df.columns]
        
        if len(available_features) < len(feature_list) * 0.7:  # Need at least 70% of features
            return False
        
        if "target" not in df.columns:
            return False
        
        X = df[available_features]
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_accuracy = 0
        best_model = None
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                accuracy = model.score(X_test_scaled, y_test)
                
                self.performance_history[name].append({
                    "timestamp": self.algorithm.Time,
                    "accuracy": accuracy,
                    "samples": len(df)
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = name
                    
                self.logger.info(f"Model {name} trained with accuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training model {name}: {str(e)}")
        
        if best_model:
            self.active_model = best_model
            self.is_trained = True
            
            if best_model in ["forest", "boost"]:
                self.feature_importance = dict(zip(
                    available_features,
                    self.models[best_model].feature_importances_
                ))
            
            self._save_model()
            
            self._check_evolution()
            
            return True
        
        return False
    
    def add_training_sample(self, features, target):
        """
        Add a new training sample to the dataset.
        
        Parameters:
        - features: Dictionary of feature values
        - target: Target value (1 for BUY, 0 for NEUTRAL, -1 for SELL)
        
        Returns:
        - Boolean indicating if sample was added
        """
        feature_list = self.feature_sets[self.current_feature_set]
        available_features = {f: features.get(f, 0) for f in feature_list if f in features}
        
        if len(available_features) < len(feature_list) * 0.7:  # Need at least 70% of features
            return False
        
        sample = available_features.copy()
        sample["target"] = target
        sample["timestamp"] = self.algorithm.Time
        
        sample_df = pd.DataFrame([sample])
        
        if self.training_data is None or len(self.training_data) == 0:
            self.training_data = sample_df
        else:
            self.training_data = pd.concat([self.training_data, sample_df], ignore_index=True)
        
        if len(self.training_data) > 10000:
            self.training_data = self.training_data.iloc[-10000:]
        
        return True
    
    def _check_evolution(self):
        """
        Check if the AI should evolve to a higher stage.
        
        Returns:
        - Boolean indicating if evolution occurred
        """
        current_time = self.algorithm.Time
        
        if (self.last_evolution_check is not None and 
            current_time - self.last_evolution_check < self.evolution_check_interval):
            return False
        
        self.last_evolution_check = current_time
        
        if not all(len(history) >= 5 for history in self.performance_history.values()):
            return False
        
        recent_performance = {}
        for model_name, history in self.performance_history.items():
            if len(history) >= 5:
                recent_performance[model_name] = sum(h["accuracy"] for h in history[-5:]) / 5
        
        best_performance = max(recent_performance.values()) if recent_performance else 0
        
        if self.evolution_stage == 1 and best_performance > 0.65:
            self.evolution_stage = 2
            self.current_feature_set = "advanced"
            self.confidence_threshold = 0.7
            self.logger.info("Meta-Adaptive AI evolved to stage 2")
            return True
            
        elif self.evolution_stage == 2 and best_performance > 0.75:
            self.evolution_stage = 3
            self.current_feature_set = "quantum"
            self.confidence_threshold = 0.75
            
            self.models["deep_neural"] = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            self.performance_history["deep_neural"] = []
            self.logger.info("Meta-Adaptive AI evolved to stage 3")
            return True
            
        return False
    
    def _save_model(self):
        """
        Save the current model state to disk.
        
        Returns:
        - Boolean indicating if save was successful
        """
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            save_data = {
                "active_model": self.active_model,
                "models": self.models,
                "scaler": self.scaler,
                "evolution_stage": self.evolution_stage,
                "current_feature_set": self.current_feature_set,
                "confidence_threshold": self.confidence_threshold,
                "feature_importance": self.feature_importance,
                "performance_history": self.performance_history
            }
            
            joblib.dump(save_data, self.model_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _load_model(self):
        """
        Load model state from disk.
        
        Returns:
        - Boolean indicating if load was successful
        """
        if not os.path.exists(self.model_path):
            return False
            
        try:
            save_data = joblib.load(self.model_path)
            
            self.active_model = save_data["active_model"]
            self.models = save_data["models"]
            self.scaler = save_data["scaler"]
            self.evolution_stage = save_data["evolution_stage"]
            self.current_feature_set = save_data["current_feature_set"]
            self.confidence_threshold = save_data["confidence_threshold"]
            self.feature_importance = save_data["feature_importance"]
            self.performance_history = save_data["performance_history"]
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        """
        Get performance metrics for all models.
        
        Returns:
        - Dictionary containing performance metrics
        """
        metrics = {
            "active_model": self.active_model,
            "evolution_stage": self.evolution_stage,
            "feature_set": self.current_feature_set,
            "confidence_threshold": self.confidence_threshold,
            "is_trained": self.is_trained,
            "training_samples": len(self.training_data) if self.training_data is not None else 0,
            "feature_importance": self.feature_importance,
            "model_performance": {}
        }
        
        for model_name, history in self.performance_history.items():
            if history:
                recent = history[-min(5, len(history)):]
                metrics["model_performance"][model_name] = {
                    "recent_accuracy": sum(h["accuracy"] for h in recent) / len(recent),
                    "samples_seen": sum(h["samples"] for h in recent),
                    "history_length": len(history)
                }
        
        return metrics
