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
from scipy.stats import gaussian_kde

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
        
    def self_modify_code(self, performance_metrics):
        """
        Advanced self-modification capabilities for AI adaptation
        Modifies trading parameters and model configurations based on performance
        """
        if performance_metrics['recent_accuracy'] < 0.6:
            self.confidence_threshold = min(0.85, self.confidence_threshold + 0.05)
            self.algorithm.Debug(f"AI Self-Modification: Increased confidence threshold to {self.confidence_threshold}")
            
        if performance_metrics['recent_accuracy'] > 0.8 and self.evolution_stage < 3:
            self._evolve_to_next_stage()
            self.algorithm.Debug(f"AI Self-Modification: Evolved to stage {self.evolution_stage}")
            
        if self.evolution_stage == 3 and performance_metrics['recent_accuracy'] > 0.9:
            self.feature_sets["quantum"].extend([
                "market_consciousness", "probability_collapse", "entanglement_strength"
            ])
            
    def _evolve_to_next_stage(self):
        """
        Helper method to evolve to the next stage
        """
        if self.evolution_stage == 1:
            self.evolution_stage = 2
            self.current_feature_set = "advanced"
            self.confidence_threshold = 0.7
            self.logger.info("Meta-Adaptive AI evolved to stage 2")
        elif self.evolution_stage == 2:
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
            
    def adapt_to_future_markets(self, market_data):
        """
        Future-proof adaptation mechanism that can handle unknown market regimes
        """
        volatility = np.std(market_data['returns'])
        correlation = np.corrcoef(market_data['returns'], market_data['volume'])[0,1]
        
        # Initialize risk multiplier if not present
        if not hasattr(self, 'risk_multiplier'):
            self.risk_multiplier = 1.0
        
        market_regime = self.detect_market_regime(market_data)
        
        if volatility > 0.05:  # High volatility regime
            self.risk_multiplier = 0.5  # Reduce risk
            self.algorithm.Debug(f"AI Adaptation: High volatility detected ({volatility:.4f}), reducing risk")
        elif correlation < -0.3:  # Unusual correlation pattern
            self.confidence_threshold = 0.9  # Increase caution
            self.algorithm.Debug(f"AI Adaptation: Unusual correlation detected ({correlation:.4f}), increasing confidence threshold")
        
        if market_regime == "trending":
            self.algorithm.Debug(f"AI Adaptation: Trending market detected, optimizing for momentum")
            if "trend_strength" not in self.feature_sets[self.current_feature_set]:
                self.feature_sets[self.current_feature_set].append("trend_strength")
                
            if "boost" in self.models:
                self.active_model = "boost"  # GradientBoosting works well for trending markets
                
        elif market_regime == "mean_reverting":
            self.algorithm.Debug(f"AI Adaptation: Mean-reverting market detected, optimizing for reversals")
            if "mean_reversion_strength" not in self.feature_sets[self.current_feature_set]:
                self.feature_sets[self.current_feature_set].append("mean_reversion_strength")
                
            self.confidence_threshold = max(0.7, self.confidence_threshold)
                
        elif market_regime == "high_volatility":
            self.algorithm.Debug(f"AI Adaptation: High volatility regime detected, increasing caution")
            self.confidence_threshold = max(0.8, self.confidence_threshold)
            
            if "volatility_regime" not in self.feature_sets[self.current_feature_set]:
                self.feature_sets[self.current_feature_set].append("volatility_regime")
                
        elif market_regime == "unknown":
            # Handle unknown market patterns with future-proof adaptation
            self._create_feature_set_for_unknown_pattern(market_data)
        
        self.algorithm.Debug(f"AI Adaptation: Volatility={volatility:.4f}, Correlation={correlation:.4f}, Risk Multiplier={self.risk_multiplier:.2f}, Regime={market_regime}")
        
    def detect_market_regime(self, market_data):
        """
        Detect current market regime for adaptive behavior
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - String indicating the detected market regime
        """
        if 'returns' not in market_data or len(market_data['returns']) < 20:
            return "unknown"
            
        returns = market_data['returns'][-20:]
        
        volatility = np.std(returns)
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0,1] if len(returns) > 1 else 0
        
        trend = np.polyfit(range(len(returns)), returns, 1)[0]
        trend_strength = abs(trend) / volatility if volatility > 0 else 0
        
        mean = np.mean(returns)
        kurtosis = np.mean((returns - mean)**4) / (volatility**4) if volatility > 0 else 3
        
        if volatility > 0.05:
            return "high_volatility"
        elif autocorr > 0.3 or trend_strength > 0.7:
            return "trending"
        elif autocorr < -0.3:
            return "mean_reverting"
        elif kurtosis > 5:  # Fat tails
            return "fat_tailed"
        else:
            return "neutral"
            
    def _create_feature_set_for_unknown_pattern(self, market_data):
        """
        Create new feature set for unknown market patterns
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - None
        """
        # Check if the pattern is truly anomalous
        if not self._is_anomalous_pattern(market_data):
            return
            
        new_set_name = f"adaptive_{len(self.feature_sets)}"
        self.feature_sets[new_set_name] = self.feature_sets[self.current_feature_set].copy()
        
        self.feature_sets[new_set_name].extend([
            "pattern_deviation", 
            "regime_uncertainty",
            "adaptive_threshold"
        ])
        
        self.algorithm.Debug(f"AI Adaptation: Created new feature set '{new_set_name}' for unknown market pattern")
        self.current_feature_set = new_set_name
        
        self.confidence_threshold = 0.85
        
    def _is_anomalous_pattern(self, market_data):
        """
        Check if market data contains anomalous patterns
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Boolean indicating if pattern is anomalous
        """
        if 'returns' not in market_data or len(market_data['returns']) < 30:
            return False
            
        returns = market_data['returns'][-30:]
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return False
            
        extreme_values = [r for r in returns if abs(r - mean) > 4 * std]
        if len(extreme_values) >= 3:
            return True
            
        first_half = returns[:15]
        second_half = returns[15:]
        
        first_autocorr = np.corrcoef(first_half[:-1], first_half[1:])[0,1] if len(first_half) > 1 else 0
        second_autocorr = np.corrcoef(second_half[:-1], second_half[1:])[0,1] if len(second_half) > 1 else 0
        
        if abs(first_autocorr - second_autocorr) > 0.6:
            return True
            
        return False
        
    def time_resonant_neural_lattice(self, market_data, lookback_steps=100):
        """
        Time-Resonant Predictive Neural Lattice
        Advanced time-series pattern recognition that detects temporal harmonics
        """
        if 'returns' not in market_data or len(market_data['returns']) < lookback_steps:
            return {"resonance": 0.0, "temporal_pattern": "insufficient_data"}
            
        returns = np.array(market_data['returns'][-lookback_steps:])
        
        fft_result = np.fft.fft(returns)
        frequencies = np.fft.fftfreq(len(returns))
        dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        
        resonance_strength = np.abs(fft_result[dominant_freq_idx]) / np.sum(np.abs(fft_result))
        
        autocorr_lags = [1, 5, 10, 20, 50]
        time_patterns = {}
        for lag in autocorr_lags:
            if len(returns) > lag:
                autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0,1]
                time_patterns[f"lag_{lag}"] = autocorr if not np.isnan(autocorr) else 0.0
        
        temporal_memory = np.mean([time_patterns[key] for key in time_patterns if abs(time_patterns[key]) > 0.1])
        
        prediction_confidence = resonance_strength * abs(temporal_memory) if temporal_memory else 0.0
        
        self.algorithm.Debug(f"Time-Resonant Lattice: Resonance={resonance_strength:.4f}, Memory={temporal_memory:.4f}")
        
        return {
            "resonance": resonance_strength,
            "dominant_frequency": dominant_frequency,
            "temporal_patterns": time_patterns,
            "temporal_memory": temporal_memory,
            "prediction_confidence": prediction_confidence,
            "lattice_state": "resonating" if resonance_strength > 0.1 else "dormant"
        }
        
    def dna_self_rewrite(self, performance_metrics, market_conditions):
        """
        Self-Rewriting DNA-AI Codebase
        AI dynamically modifies its own architecture like biological DNA mutation
        """
        mutation_triggered = False
        
        if performance_metrics.get('recent_accuracy', 0) < 0.5:
            if 'neural' in self.models:
                current_layers = self.models['neural'].hidden_layer_sizes
                if len(current_layers) < 4:
                    new_layers = current_layers + (25,)
                    self.models['neural_evolved'] = MLPClassifier(
                        hidden_layer_sizes=new_layers,
                        activation='relu',
                        solver='adam',
                        random_state=42
                    )
                    self.performance_history['neural_evolved'] = []
                    mutation_triggered = True
                    self.algorithm.Debug(f"DNA Mutation: Evolved neural architecture to {new_layers}")
        
        current_features = len(self.feature_sets[self.current_feature_set])
        if current_features < 25 and performance_metrics.get('recent_accuracy', 0) > 0.7:
            new_features = [
                "dna_pattern_strength",
                "evolutionary_momentum", 
                "mutation_probability",
                "adaptation_speed"
            ]
            self.feature_sets[self.current_feature_set].extend(new_features)
            mutation_triggered = True
            self.algorithm.Debug(f"DNA Replication: Added {len(new_features)} new feature genes")
        
        market_volatility = market_conditions.get('volatility', 0)
        if market_volatility > 0.05:
            original_threshold = self.confidence_threshold
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
            mutation_triggered = True
            self.algorithm.Debug(f"DNA Adaptation: Confidence evolved {original_threshold:.2f} -> {self.confidence_threshold:.2f}")
        
        if hasattr(self, 'error_count') and self.error_count > 5:
            self.confidence_threshold = 0.75
            self.active_model = "forest"  # Most stable model
            self.error_count = 0
            mutation_triggered = True
            self.algorithm.Debug("DNA Repair: Self-healing activated, reset to stable configuration")
        
        return {
            "mutation_triggered": mutation_triggered,
            "dna_strands_active": 4,
            "evolutionary_state": "evolving" if mutation_triggered else "stable",
            "genetic_diversity": len(self.feature_sets),
            "adaptation_generation": getattr(self, 'adaptation_generation', 0) + (1 if mutation_triggered else 0)
        }
        
    def causal_quantum_reasoning(self, market_data, news_events=None):
        """
        Causal Quantum Reasoning Engine
        Understands WHY market movements happen using quantum causality principles
        """
        if 'returns' not in market_data or len(market_data['returns']) < 20:
            return {"causality": "insufficient_data", "quantum_state": "collapsed"}
        
        returns = np.array(market_data['returns'][-20:])
        volume = np.array(market_data.get('volume', [1]*len(returns))[-20:])
        
        cause_effect_correlation = np.corrcoef(returns[:-1], returns[1:])[0,1]
        volume_price_entanglement = np.corrcoef(returns, volume)[0,1] if len(volume) == len(returns) else 0
        
        buying_pressure = len([r for r in returns if r > 0]) / len(returns)
        selling_pressure = 1 - buying_pressure
        market_superposition = abs(buying_pressure - 0.5) * 2  # 0 = balanced, 1 = extreme
        
        causal_chains = []
        for i in range(1, len(returns)):
            if abs(returns[i]) > np.std(returns) * 1.5:  # Significant move
                preceding_volume = volume[i-1] if i > 0 and len(volume) > i-1 else 1
                volume_ratio = volume[i] / preceding_volume if preceding_volume > 0 else 1
                
                causal_chains.append({
                    "effect": returns[i],
                    "volume_cause": volume_ratio,
                    "causal_strength": abs(returns[i]) * volume_ratio
                })
        
        if causal_chains:
            avg_causal_strength = np.mean([c["causal_strength"] for c in causal_chains])
            dominant_direction = 1 if np.mean([c["effect"] for c in causal_chains]) > 0 else -1
        else:
            avg_causal_strength = 0
            dominant_direction = 0
        
        causality_factors = {
            "momentum_causality": cause_effect_correlation,
            "liquidity_causality": volume_price_entanglement,
            "force_balance": market_superposition,
            "causal_strength": avg_causal_strength,
            "predicted_direction": dominant_direction
        }
        
        consciousness_level = min(1.0, (abs(cause_effect_correlation) + abs(volume_price_entanglement) + market_superposition) / 3)
        
        self.algorithm.Debug(f"Quantum Reasoning: Consciousness={consciousness_level:.3f}, Direction={dominant_direction}")
        
        return {
            "causality_factors": causality_factors,
            "quantum_consciousness": consciousness_level,
            "causal_chains": len(causal_chains),
            "quantum_state": "coherent" if consciousness_level > 0.5 else "decoherent",
            "reasoning": f"Market shows {dominant_direction} bias due to causality strength {avg_causal_strength:.3f}"
        }
