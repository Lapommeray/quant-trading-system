"""
Dark Pool Liquidity Predictor Module

This module implements the Dark Pool Liquidity Predictor for the QMP Overrider system.
It uses LightGBM to predict liquidity in dark pools and determine optimal order sizing.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import time

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class LiquidityPredictor:
    """
    Dark Pool Liquidity Predictor for the QMP Overrider system.
    
    This class uses LightGBM to predict liquidity in dark pools and determine
    optimal order sizing based on predicted liquidity.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the Liquidity Predictor.
        
        Parameters:
            model_dir: Directory to store model files (or None for default)
        """
        self.logger = logging.getLogger("LiquidityPredictor")
        
        if model_dir is None:
            self.model_dir = Path("models/dark_pool")
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.prediction_history = []
        
        if LIGHTGBM_AVAILABLE:
            self._load_model()
        else:
            self.logger.warning("LightGBM not available, using fallback prediction")
        
        self.logger.info("Dark Pool Liquidity Predictor initialized")
    
    def _load_model(self):
        """Load model from file if available."""
        model_file = self.model_dir / "liquidity_lgb.txt"
        scaler_file = self.model_dir / "liquidity_scaler.bin"
        
        if model_file.exists() and scaler_file.exists():
            try:
                self.model = lgb.Booster(model_file=str(model_file))
                self.scaler = joblib.load(scaler_file)
                self.logger.info(f"Loaded model from {model_file}")
                
                if hasattr(self.model, 'feature_importance'):
                    feature_names = self.model.feature_name()
                    importance = self.model.feature_importance()
                    self.feature_importance = dict(zip(feature_names, importance))
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self._train_default_model()
        else:
            self.logger.info("No existing model found, training default model")
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a default model with synthetic data."""
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM not available, cannot train default model")
            return
        
        self.logger.info("Training default model with synthetic data")
        
        X, y = self._generate_synthetic_data(1000)
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=100)
        
        self.model.save_model(str(self.model_dir / "liquidity_lgb.txt"))
        joblib.dump(self.scaler, self.model_dir / "liquidity_scaler.bin")
        
        feature_names = self.model.feature_name()
        importance = self.model.feature_importance()
        self.feature_importance = dict(zip(feature_names, importance))
        
        self.logger.info("Default model trained and saved")
    
    def _generate_synthetic_data(self, n_samples):
        """
        Generate synthetic data for training.
        
        Parameters:
            n_samples: Number of samples to generate
            
        Returns:
            X, y: Features and target
        """
        feature_names = [
            'time_of_day_sin', 'time_of_day_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'recent_fill_rate', 'recent_fill_rate_std',
            'market_volatility', 'spread_bps',
            'order_book_imbalance', 'trade_size_normalized',
            'pool_latency_ms', 'pool_recent_volume',
            'pool_recent_trades', 'pool_recent_rejections',
            'market_regime', 'economic_surprise_index',
            'vix_level', 'sector_flow',
            'retail_sentiment', 'institutional_flow'
        ]
        
        X = np.random.randn(n_samples, len(feature_names))
        
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            time_effect = 0.2 * X[i, 0] + 0.3 * X[i, 1]
            
            fill_effect = 0.4 * X[i, 4]
            
            vol_effect = -0.3 * X[i, 6]
            
            imbalance_effect = -0.2 * X[i, 8]
            
            pool_effect = 0.3 * X[i, 10] + 0.2 * X[i, 11]
            
            regime_effect = 0.4 * X[i, 14]
            
            vix_effect = -0.3 * X[i, 16]
            
            y[i] = 0.7 + 0.1 * (
                time_effect + fill_effect + vol_effect + 
                imbalance_effect + pool_effect + regime_effect + vix_effect
            )
            
            y[i] += np.random.normal(0, 0.05)
            
            y[i] = max(0.1, min(1.0, y[i]))
        
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y
    
    def predict_liquidity(self, features):
        """
        Predict liquidity for a dark pool.
        
        Parameters:
            features: Dictionary of features
            
        Returns:
            Predicted liquidity (0.0 to 1.0)
        """
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        required_features = [
            'time_of_day_sin', 'time_of_day_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'recent_fill_rate', 'recent_fill_rate_std',
            'market_volatility', 'spread_bps',
            'order_book_imbalance', 'trade_size_normalized',
            'pool_latency_ms', 'pool_recent_volume',
            'pool_recent_trades', 'pool_recent_rejections',
            'market_regime', 'economic_surprise_index',
            'vix_level', 'sector_flow',
            'retail_sentiment', 'institutional_flow'
        ]
        
        for feature in required_features:
            if feature not in features_df.columns:
                features_df[feature] = 0.0
        
        if LIGHTGBM_AVAILABLE and self.model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(features_df[required_features])
            
            prediction = self.model.predict(X_scaled)[0]
        else:
            prediction = self._fallback_prediction(features_df)
        
        self._record_prediction(features_df, prediction)
        
        return prediction
    
    def _fallback_prediction(self, features_df):
        """
        Fallback prediction when model is not available.
        
        Parameters:
            features_df: DataFrame of features
            
        Returns:
            Predicted liquidity (0.0 to 1.0)
        """
        prediction = 0.7  # Base liquidity
        
        time_sin = features_df['time_of_day_sin'].values[0]
        time_cos = features_df['time_of_day_cos'].values[0]
        time_effect = 0.1 * (time_sin + time_cos)
        
        if 'recent_fill_rate' in features_df:
            fill_rate = features_df['recent_fill_rate'].values[0]
            fill_effect = 0.2 * (fill_rate - 0.5)
        else:
            fill_effect = 0
        
        if 'market_volatility' in features_df:
            volatility = features_df['market_volatility'].values[0]
            vol_effect = -0.2 * (volatility - 0.5)
        else:
            vol_effect = 0
        
        if 'pool_latency_ms' in features_df:
            latency = features_df['pool_latency_ms'].values[0]
            latency_effect = -0.1 * (latency / 100)
        else:
            latency_effect = 0
        
        prediction += time_effect + fill_effect + vol_effect + latency_effect
        
        prediction = max(0.1, min(1.0, prediction))
        
        return prediction
    
    def _record_prediction(self, features_df, prediction):
        """
        Record a prediction for analysis.
        
        Parameters:
            features_df: DataFrame of features
            prediction: Predicted liquidity
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': float(prediction),
            'features': {col: float(features_df[col].values[0]) for col in features_df.columns}
        }
        
        self.prediction_history.append(record)
        
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        if len(self.prediction_history) % 100 == 0:
            self._save_prediction_history()
    
    def _save_prediction_history(self):
        """Save prediction history to file."""
        history_file = self.model_dir / "prediction_history.json"
        
        try:
            with open(history_file, "w") as f:
                json.dump(self.prediction_history, f, indent=2)
            
            self.logger.debug(f"Saved prediction history to {history_file}")
        except Exception as e:
            self.logger.error(f"Error saving prediction history: {e}")
    
    def get_optimal_sizing(self, prediction, order_size, risk_tolerance=0.5):
        """
        Get optimal order sizing based on predicted liquidity.
        
        Parameters:
            prediction: Predicted liquidity (0.0 to 1.0)
            order_size: Original order size
            risk_tolerance: Risk tolerance (0.0 to 1.0)
            
        Returns:
            Optimal order size
        """
        if prediction > 0.8:
            sizing_factor = 1.0
        elif prediction > 0.6:
            sizing_factor = 0.7 + 0.3 * risk_tolerance
        elif prediction > 0.4:
            sizing_factor = 0.4 + 0.3 * risk_tolerance
        elif prediction > 0.2:
            sizing_factor = 0.2 + 0.2 * risk_tolerance
        else:
            sizing_factor = 0.1 * risk_tolerance
        
        optimal_size = order_size * sizing_factor
        
        return optimal_size
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary of feature importance
        """
        return self.feature_importance
    
    def encode_time_features(self, timestamp=None):
        """
        Encode time features for prediction.
        
        Parameters:
            timestamp: Timestamp to encode (or None for now)
            
        Returns:
            Dictionary of encoded time features
        """
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        
        hours = timestamp.hour + timestamp.minute / 60
        time_of_day_sin = np.sin(2 * np.pi * hours / 24)
        time_of_day_cos = np.cos(2 * np.pi * hours / 24)
        
        day = timestamp.weekday()
        day_of_week_sin = np.sin(2 * np.pi * day / 7)
        day_of_week_cos = np.cos(2 * np.pi * day / 7)
        
        return {
            'time_of_day_sin': time_of_day_sin,
            'time_of_day_cos': time_of_day_cos,
            'day_of_week_sin': day_of_week_sin,
            'day_of_week_cos': day_of_week_cos
        }
    
    def train(self, X, y):
        """
        Train the model with new data.
        
        Parameters:
            X: Features DataFrame
            y: Target array
            
        Returns:
            Training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM not available, cannot train model")
            return None
        
        self.logger.info(f"Training model with {len(X)} samples")
        
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        train_data = lgb.Dataset(X_scaled, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=100)
        
        self.model.save_model(str(self.model_dir / "liquidity_lgb.txt"))
        joblib.dump(self.scaler, self.model_dir / "liquidity_scaler.bin")
        
        feature_names = self.model.feature_name()
        importance = self.model.feature_importance()
        self.feature_importance = dict(zip(feature_names, importance))
        
        self.logger.info("Model trained and saved")
        
        return {
            'feature_importance': self.feature_importance
        }
    
    def generate_liquidity_report(self, pool_name, start_time=None, end_time=None):
        """
        Generate a liquidity report for a dark pool.
        
        Parameters:
            pool_name: Name of the dark pool
            start_time: Start time for the report (or None for all time)
            end_time: End time for the report (or None for now)
            
        Returns:
            Liquidity report as a string
        """
        if start_time is None:
            start_time = datetime.fromtimestamp(0)
        elif isinstance(start_time, (int, float)):
            start_time = datetime.fromtimestamp(start_time)
        
        if end_time is None:
            end_time = datetime.now()
        elif isinstance(end_time, (int, float)):
            end_time = datetime.fromtimestamp(end_time)
        
        filtered_predictions = []
        for record in self.prediction_history:
            try:
                record_time = datetime.fromisoformat(record['timestamp'])
                if start_time <= record_time <= end_time:
                    filtered_predictions.append(record)
            except (ValueError, TypeError):
                pass
        
        report = f"""
DARK POOL LIQUIDITY REPORT - {pool_name}
======================================
Period: {start_time.isoformat()} to {end_time.isoformat()}
Predictions: {len(filtered_predictions)}

Liquidity Summary:
- Average Predicted Liquidity: {np.mean([p['prediction'] for p in filtered_predictions]):.2f}
- Min Predicted Liquidity: {min([p['prediction'] for p in filtered_predictions]):.2f}
- Max Predicted Liquidity: {max([p['prediction'] for p in filtered_predictions]):.2f}

Key Factors Affecting Liquidity:
"""
        
        if self.feature_importance:
            sorted_importance = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, importance in sorted_importance[:5]:
                report += f"- {feature}: {importance}\n"
        
        report += f"\nOptimal Trading Times:\n"
        
        if filtered_predictions:
            time_data = []
            for record in filtered_predictions:
                try:
                    record_time = datetime.fromisoformat(record['timestamp'])
                    hour = record_time.hour
                    day = record_time.weekday()
                    time_data.append((hour, day, record['prediction']))
                except (ValueError, TypeError):
                    pass
            
            hour_liquidity = defaultdict(list)
            for hour, _, prediction in time_data:
                hour_liquidity[hour].append(prediction)
            
            best_hours = sorted(
                [(hour, np.mean(predictions)) for hour, predictions in hour_liquidity.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for hour, liquidity in best_hours[:3]:
                report += f"- {hour:02d}:00 - {hour+1:02d}:00: {liquidity:.2f}\n"
            
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_liquidity = defaultdict(list)
            for _, day, prediction in time_data:
                day_liquidity[day].append(prediction)
            
            best_days = sorted(
                [(day, np.mean(predictions)) for day, predictions in day_liquidity.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            report += f"\nBest Days for Liquidity:\n"
            for day, liquidity in best_days[:3]:
                report += f"- {day_names[day]}: {liquidity:.2f}\n"
        
        report += f"\nReport generated at: {datetime.now().isoformat()}"
        
        return report

class DarkPoolOracle:
    """
    Dark Pool Oracle for the QMP Overrider system.
    
    This class combines the Liquidity Predictor and Dark Pool Router
    to provide optimal dark pool access.
    """
    
    def __init__(self):
        """Initialize the Dark Pool Oracle."""
        self.logger = logging.getLogger("DarkPoolOracle")
        
        self.predictor = LiquidityPredictor()
        
        try:
            from dark_pool.failover import DarkPoolRouter
            self.router = DarkPoolRouter()
            self.router_available = True
        except ImportError:
            self.logger.warning("Dark Pool Router not available")
            self.router_available = False
        
        self.logger.info("Dark Pool Oracle initialized")
    
    def get_optimal_execution(self, order, risk_tolerance=0.5):
        """
        Get optimal execution parameters for a dark pool order.
        
        Parameters:
            order: Order details
            risk_tolerance: Risk tolerance (0.0 to 1.0)
            
        Returns:
            Execution parameters
        """
        time_features = self.predictor.encode_time_features()
        
        pool_features = {}
        if self.router_available:
            self.router.check_pool_health()
            
            for pool_name, pool_data in self.router.pools.items():
                pool_features[pool_name] = {
                    'recent_fill_rate': pool_data['fill_rate'],
                    'recent_fill_rate_std': 0.05,  # Placeholder
                    'pool_latency_ms': pool_data['latency'],
                    'pool_recent_volume': 1000,  # Placeholder
                    'pool_recent_trades': 100,  # Placeholder
                    'pool_recent_rejections': 5  # Placeholder
                }
        else:
            pool_features = {
                'pool_alpha': {
                    'recent_fill_rate': 0.95,
                    'recent_fill_rate_std': 0.05,
                    'pool_latency_ms': 15,
                    'pool_recent_volume': 1000,
                    'pool_recent_trades': 100,
                    'pool_recent_rejections': 5
                },
                'pool_sigma': {
                    'recent_fill_rate': 0.90,
                    'recent_fill_rate_std': 0.07,
                    'pool_latency_ms': 12,
                    'pool_recent_volume': 800,
                    'pool_recent_trades': 80,
                    'pool_recent_rejections': 8
                },
                'pool_omega': {
                    'recent_fill_rate': 0.85,
                    'recent_fill_rate_std': 0.10,
                    'pool_latency_ms': 10,
                    'pool_recent_volume': 600,
                    'pool_recent_trades': 60,
                    'pool_recent_rejections': 12
                }
            }
        
        market_features = {
            'market_volatility': order.get('market_volatility', 0.1),
            'spread_bps': order.get('spread_bps', 5),
            'order_book_imbalance': order.get('order_book_imbalance', 1.0),
            'trade_size_normalized': order.get('size', 100) / 1000,
            'market_regime': order.get('market_regime', 0.5),
            'economic_surprise_index': order.get('economic_surprise_index', 0),
            'vix_level': order.get('vix_level', 15) / 100,
            'sector_flow': order.get('sector_flow', 0),
            'retail_sentiment': order.get('retail_sentiment', 0.5),
            'institutional_flow': order.get('institutional_flow', 0)
        }
        
        pool_liquidity = {}
        for pool_name, pool_data in pool_features.items():
            features = {**time_features, **pool_data, **market_features}
            
            liquidity = self.predictor.predict_liquidity(features)
            
            optimal_size = self.predictor.get_optimal_sizing(
                liquidity,
                order.get('size', 100),
                risk_tolerance
            )
            
            pool_liquidity[pool_name] = {
                'liquidity': liquidity,
                'optimal_size': optimal_size
            }
        
        best_pool = max(pool_liquidity.items(), key=lambda x: x[1]['liquidity'])
        
        execution_result = None
        if self.router_available:
            routed_order = {**order, 'size': best_pool[1]['optimal_size']}
            
            execution_result = self.router.route_order(routed_order, force_pool=best_pool[0])
        
        return {
            'pool_liquidity': pool_liquidity,
            'selected_pool': best_pool[0],
            'predicted_liquidity': best_pool[1]['liquidity'],
            'optimal_size': best_pool[1]['optimal_size'],
            'execution_result': execution_result,
            'timestamp': datetime.now().isoformat()
        }
