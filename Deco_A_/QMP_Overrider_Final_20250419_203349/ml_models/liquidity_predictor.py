"""
Liquidity Predictor

This module provides a machine learning model for predicting market liquidity.
It uses historical order book data to predict future liquidity conditions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
from datetime import datetime, timedelta

class LiquidityPredictor:
    """
    Liquidity Predictor
    
    Provides a machine learning model for predicting market liquidity.
    It uses historical order book data to predict future liquidity conditions.
    """
    
    def __init__(self, model_type="random_forest", model_path=None):
        """
        Initialize Liquidity Predictor
        
        Parameters:
        - model_type: Type of model to use (random_forest or gradient_boosting)
        - model_path: Path to saved model
        """
        self.model_type = model_type
        self.model_path = model_path
        
        self.logger = self._setup_logger()
        self.logger.info(f"Initializing Liquidity Predictor with {model_type} model")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("LiquidityPredictor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_model(self):
        """Initialize model"""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_features(self, order_book_data, trades_data=None, market_data=None):
        """
        Extract features from order book data
        
        Parameters:
        - order_book_data: Order book data
        - trades_data: Trades data
        - market_data: Market data
        
        Returns:
        - Features
        """
        features = {}
        
        if "bids" in order_book_data and "asks" in order_book_data:
            bids = order_book_data["bids"]
            asks = order_book_data["asks"]
            
            best_bid = bids["price"].max() if not bids.empty else 0
            best_ask = asks["price"].min() if not asks.empty else 0
            
            if best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid
                spread_pct = spread / best_bid
                
                features["spread"] = spread
                features["spread_pct"] = spread_pct
            
            bid_volume = bids["quantity"].sum() if "quantity" in bids.columns else bids["volume"].sum()
            ask_volume = asks["quantity"].sum() if "quantity" in asks.columns else asks["volume"].sum()
            
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                features["imbalance"] = imbalance
            
            features["bid_depth"] = len(bids)
            features["ask_depth"] = len(asks)
            
            if not bids.empty:
                best_bid_volume = bids.loc[bids["price"] == best_bid, "quantity"].sum() if "quantity" in bids.columns else bids.loc[bids["price"] == best_bid, "volume"].sum()
                features["best_bid_volume"] = best_bid_volume
            
            if not asks.empty:
                best_ask_volume = asks.loc[asks["price"] == best_ask, "quantity"].sum() if "quantity" in asks.columns else asks.loc[asks["price"] == best_ask, "volume"].sum()
                features["best_ask_volume"] = best_ask_volume
        
        if trades_data is not None:
            features["recent_trade_volume"] = trades_data["volume"].sum()
            
            features["trade_frequency"] = len(trades_data)
            
            features["avg_trade_size"] = trades_data["volume"].mean()
            
            if "side" in trades_data.columns:
                buy_volume = trades_data.loc[trades_data["side"] == "buy", "volume"].sum()
                sell_volume = trades_data.loc[trades_data["side"] == "sell", "volume"].sum()
                
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    trade_direction = (buy_volume - sell_volume) / total_volume
                    features["trade_direction"] = trade_direction
        
        if market_data is not None:
            if "close" in market_data.columns:
                returns = market_data["close"].pct_change().dropna()
                features["volatility"] = returns.std()
            
            if "volume" in market_data.columns:
                volume_trend = market_data["volume"].pct_change().mean()
                features["volume_trend"] = volume_trend
            
            if "close" in market_data.columns:
                price_trend = market_data["close"].pct_change().mean()
                features["price_trend"] = price_trend
        
        now = datetime.now()
        features["hour"] = now.hour
        features["minute"] = now.minute
        features["day_of_week"] = now.weekday()
        
        return features
    
    def train(self, features_df, target_df):
        """
        Train model
        
        Parameters:
        - features_df: Features DataFrame
        - target_df: Target DataFrame
        
        Returns:
        - Training metrics
        """
        if features_df.empty or target_df.empty:
            self.logger.error("Empty training data")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target_df, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(zip(features_df.columns, self.model.feature_importances_))
        
        self.logger.info(f"Model trained with MSE: {mse:.4f}, R2: {r2:.4f}")
        
        return {
            "mse": mse,
            "r2": r2,
            "feature_importance": self.feature_importance
        }
    
    def predict(self, features):
        """
        Predict liquidity
        
        Parameters:
        - features: Features
        
        Returns:
        - Predicted liquidity
        """
        if self.model is None:
            self.logger.error("Model not trained")
            return None
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)
        
        return prediction[0]
    
    def save_model(self, path):
        """
        Save model
        
        Parameters:
        - path: Path to save model
        
        Returns:
        - True if successful, False otherwise
        """
        if self.model is None:
            self.logger.error("Model not trained")
            return False
        
        try:
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_importance": self.feature_importance,
                "model_type": self.model_type
            }
            
            joblib.dump(model_data, path)
            
            self.logger.info(f"Model saved to {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path):
        """
        Load model
        
        Parameters:
        - path: Path to load model from
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            model_data = joblib.load(path)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_importance = model_data["feature_importance"]
            self.model_type = model_data["model_type"]
            
            self.logger.info(f"Model loaded from {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self):
        """
        Get feature importance
        
        Returns:
        - Feature importance
        """
        return self.feature_importance
