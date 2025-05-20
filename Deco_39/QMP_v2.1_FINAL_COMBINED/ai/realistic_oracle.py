"""
Realistic Oracle

A realistic implementation of price prediction using statistical methods and machine learning
instead of fictional quantum technologies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import talib as ta
import datetime

class RealisticOracle:
    """
    A realistic implementation of price prediction using statistical methods and machine learning.
    Replaces the fictional QuantumOracle with practical techniques.
    """
    
    def __init__(self, lookback_periods=30, prediction_horizon=5, confidence_threshold=0.7):
        """
        Initialize the RealisticOracle
        
        Parameters:
        - lookback_periods: Number of periods to use for feature generation
        - prediction_horizon: Number of periods to predict into the future
        - confidence_threshold: Threshold for prediction confidence
        """
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.is_trained = False
        self.feature_importance = {}
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_history = []
        self.market_regimes = {}
    
    def _generate_features(self, price_data):
        """
        Generate features from price data
        
        Parameters:
        - price_data: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with features
        """
        df = price_data.copy()
        
        if len(df) >= 14:
            df['rsi'] = ta.RSI(df['close'].values, timeperiod=14)
            df['cci'] = ta.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['mfi'] = ta.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)
            
            df['adx'] = ta.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'].values)
            
            df['atr'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.BBANDS(df['close'].values)
            
            df['obv'] = ta.OBV(df['close'].values, df['volume'].values)
            df['ad'] = ta.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        df['price_sma_5_ratio'] = df['close'] / df['sma_5']
        df['price_sma_10_ratio'] = df['close'] / df['sma_10']
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']
        
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body'] / df['range']
        df['upper_shadow'] = df['high'] - df['close'].where(df['close'] >= df['open'], df['open'])
        df['lower_shadow'] = df['close'].where(df['close'] <= df['open'], df['open']) - df['low']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['range']
        
        if 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_month'] = df['timestamp'].dt.day
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        df = df.dropna()
        
        return df
    
    def _prepare_data(self, price_data):
        """
        Prepare data for training
        
        Parameters:
        - price_data: DataFrame with OHLCV data
        
        Returns:
        - X: Features
        - y: Target
        """
        df = self._generate_features(price_data)
        
        df['target'] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y
    
    def train(self, price_data):
        """
        Train the model
        
        Parameters:
        - price_data: DataFrame with OHLCV data
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            X, y = self._prepare_data(price_data)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            y_train_reshaped = y_train.values.reshape(-1, 1)
            y_test_reshaped = y_test.values.reshape(-1, 1)
            self.target_scaler.fit(y_train_reshaped)
            y_train_scaled = self.target_scaler.transform(y_train_reshaped).ravel()
            y_test_scaled = self.target_scaler.transform(y_test_reshaped).ravel()
            
            self.rf_model.fit(X_train_scaled, y_train_scaled)
            self.gb_model.fit(X_train_scaled, y_train_scaled)
            
            rf_pred = self.rf_model.predict(X_test_scaled)
            gb_pred = self.gb_model.predict(X_test_scaled)
            
            rf_mse = mean_squared_error(y_test_scaled, rf_pred)
            gb_mse = mean_squared_error(y_test_scaled, gb_pred)
            
            self.feature_importance = dict(zip(X.columns, self.rf_model.feature_importances_))
            
            self.is_trained = True
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_move(self, symbol, price_data):
        """
        Predict the next price movement
        
        Parameters:
        - symbol: Symbol to predict
        - price_data: DataFrame with OHLCV data
        
        Returns:
        - Dictionary with prediction details
        """
        if not self.is_trained:
            return {
                "price": None,
                "time": datetime.datetime.now() + datetime.timedelta(minutes=self.prediction_horizon),
                "certainty": 0.0,
                "direction": "UNKNOWN"
            }
        
        try:
            df = self._generate_features(price_data)
            
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
            
            X_latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
            
            X_latest_scaled = self.feature_scaler.transform(X_latest)
            
            rf_pred_scaled = self.rf_model.predict(X_latest_scaled)
            gb_pred_scaled = self.gb_model.predict(X_latest_scaled)
            
            ensemble_pred_scaled = (rf_pred_scaled + gb_pred_scaled) / 2
            
            ensemble_pred = self.target_scaler.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).ravel()[0]
            
            current_price = price_data['close'].iloc[-1]
            predicted_price = current_price * (1 + ensemble_pred)
            
            prediction_time = datetime.datetime.now() + datetime.timedelta(minutes=self.prediction_horizon)
            
            model_agreement = 1 - abs(rf_pred_scaled[0] - gb_pred_scaled[0]) / max(abs(rf_pred_scaled[0]), abs(gb_pred_scaled[0]))
            certainty = min(0.95, model_agreement)  # Cap at 95% to avoid overconfidence
            
            direction = "BUY" if ensemble_pred > 0 else "SELL"
            
            prediction = {
                "price": predicted_price,
                "time": prediction_time,
                "certainty": certainty,
                "direction": direction
            }
            
            self.last_prediction = prediction
            self.last_confidence = certainty
            self.prediction_history.append(prediction)
            
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return prediction
        except Exception as e:
            print(f"Error predicting next move: {e}")
            return {
                "price": None,
                "time": datetime.datetime.now() + datetime.timedelta(minutes=self.prediction_horizon),
                "certainty": 0.0,
                "direction": "UNKNOWN"
            }
    
    def get_market_regime(self, symbol, price_data):
        """
        Determine the current market regime
        
        Parameters:
        - symbol: Symbol to analyze
        - price_data: DataFrame with OHLCV data
        
        Returns:
        - Dictionary with market regime details
        """
        try:
            df = self._generate_features(price_data)
            
            volatility = df['volatility'].iloc[-1]
            
            adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0
            
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            
            if volatility > 0.02:  # High volatility
                if adx > 25:  # Strong trend
                    if rsi > 70:
                        regime = "STRONG_UPTREND"
                    elif rsi < 30:
                        regime = "STRONG_DOWNTREND"
                    else:
                        regime = "VOLATILE_TREND"
                else:  # Weak trend
                    regime = "CHOPPY"
            else:  # Low volatility
                if adx > 25:  # Strong trend
                    if rsi > 70:
                        regime = "STEADY_UPTREND"
                    elif rsi < 30:
                        regime = "STEADY_DOWNTREND"
                    else:
                        regime = "STEADY_TREND"
                else:  # Weak trend
                    regime = "RANGING"
            
            self.market_regimes[symbol] = {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": adx,
                "momentum": rsi,
                "timestamp": datetime.datetime.now()
            }
            
            return self.market_regimes[symbol]
        except Exception as e:
            print(f"Error determining market regime: {e}")
            return {
                "regime": "UNKNOWN",
                "volatility": 0,
                "trend_strength": 0,
                "momentum": 50,
                "timestamp": datetime.datetime.now()
            }
    
    def get_prediction_accuracy(self):
        """
        Calculate prediction accuracy based on historical predictions
        
        Returns:
        - Dictionary with accuracy metrics
        """
        if not self.prediction_history:
            return {
                "accuracy": 0.0,
                "direction_accuracy": 0.0,
                "mse": 0.0,
                "sample_size": 0
            }
        
        correct_direction = 0
        squared_errors = []
        
        for i, prediction in enumerate(self.prediction_history[:-1]):
            if prediction["direction"] == self.prediction_history[i+1]["direction"]:
                correct_direction += 1
            
            if prediction["price"] is not None and self.prediction_history[i+1]["price"] is not None:
                squared_error = (prediction["price"] - self.prediction_history[i+1]["price"]) ** 2
                squared_errors.append(squared_error)
        
        sample_size = len(self.prediction_history) - 1
        direction_accuracy = correct_direction / sample_size if sample_size > 0 else 0
        mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0
        accuracy = 1 - min(1, mse / 100)  # Normalize MSE to 0-1 range
        
        return {
            "accuracy": accuracy,
            "direction_accuracy": direction_accuracy,
            "mse": mse,
            "sample_size": sample_size
        }
