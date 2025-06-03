"""
Institutional-grade indicators for quantitative trading
"""

import numpy as np
import pandas as pd
import warnings
import logging

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. HestonVolatility will have limited functionality.")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. ML_RSI will have limited functionality.")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. RegimeDetector will have limited functionality.")

class HestonVolatility:
    """
    Implements Heston stochastic volatility model on top of existing price data
    Requires: Close prices (as pandas Series)
    Returns: Implied volatility surface and model parameters
    """
    
    def __init__(self, lookback: int = 30, risk_free: float = 0.01):
        self.lookback = lookback
        self.r = risk_free
        self.logger = logging.getLogger('HestonVolatility')
        
    def heston_objective(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Objective function for Heston model calibration"""
        kappa, theta, sigma, rho, v0 = params
        
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
            return 1e6
        if abs(rho) >= 1:
            return 1e6
        if 2 * kappa * theta <= sigma**2:
            return 1e6
            
        dt = 1/252
        n = len(returns)
        
        v = np.zeros(n)
        v[0] = v0
        
        log_likelihood = 0
        
        for i in range(1, n):
            v_prev = v[i-1]
            
            v[i] = max(v_prev + kappa * (theta - v_prev) * dt + 
                      sigma * np.sqrt(v_prev * dt) * np.random.normal(), 1e-6)
            
            expected_return = self.r * dt
            variance = v_prev * dt
            
            if variance > 0:
                log_likelihood -= 0.5 * np.log(2 * np.pi * variance)
                log_likelihood -= 0.5 * (returns[i] - expected_return)**2 / variance
        
        return -log_likelihood
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate Heston volatility for given price series"""
        if len(prices) < self.lookback:
            return pd.Series(index=prices.index, dtype=float)
        
        returns = np.log(prices / prices.shift(1))
        returns = returns.dropna()
        
        if not SCIPY_AVAILABLE:
            return returns.rolling(window=self.lookback).std() * np.sqrt(252)
        
        volatilities = []
        
        for i in range(self.lookback, len(returns)):
            window_returns = returns.iloc[i-self.lookback:i].values
            
            initial_guess = [2.0, 0.04, 0.3, -0.7, 0.04]
            bounds = [(0.1, 10), (0.01, 1), (0.1, 2), (-0.99, 0.99), (0.01, 1)]
            
            try:
                result = minimize(
                    self.heston_objective,
                    initial_guess,
                    args=(window_returns,),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    kappa, theta, sigma, rho, v0 = result.x
                    vol = np.sqrt(theta)
                else:
                    vol = np.std(window_returns) * np.sqrt(252)
                    
            except Exception as e:
                self.logger.warning(f"Heston calibration failed: {e}")
                vol = np.std(window_returns) * np.sqrt(252)
            
            volatilities.append(vol)
        
        vol_series = pd.Series(
            volatilities,
            index=returns.index[self.lookback:],
            name='heston_volatility'
        )
        
        return vol_series.reindex(prices.index)

class ML_RSI:
    """
    Machine Learning enhanced RSI using gradient boosting
    Incorporates volume, volatility, and momentum features
    """
    
    def __init__(self, window: int = 14, lookahead: int = 5):
        self.window = window
        self.lookahead = lookahead
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.logger = logging.getLogger('ML_RSI')
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
    
    def _calculate_features(self, prices: pd.Series, volume: pd.Series = None) -> pd.DataFrame:
        """Calculate technical features for ML model"""
        features = pd.DataFrame(index=prices.index)
        
        returns = prices.pct_change()
        
        features['rsi'] = self._traditional_rsi(prices)
        features['price_momentum'] = prices / prices.shift(5) - 1
        features['volatility'] = returns.rolling(window=10).std()
        features['price_position'] = (prices - prices.rolling(20).min()) / (
            prices.rolling(20).max() - prices.rolling(20).min())
        
        if volume is not None:
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            features['price_volume'] = returns * volume
        else:
            features['volume_ratio'] = 1.0
            features['price_volume'] = returns
        
        features['ma_ratio'] = prices / prices.rolling(20).mean()
        features['bb_position'] = (prices - prices.rolling(20).mean()) / (
            2 * prices.rolling(20).std())
        
        return features.ffill().fillna(0)
    
    def _traditional_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate traditional RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fit(self, prices: pd.Series, volume: pd.Series = None):
        """Fit the ML model on historical data"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available. Using traditional RSI.")
            return
        
        features = self._calculate_features(prices, volume)
        
        target = prices.shift(-self.lookahead).pct_change(self.lookahead)
        
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        
        if valid_idx.sum() < 50:
            self.logger.warning("Insufficient data for ML training")
            return
        
        X = features[valid_idx]
        y = target[valid_idx]
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        self.logger.info(f"ML_RSI model fitted with {len(X)} samples")
    
    def calculate(self, prices: pd.Series, volume: pd.Series = None) -> pd.Series:
        """Calculate ML-enhanced RSI"""
        traditional_rsi = self._traditional_rsi(prices)
        
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return traditional_rsi
        
        features = self._calculate_features(prices, volume)
        
        try:
            X_scaled = self.scaler.transform(features.ffill().fillna(0))
            predictions = self.model.predict(X_scaled)
            
            ml_adjustment = pd.Series(predictions, index=prices.index)
            ml_adjustment = (ml_adjustment - ml_adjustment.mean()) / ml_adjustment.std() * 10
            
            enhanced_rsi = traditional_rsi + ml_adjustment
            enhanced_rsi = np.clip(enhanced_rsi, 0, 100)
            
            return enhanced_rsi
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return traditional_rsi

class OrderFlowImbalance:
    """
    Order flow imbalance indicator using volume and price action
    Estimates institutional vs retail flow
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.logger = logging.getLogger('OrderFlowImbalance')
    
    def calculate(self, prices: pd.Series, volume: pd.Series, 
                 high: pd.Series = None, low: pd.Series = None) -> pd.Series:
        """Calculate order flow imbalance"""
        
        if high is None:
            high = prices
        if low is None:
            low = prices
        
        returns = prices.pct_change()
        
        typical_price = (high + low + prices) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(returns > 0, 0)
        negative_flow = money_flow.where(returns < 0, 0)
        
        positive_flow_ma = positive_flow.rolling(window=self.window).sum()
        negative_flow_ma = negative_flow.rolling(window=self.window).sum()
        
        total_flow = positive_flow_ma + negative_flow_ma
        
        flow_ratio = np.where(
            total_flow != 0,
            (positive_flow_ma - negative_flow_ma) / total_flow,
            0
        )
        
        large_volume_threshold = volume.rolling(window=50).quantile(0.8)
        large_volume_mask = volume > large_volume_threshold
        
        institutional_flow = money_flow.where(large_volume_mask, 0)
        retail_flow = money_flow.where(~large_volume_mask, 0)
        
        inst_flow_ma = institutional_flow.rolling(window=self.window).sum()
        retail_flow_ma = retail_flow.rolling(window=self.window).sum()
        
        total_inst_retail = inst_flow_ma + retail_flow_ma
        
        institutional_ratio = np.where(
            total_inst_retail != 0,
            inst_flow_ma / total_inst_retail,
            0.5
        )
        
        imbalance = pd.Series(
            flow_ratio * institutional_ratio,
            index=prices.index,
            name='order_flow_imbalance'
        )
        
        return imbalance

class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models
    Identifies bull, bear, and sideways market conditions
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.is_fitted = False
        self.regime_names = ['Bear', 'Sideways', 'Bull']
        self.logger = logging.getLogger('RegimeDetector')
        
        if HMM_AVAILABLE:
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                random_state=42
            )
    
    def _prepare_features(self, prices: pd.Series) -> np.ndarray:
        """Prepare features for regime detection"""
        returns = prices.pct_change().dropna()
        
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(window=20).std(),
            'momentum': prices.pct_change(20),
            'trend': prices.rolling(50).mean() / prices.rolling(200).mean() - 1
        }).ffill().fillna(0)
        
        return features.values
    
    def fit(self, prices: pd.Series):
        """Fit the regime detection model"""
        if not HMM_AVAILABLE:
            self.logger.warning("hmmlearn not available. Using simple regime detection.")
            return
        
        features = self._prepare_features(prices)
        
        if len(features) < 100:
            self.logger.warning("Insufficient data for regime detection")
            return
        
        try:
            self.model.fit(features)
            self.is_fitted = True
            self.logger.info(f"Regime detector fitted with {len(features)} observations")
        except Exception as e:
            self.logger.error(f"Failed to fit regime model: {e}")
    
    def predict(self, prices: pd.Series) -> pd.Series:
        """Predict market regimes"""
        if not HMM_AVAILABLE or not self.is_fitted:
            return self._simple_regime_detection(prices)
        
        features = self._prepare_features(prices)
        
        try:
            regimes = self.model.predict(features)
            
            regime_series = pd.Series(
                regimes,
                index=prices.index[1:],
                name='market_regime'
            )
            
            return regime_series.reindex(prices.index).ffill()
            
        except Exception as e:
            self.logger.warning(f"Regime prediction failed: {e}")
            return self._simple_regime_detection(prices)
    
    def _simple_regime_detection(self, prices: pd.Series) -> pd.Series:
        """Simple regime detection fallback"""
        returns = prices.pct_change()
        volatility = returns.rolling(window=20).std()
        momentum = prices.pct_change(20)
        
        regimes = np.zeros(len(prices))
        
        bear_mask = (momentum < -0.05) | (volatility > volatility.quantile(0.8))
        bull_mask = (momentum > 0.05) & (volatility < volatility.quantile(0.6))
        
        regimes[bear_mask] = 0
        regimes[bull_mask] = 2
        regimes[~(bear_mask | bull_mask)] = 1
        
        return pd.Series(regimes, index=prices.index, name='market_regime')
    
    def get_regime_probabilities(self, prices: pd.Series) -> pd.DataFrame:
        """Get regime probabilities for each time period"""
        if not HMM_AVAILABLE or not self.is_fitted:
            regimes = self._simple_regime_detection(prices)
            probs = pd.DataFrame(index=prices.index)
            for i in range(self.n_regimes):
                probs[f'regime_{i}_prob'] = (regimes == i).astype(float)
            return probs
        
        features = self._prepare_features(prices)
        
        try:
            log_probs = self.model.predict_proba(features)
            probs = np.exp(log_probs)
            
            prob_df = pd.DataFrame(
                probs,
                index=prices.index[1:],
                columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
            )
            
            return prob_df.reindex(prices.index).ffill().fillna(1/self.n_regimes)
            
        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            regimes = self._simple_regime_detection(prices)
            probs = pd.DataFrame(index=prices.index)
            for i in range(self.n_regimes):
                probs[f'regime_{i}_prob'] = (regimes == i).astype(float)
            return probs
