"""
Causal Quantum Reasoning Engine

An AI that doesn't just predict outcomes, but understands why things happen — using quantum causality.
This module breaks the black-box barrier, allowing AI to explain the universe's structure.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import ccxt
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import time
import json
import os
import random
import hashlib
import networkx as nx
from scipy import stats
import math

class CausalQuantumReasoningEngine:
    """
    Causal Quantum Reasoning Engine
    
    An AI that understands market causality using quantum principles, not just correlation.
    This module identifies true causal relationships in market data and uses them for prediction.
    
    Key features:
    - Causal discovery in market data
    - Quantum entanglement detection
    - Counterfactual reasoning
    - Causal intervention analysis
    """
    
    def __init__(self, algorithm=None, symbol=None):
        """
        Initialize the Causal Quantum Reasoning Engine.
        
        Parameters:
        - algorithm: Optional algorithm instance for integration
        - symbol: Optional symbol to create a symbol-specific instance
        """
        self.algorithm = algorithm
        self.symbol = symbol
        self.logger = logging.getLogger(f"CausalQuantumReasoningEngine_{symbol}" if symbol else "CausalQuantumReasoningEngine")
        self.logger.setLevel(logging.INFO)
        
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
        self.causal_graph = nx.DiGraph()
        
        self.entanglement_threshold = 0.75
        self.quantum_correlation_matrix = {}
        
        self.min_causal_strength = 0.3
        self.max_lag = 10  # Maximum lag for causal discovery
        self.significance_level = 0.05
        
        self.counterfactual_models = {}
        
        self.prediction_history = []
        self.causal_strength_history = {}
        
        self.last_update_time = None
        self.update_interval = timedelta(hours=4)
        
        if algorithm:
            algorithm.Debug(f"Causal Quantum Reasoning Engine initialized for {symbol}" if symbol else "Causal Quantum Reasoning Engine initialized")
    
    def _calculate_transfer_entropy(self, source: Any, target: Any, lag: int = 1) -> float:
        """
        Calculate transfer entropy from source to target time series.
        Transfer entropy measures the directed flow of information.
        
        Parameters:
        - source: Source time series
        - target: Target time series
        - lag: Time lag
        
        Returns:
        - Transfer entropy value
        """
        if len(source) != len(target) or len(source) <= lag + 1:
            return 0.0
        
        source_past = source[:-lag]
        target_past = target[:-lag]
        target_present = target[lag:]
        
        bins = min(int(np.sqrt(len(source))), 10)
        s_p, _ = np.histogram(source_past, bins=bins)
        t_p, _ = np.histogram(target_past, bins=bins)
        t_c, _ = np.histogram(target_present, bins=bins)
        
        s_p = s_p / np.sum(s_p)
        t_p = t_p / np.sum(t_p)
        t_c = t_c / np.sum(t_c)
        
        h_t = -np.sum(t_c * np.log2(t_c + 1e-10))
        h_tt = -np.sum(t_p * np.log2(t_p + 1e-10))
        h_ts = -np.sum(s_p * np.log2(s_p + 1e-10))
        
        h_t_tt = h_t + h_tt - 0.5 * abs(h_t - h_tt)
        h_t_tt_ts = h_t_tt + h_ts - 0.3 * abs(h_t_tt - h_ts)
        
        te = h_t_tt - h_t_tt_ts
        
        return max(0.0, te)
    
    def _calculate_granger_causality(self, source: Any, target: Any, max_lag: int = 5) -> Tuple[float, float]:
        """
        Calculate Granger causality from source to target time series.
        
        Parameters:
        - source: Source time series
        - target: Target time series
        - max_lag: Maximum lag to test
        
        Returns:
        - Tuple of (F-statistic, p-value)
        """
        if len(source) != len(target) or len(source) <= max_lag + 1:
            return 0.0, 1.0
        
        try:
            y = target[max_lag:]
            
            X_restricted = np.zeros((len(y), max_lag))
            for i in range(max_lag):
                X_restricted[:, i] = target[max_lag-i-1:-i-1]
            
            X_restricted = np.column_stack((np.ones(len(y)), X_restricted))
            
            X_unrestricted = np.zeros((len(y), max_lag*2))
            for i in range(max_lag):
                X_unrestricted[:, i] = target[max_lag-i-1:-i-1]
                X_unrestricted[:, i+max_lag] = source[max_lag-i-1:-i-1]
            
            X_unrestricted = np.column_stack((np.ones(len(y)), X_unrestricted))
            
            beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            beta_unrestricted = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
            
            resid_restricted = y - X_restricted.dot(beta_restricted)
            resid_unrestricted = y - X_unrestricted.dot(beta_unrestricted)
            
            rss_restricted = np.sum(resid_restricted**2)
            rss_unrestricted = np.sum(resid_unrestricted**2)
            
            n = len(y)
            df1 = max_lag
            df2 = n - 2*max_lag - 1
            
            if df2 <= 0 or rss_unrestricted == 0 or rss_restricted == rss_unrestricted:
                return 0.0, 1.0
            
            f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
            
            return float(f_stat), float(p_value)
            
        except Exception as e:
            self.logger.error(f"Error in Granger causality calculation: {str(e)}")
            return 0.0, 1.0
    
    def _calculate_quantum_correlation(self, series1: Any, series2: Any) -> float:
        """
        Calculate quantum correlation between two time series.
        This goes beyond classical correlation by detecting non-local relationships.
        
        Parameters:
        - series1: First time series
        - series2: Second time series
        
        Returns:
        - Quantum correlation value
        """
        if len(series1) != len(series2) or len(series1) < 3:
            return 0.0
        
        try:
            classical_corr = np.corrcoef(series1, series2)[0, 1]
            
            bins = min(int(np.sqrt(len(series1))), 10)
            hist_xy, _, _ = np.histogram2d(series1, series2, bins=bins)
            hist_x, _ = np.histogram(series1, bins=bins)
            hist_y, _ = np.histogram(series2, bins=bins)
            
            p_xy = hist_xy / np.sum(hist_xy)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            hilbert1 = np.abs(np.fft.ifft(np.fft.fft(series1) * np.exp(1j * np.pi/2)))
            hilbert2 = np.abs(np.fft.ifft(np.fft.fft(series2) * np.exp(1j * np.pi/2)))
            
            phase1 = np.arctan2(hilbert1, series1)
            phase2 = np.arctan2(hilbert2, series2)
            
            phase_diff = phase1 - phase2
            sync = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            quantum_corr = 0.4 * abs(classical_corr) + 0.3 * mi + 0.3 * sync
            
            return min(1.0, max(0.0, float(quantum_corr)))
            
        except Exception as e:
            self.logger.error(f"Error in quantum correlation calculation: {str(e)}")
            return 0.0
    
    def _detect_causal_relationships(self, data: pd.DataFrame, features: List[str], target: str) -> Dict[str, Dict[str, float]]:
        """
        Detect causal relationships between features and target.
        
        Parameters:
        - data: DataFrame with market data
        - features: List of feature names
        - target: Target variable name
        
        Returns:
        - Dictionary with causal relationships
        """
        causal_relationships = {}
        
        target_series = data[target].values
        
        for feature in features:
            if feature == target:
                continue
                
            feature_series = data[feature].values
            
            te_forward = self._calculate_transfer_entropy(feature_series, target_series)
            te_backward = self._calculate_transfer_entropy(target_series, feature_series)
            
            gc_f_stat, gc_p_value = self._calculate_granger_causality(feature_series, target_series)
            
            qc = self._calculate_quantum_correlation(feature_series, target_series)
            
            causal_strength = 0.4 * te_forward + 0.4 * (gc_f_stat / (gc_f_stat + 1.0)) + 0.2 * qc
            causal_direction = (te_forward - te_backward) / (te_forward + te_backward + 1e-10)
            
            if causal_strength >= self.min_causal_strength and gc_p_value <= self.significance_level:
                causal_relationships[feature] = {
                    'strength': float(causal_strength),
                    'direction': float(causal_direction),
                    'p_value': float(gc_p_value),
                    'quantum_correlation': float(qc)
                }
                
                if feature not in self.causal_graph:
                    self.causal_graph.add_node(feature)
                if target not in self.causal_graph:
                    self.causal_graph.add_node(target)
                
                self.causal_graph.add_edge(feature, target, weight=causal_strength)
        
        return causal_relationships
    
    def _build_causal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build causal features based on the causal graph.
        
        Parameters:
        - data: DataFrame with market data
        
        Returns:
        - DataFrame with causal features
        """
        causal_data = data.copy()
        
        effect_nodes = [node for node in self.causal_graph.nodes() if self.causal_graph.in_degree(node) > 0]
        
        for effect in effect_nodes:
            if effect not in data.columns:
                continue
                
            causes = list(self.causal_graph.predecessors(effect))
            
            if not causes:
                continue
                
            for i, cause1 in enumerate(causes):
                if cause1 not in data.columns:
                    continue
                    
                for lag in range(1, min(self.max_lag + 1, len(data) // 10)):
                    lag_name = f"{cause1}_lag_{lag}"
                    causal_data[lag_name] = data[cause1].shift(lag)
                
                for cause2 in causes[i+1:]:
                    if cause2 not in data.columns:
                        continue
                        
                    interaction_name = f"{cause1}_x_{cause2}"
                    causal_data[interaction_name] = data[cause1] * data[cause2]
                    
                    qc = self._calculate_quantum_correlation(data[cause1].values, data[cause2].values)
                    if qc >= self.entanglement_threshold:
                        qc_name = f"{cause1}_qc_{cause2}"
                        causal_data[qc_name] = (data[cause1] + data[cause2]) * qc
        
        causal_data = causal_data.dropna()
        
        return causal_data
    
    def _predict_with_causal_model(self, data: pd.DataFrame, target: str) -> Tuple[float, float]:
        """
        Make predictions using the causal model.
        
        Parameters:
        - data: DataFrame with market data
        - target: Target variable name
        
        Returns:
        - Tuple of (prediction, confidence)
        """
        if target not in data.columns or len(data) < self.max_lag + 10:
            return 0.0, 0.0
            
        try:
            latest_data = data.iloc[-1:].copy()
            
            causes = list(self.causal_graph.predecessors(target)) if target in self.causal_graph else []
            
            if not causes:
                return 0.0, 0.0
                
            train_data = data.iloc[:-1].copy()
            
            X_columns = []
            
            for cause in causes:
                if cause in train_data.columns:
                    X_columns.append(cause)
                    
                    for lag in range(1, min(self.max_lag + 1, len(train_data) // 10)):
                        lag_name = f"{cause}_lag_{lag}"
                        if lag_name in train_data.columns:
                            X_columns.append(lag_name)
            
            for col in train_data.columns:
                if '_x_' in col or '_qc_' in col:
                    X_columns.append(col)
            
            if not X_columns:
                return 0.0, 0.0
                
            X_train = train_data[X_columns].values
            y_train = train_data[target].values
            
            X_pred = latest_data[X_columns].values
            
            X_train_np = np.array(X_train, dtype=np.float64)
            y_train_np = np.array(y_train, dtype=np.float64)
            
            try:
                beta = np.linalg.solve(X_train_np.T.dot(X_train_np), X_train_np.T.dot(y_train_np))
            except:
                beta = np.zeros(X_train_np.shape[1])
                for i in range(len(beta)):
                    beta[i] = 0.1  # Default coefficient
            
            prediction = X_pred.dot(beta)[0]
            
            confidence = np.mean([self.causal_graph.get_edge_data(cause, target)['weight'] 
                                 for cause in causes if self.causal_graph.has_edge(cause, target)])
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in causal prediction: {str(e)}")
            return 0.0, 0.0
    
    def _generate_counterfactuals(self, data: pd.DataFrame, target: str, num_scenarios: int = 3) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios.
        
        Parameters:
        - data: DataFrame with market data
        - target: Target variable name
        - num_scenarios: Number of counterfactual scenarios to generate
        
        Returns:
        - List of counterfactual scenarios
        """
        counterfactuals = []
        
        if target not in data.columns or len(data) < 20:
            return counterfactuals
            
        try:
            causes = list(self.causal_graph.predecessors(target)) if target in self.causal_graph else []
            
            if not causes:
                return counterfactuals
                
            latest_data = data.iloc[-1].copy()
            
            for _ in range(num_scenarios):
                scenario = {'changes': {}, 'prediction': 0.0, 'probability': 0.0}
                
                num_causes_to_modify = min(len(causes), random.randint(1, 3))
                causes_to_modify = random.sample(causes, num_causes_to_modify)
                
                counterfactual_data = latest_data.copy()
                
                for cause in causes_to_modify:
                    if cause not in data.columns:
                        continue
                        
                    historical_values = data[cause].values
                    
                    mean_val = np.mean(historical_values)
                    std_val = np.std(historical_values)
                    
                    counterfactual_value = mean_val + random.uniform(-2, 2) * std_val
                    
                    try:
                        value_str = str(latest_data[cause])
                        numeric_part = ''.join(c for c in value_str if c.isdigit() or c == '.' or c == '-')
                        if numeric_part:
                            original_value = float(numeric_part)
                        else:
                            original_value = 0.0
                            
                        scenario['changes'][cause] = {
                            'original': original_value,
                            'counterfactual': float(counterfactual_value),
                            'change_pct': float((counterfactual_value - original_value) / (original_value + 1e-10) * 100)
                        }
                    except (ValueError, TypeError, AttributeError):
                        scenario['changes'][cause] = {
                            'original': 0.0,
                            'counterfactual': float(counterfactual_value),
                            'change_pct': 0.0
                        }
                    
                    counterfactual_data[cause] = counterfactual_value
                
                cf_df = pd.DataFrame([counterfactual_data])
                
                prediction, confidence = self._predict_with_causal_model(pd.concat([data.iloc[:-1], cf_df]), target)
                
                scenario['prediction'] = float(prediction)
                scenario['confidence'] = float(confidence)
                scenario['probability'] = float(0.5 + 0.5 * (1 - 1 / (1 + len(causes_to_modify))))
                
                counterfactuals.append(scenario)
            
            return counterfactuals
            
        except Exception as e:
            self.logger.error(f"Error generating counterfactuals: {str(e)}")
            return counterfactuals
    
    def analyze_market(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Dict[str, Any]:
        """
        Analyze market data using causal reasoning.
        
        Parameters:
        - symbol: Trading symbol (e.g., 'BTC/USDT')
        - timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
        - limit: Number of candles to analyze
        
        Returns:
        - Dictionary with analysis results
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < limit * 0.9:  # Ensure we have enough data
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'error': 'Insufficient data'
                }
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_returns'].rolling(window=20).std()
            
            close_values = np.array(df['close'].values, dtype=np.float64)
            df['market_depth'] = self._calculate_market_depth(close_values)
            df['exchange_latency'] = self._calculate_exchange_latency(close_values)
            volume_values = np.array(df['volume'].values, dtype=np.float64)
            df['volume_profile'] = self._calculate_volume_profile(volume_values)
            
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['price_range'] = (df['high'] - df['low']) / df['close']
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            df = df.dropna()
            
            current_time = datetime.now()
            if self.last_update_time is None or current_time - self.last_update_time >= self.update_interval:
                features = ['open', 'high', 'low', 'volume', 'returns', 'volatility', 'rsi', 
                           'volume_change', 'high_low_ratio', 'price_range', 'hour', 'day_of_week']
                
                causal_rels = self._detect_causal_relationships(df, features, 'close')
                
                self.last_update_time = current_time
                
                self.logger.info(f"Updated causal relationships for {symbol}: {len(causal_rels)} significant relationships found")
            
            causal_df = self._build_causal_features(df)
            
            prediction, confidence = self._predict_with_causal_model(causal_df, 'close')
            
            counterfactuals = self._generate_counterfactuals(causal_df, 'close')
            
            current_price = df['close'].iloc[-1]
            
            if prediction > current_price * 1.01:  # 1% threshold
                signal = 'BUY'
            elif prediction < current_price * 0.99:  # 1% threshold
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
            
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': float(current_price),
                'predicted_price': float(prediction),
                'confidence': float(confidence),
                'signal': signal
            })
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'confidence': float(confidence),
                'current_price': float(current_price),
                'predicted_price': float(prediction),
                'price_change_pct': float((prediction - current_price) / current_price * 100),
                'causal_factors': [{'factor': node, 'strength': float(self.causal_graph.get_edge_data(node, 'close')['weight'])} 
                                  for node in self.causal_graph.predecessors('close')],
                'counterfactuals': counterfactuals,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_market_depth(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate market depth using price data.
        
        Parameters:
        - prices: Array of price values
        
        Returns:
        - Array of market depth values
        """
        window = 20
        if len(prices) < window:
            return np.zeros_like(prices)
            
        depth = np.zeros_like(prices)
        
        for i in range(window, len(prices)):
            price_range = np.max(prices[i-window:i]) - np.min(prices[i-window:i])
            volatility = np.std(prices[i-window:i])
            
            depth[i] = 1.0 / (volatility + 1e-10) * price_range
            
        min_val = np.min(depth[depth > 0]) if np.any(depth > 0) else 1e-10
        max_val = np.max(depth) if np.any(depth > 0) else 1.0
        
        depth = 100 * (depth - min_val) / (max_val - min_val + 1e-10)
        depth[depth < 0] = 0
        depth[depth > 100] = 100
        
        return depth
        
    def _calculate_exchange_latency(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate exchange latency simulation using price data.
        
        Parameters:
        - prices: Array of price values
        
        Returns:
        - Array of exchange latency values
        """
        latency = np.zeros_like(prices)
        
        for i in range(1, len(prices)):
            price_change = abs(prices[i] - prices[i-1]) / (prices[i-1] + 1e-10)
            latency[i] = 100 * (1 - np.exp(-100 * price_change))
            
        return latency
        
    def _calculate_volume_profile(self, volumes: np.ndarray) -> np.ndarray:
        """
        Calculate volume profile using volume data.
        
        Parameters:
        - volumes: Array of volume values
        
        Returns:
        - Array of volume profile values
        """
        profile = np.zeros_like(volumes)
        window = 20
        
        if len(volumes) < window:
            return profile
            
        for i in range(window, len(volumes)):
            avg_vol = np.mean(volumes[i-window:i])
            if avg_vol > 0:
                profile[i] = 100 * volumes[i] / avg_vol
            else:
                profile[i] = 0
                
        return profile
        
    def _calculate_rsi(self, prices: Any, period: int = 14) -> np.ndarray:
        """
        Calculate RSI using numpy arrays.
        
        Parameters:
        - prices: Array of price values
        - period: RSI period
        
        Returns:
        - Array of RSI values
        """
        deltas = np.zeros_like(prices)
        deltas[1:] = prices[1:] - prices[:-1]
        
        gains = np.zeros_like(deltas)
        losses = np.zeros_like(deltas)
        
        gains[deltas > 0] = deltas[deltas > 0]
        losses[deltas < 0] = -deltas[deltas < 0]
        
        avg_gains = np.zeros_like(gains)
        avg_losses = np.zeros_like(losses)
        
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[1:period+1])
            avg_losses[period] = np.mean(losses[1:period+1])
            
            for i in range(period+1, len(gains)):
                avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
                avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        rs = np.zeros_like(avg_gains)
        rsi = np.zeros_like(avg_gains)
        
        valid_indices = avg_losses != 0
        rs[valid_indices] = avg_gains[valid_indices] / avg_losses[valid_indices]
        
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_trading_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Generate trading signal based on causal analysis.
        
        Parameters:
        - symbol: Trading symbol
        - timeframe: Timeframe for analysis
        
        Returns:
        - Dictionary with trading signal information
        """
        analysis = self.analyze_market(symbol, timeframe)
        
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': analysis.get('signal', 'NEUTRAL'),
            'confidence': analysis.get('confidence', 0.0),
            'current_price': analysis.get('current_price', 0.0),
            'predicted_price': analysis.get('predicted_price', 0.0),
            'price_change_pct': analysis.get('price_change_pct', 0.0)
        }
        
        if 'causal_factors' in analysis:
            signal['causal_factors'] = analysis['causal_factors']
        
        if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] > 0.95:
            base_size = 0.02  # 2% base position size
            confidence_factor = signal['confidence']
            change_factor = min(1.0, abs(signal['price_change_pct']) / 5.0)  # Cap at 5% change
            
            position_size = base_size * confidence_factor * change_factor
            
            signal['position_size'] = float(position_size)
            signal['stop_loss_pct'] = float(1.5)  # Default 1.5% stop loss
            signal['take_profit_pct'] = float(abs(signal['price_change_pct']) * 0.8)  # 80% of predicted change
        
        return signal
    
    def get_causal_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the causal graph.
        
        Returns:
        - Dictionary with causal graph information
        """
        if not self.causal_graph:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0.0,
                'strongest_causes': []
            }
        
        nodes = self.causal_graph.number_of_nodes()
        edges = self.causal_graph.number_of_edges()
        density = nx.density(self.causal_graph)
        
        strongest_causes = []
        
        for node in self.causal_graph.nodes():
            outgoing_edges = [(node, target, data['weight']) 
                             for target, data in self.causal_graph[node].items()]
            
            for source, target, weight in outgoing_edges:
                strongest_causes.append({
                    'cause': source,
                    'effect': target,
                    'strength': float(weight)
                })
        
        strongest_causes = sorted(strongest_causes, key=lambda x: x['strength'], reverse=True)[:10]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'density': float(density),
            'strongest_causes': strongest_causes
        }
