"""
Causal Flow Splitter (CFS)

A predictive engine that separates true cause from correlated noise in order flow.
Result: Pure signal trading — not volume, but motive.
True Edge: Trade intent, not outcome.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

class CausalFlowSplitter:
    """
    Causal Flow Splitter (CFS) module that separates true cause from correlated
    noise in order flow to identify trading intent rather than outcome.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Causal Flow Splitter module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('CFS')
        self.causal_map = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=30)
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'causal_detection_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'noise_reduction': 0.0,
            'successful_trades': 0
        }
    
    def _fetch_order_flow_data(self, symbol: str, timeframe: str = '1m', limit: int = 500) -> pd.DataFrame:
        """
        Fetch order flow data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        - timeframe: Timeframe for data
        - limit: Maximum number of candles to fetch
        
        Returns:
        - DataFrame with order flow data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 10:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_range'] = (df['close'] - df['open']) / df['open']
            
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            df['volatility'] = df['log_returns'].rolling(window=20).std()
            
            df['market_depth'] = self._calculate_market_depth(df)
            
            df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
            
            df['trade_intensity'] = df['volume'] / df['high_low_range']
            
            df['momentum'] = df['close'].pct_change(5)
            
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching order flow data: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_market_depth(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market depth using price and volume data.
        
        Parameters:
        - df: DataFrame with price and volume data
        
        Returns:
        - Series with market depth values
        """
        
        depth = pd.Series(index=df.index, dtype=float)
        
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            price_range = (window['high'].max() - window['low'].min()) / window['close'].iloc[-1]
            volume_sum = window['volume'].sum()
            
            depth.iloc[i] = volume_sum / (price_range + 1e-10)
        
        if not depth.empty:
            min_val = depth[depth > 0].min() if not depth[depth > 0].empty else 1e-10
            max_val = depth.max() if not depth.empty else 1.0
            
            depth = 100 * (depth - min_val) / (max_val - min_val + 1e-10)
            depth = depth.clip(0, 100)
        
        return depth
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow imbalance using price and volume data.
        
        Parameters:
        - df: DataFrame with price and volume data
        
        Returns:
        - Series with order flow imbalance values
        """
        
        imbalance = pd.Series(index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
            volume = df['volume'].iloc[i]
            
            imbalance.iloc[i] = np.sign(price_change) * volume
        
        if not imbalance.empty:
            max_abs = max(abs(imbalance.max()), abs(imbalance.min())) if not imbalance.empty else 1.0
            
            if max_abs > 0:
                imbalance = imbalance / max_abs
        
        return imbalance
    
    def _detect_causal_relationships(self, df: pd.DataFrame, max_lag: int = 5) -> Dict[str, Any]:
        """
        Detect causal relationships in order flow data.
        
        Parameters:
        - df: DataFrame with order flow data
        - max_lag: Maximum lag for Granger causality tests
        
        Returns:
        - Dictionary with causal relationships
        """
        if df.empty or len(df) < max_lag + 10:
            return {
                'causal_factors': {},
                'noise_factors': [],
                'confidence': 0.0
            }
        
        features = ['price_change', 'volume_change', 'high_low_range', 'close_open_range', 
                   'volatility', 'market_depth', 'order_flow_imbalance', 'trade_intensity', 'momentum']
        
        target = 'price_change'
        
        causal_factors = {}
        noise_factors = []
        
        for feature in features:
            if feature == target:
                continue
                
            data = pd.DataFrame({
                'y': df[target].values,
                'x': df[feature].values
            }).dropna()
            
            if len(data) < max_lag + 10:
                continue
                
            try:
                gc_res = grangercausalitytests(data, max_lag, verbose=False)
                
                p_values = [gc_res[lag+1][0]['ssr_ftest'][1] for lag in range(max_lag)]
                
                min_p_value = min(p_values)
                best_lag = p_values.index(min_p_value) + 1
                
                if min_p_value < 0.05:  # Statistically significant
                    correlation = np.corrcoef(data['x'], data['y'])[0, 1]
                    
                    causality_strength = 1.0 - min_p_value
                    
                    causal_factors[feature] = {
                        'lag': best_lag,
                        'p_value': float(min_p_value),
                        'correlation': float(correlation),
                        'causality_strength': float(causality_strength)
                    }
                else:
                    noise_factors.append(feature)
                    
            except Exception as e:
                self.logger.error(f"Error in Granger causality test for {feature}: {str(e)}")
                noise_factors.append(feature)
        
        if causal_factors:
            avg_strength = sum(factor['causality_strength'] for factor in causal_factors.values()) / len(causal_factors)
            confidence = min(0.5 + avg_strength * 0.5, 0.99)  # Cap at 0.99
        else:
            confidence = 0.0
        
        return {
            'causal_factors': causal_factors,
            'noise_factors': noise_factors,
            'confidence': float(confidence)
        }
    
    def _predict_with_causal_model(self, df: pd.DataFrame, causal_factors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions using causal factors.
        
        Parameters:
        - df: DataFrame with order flow data
        - causal_factors: Dictionary with causal factors
        
        Returns:
        - Dictionary with prediction results
        """
        if df.empty or not causal_factors:
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
        
        X = pd.DataFrame()
        
        for feature, info in causal_factors.items():
            lag = info['lag']
            X[f"{feature}_lag{lag}"] = df[feature].shift(lag)
        
        X = X.dropna()
        
        if X.empty:
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
        
        last_row = np.array(X.iloc[-1].values).reshape(1, -1)
        
        try:
            y = df['price_change'].shift(-1).dropna()
            X_train = X.iloc[:-1]
            y_train = y.iloc[:len(X_train)]
            
            if len(X_train) < 10 or len(y_train) < 10:
                return {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'direction': 'NEUTRAL'
                }
            
            X_train = sm.add_constant(X_train)
            last_row_with_const = np.hstack([1, last_row])
            
            model = sm.OLS(y_train, X_train)
            results = model.fit()
            
            prediction = results.predict(last_row_with_const)[0]
            
            r_squared = results.rsquared
            prediction_magnitude = abs(prediction) / df['price_change'].std()
            
            confidence = min(0.5 + r_squared * 0.25 + prediction_magnitude * 0.25, 0.99)
            
            if prediction > 0:
                direction = 'BUY'
            elif prediction < 0:
                direction = 'SELL'
            else:
                direction = 'NEUTRAL'
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'direction': direction,
                'r_squared': float(r_squared),
                'prediction_magnitude': float(prediction_magnitude)
            }
            
        except Exception as e:
            self.logger.error(f"Error in causal prediction: {str(e)}")
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'direction': 'NEUTRAL',
                'error': str(e)
            }
    
    def update_causal_map(self, symbols: List[str]) -> None:
        """
        Update the causal map for multiple symbols.
        
        Parameters:
        - symbols: List of trading symbols
        """
        current_time = datetime.now()
        
        if current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        for symbol in symbols:
            df = self._fetch_order_flow_data(symbol)
            
            if df.empty:
                continue
                
            causal_relationships = self._detect_causal_relationships(df)
            
            self.causal_map[symbol] = {
                'timestamp': current_time.isoformat(),
                'causal_factors': causal_relationships['causal_factors'],
                'noise_factors': causal_relationships['noise_factors'],
                'confidence': causal_relationships['confidence']
            }
        
        self.logger.info(f"Updated causal map for {len(symbols)} symbols")
    
    def split_causal_flow(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split causal flow from noise to generate trading signals.
        
        Parameters:
        - symbol: Trading symbol
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with trading signal
        """
        try:
            if symbol not in self.causal_map:
                self.update_causal_map([symbol])
            
            df = self._fetch_order_flow_data(symbol)
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            if symbol in self.causal_map:
                causal_factors = self.causal_map[symbol]['causal_factors']
            else:
                causal_relationships = self._detect_causal_relationships(df)
                causal_factors = causal_relationships['causal_factors']
                
                self.causal_map[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'causal_factors': causal_factors,
                    'noise_factors': causal_relationships['noise_factors'],
                    'confidence': causal_relationships['confidence']
                }
            
            prediction = self._predict_with_causal_model(df, causal_factors)
            
            signal = prediction['direction']
            confidence = prediction['confidence']
            
            if confidence >= self.confidence_threshold and signal in ['BUY', 'SELL']:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': float(confidence),
                    'prediction': float(prediction['prediction']),
                    'causal_factors': {k: v['causality_strength'] for k, v in causal_factors.items()},
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': float(confidence),
                    'prediction': float(prediction['prediction']) if 'prediction' in prediction else 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error splitting causal flow: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Causal Flow Splitter.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'causal_detection_accuracy': float(self.performance['causal_detection_accuracy']),
            'prediction_accuracy': float(self.performance['prediction_accuracy']),
            'noise_reduction': float(self.performance['noise_reduction']),
            'successful_trades': int(self.performance['successful_trades']),
            'symbols_analyzed': len(self.causal_map),
            'timestamp': datetime.now().isoformat()
        }
