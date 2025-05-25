"""
Truth Verification Core (TVC)

Discovery: An AI core that can detect lies, propaganda, or corrupted knowledge by comparing to cosmic invariant truth patterns.
Why it matters: Shatters fake news, deepfakes, and misinformation at the root.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import random
from collections import defaultdict

class TruthVerificationCore:
    """
    Truth Verification Core (TVC) module that detects market manipulation, false signals,
    and misinformation by comparing to invariant truth patterns in market data.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Truth Verification Core module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('TVC')
        self.truth_patterns = {}
        self.manipulation_signals = {}
        self.verification_results = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=10)
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'manipulation_detection_accuracy': 0.0,
            'false_signal_detection_rate': 0.0,
            'verification_speed': 0.0,
            'successful_trades': 0
        }
    
    def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with market data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            order_book = self.exchange.fetch_order_book(symbol)
            
            trades = self.exchange.fetch_trades(symbol, limit=100)
            
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            ohlcv_data = {}
            
            for tf in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=100)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    ohlcv_data[tf] = df.to_dict('records')
            
            market_data = {
                'symbol': symbol,
                'ticker': ticker,
                'order_book': order_book,
                'trades': trades,
                'ohlcv': ohlcv_data,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_truth_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract invariant truth patterns from market data.
        
        Parameters:
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with truth patterns
        """
        if not market_data or 'error' in market_data:
            return {}
            
        symbol = market_data['symbol']
        truth_patterns = {}
        
        if 'ohlcv' in market_data and '1h' in market_data['ohlcv']:
            ohlcv = market_data['ohlcv']['1h']
            
            if len(ohlcv) >= 24:
                volumes = [candle['volume'] for candle in ohlcv]
                prices = [candle['close'] for candle in ohlcv]
                
                vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else 0
                
                volume_std = np.std(volumes)
                volume_mean = np.mean(volumes)
                
                truth_patterns['volume_profile'] = {
                    'vwap': float(vwap),
                    'volume_std': float(volume_std),
                    'volume_mean': float(volume_mean),
                    'volume_cv': float(volume_std / volume_mean) if volume_mean > 0 else 0
                }
        
        if 'order_book' in market_data:
            order_book = market_data['order_book']
            
            if 'bids' in order_book and 'asks' in order_book:
                bids = order_book['bids']
                asks = order_book['asks']
                
                if bids and asks:
                    spread = (asks[0][0] - bids[0][0]) / bids[0][0] if bids[0][0] > 0 else 0
                    
                    bid_depth = sum(bid[1] for bid in bids[:10])
                    ask_depth = sum(ask[1] for ask in asks[:10])
                    
                    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
                    
                    truth_patterns['order_book'] = {
                        'spread': float(spread),
                        'bid_depth': float(bid_depth),
                        'ask_depth': float(ask_depth),
                        'imbalance': float(imbalance)
                    }
        
        if 'ohlcv' in market_data and '1h' in market_data['ohlcv']:
            ohlcv = market_data['ohlcv']['1h']
            
            if len(ohlcv) >= 24:
                closes = [candle['close'] for candle in ohlcv]
                returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                
                if returns:
                    return_mean = np.mean(returns)
                    return_std = np.std(returns)
                    return_skew = stats.skew(returns) if len(returns) >= 3 else 0
                    return_kurtosis = stats.kurtosis(returns) if len(returns) >= 4 else 0
                    
                    truth_patterns['price_action'] = {
                        'return_mean': float(return_mean),
                        'return_std': float(return_std),
                        'return_skew': float(return_skew),
                        'return_kurtosis': float(return_kurtosis)
                    }
        
        if 'trades' in market_data:
            trades = market_data['trades']
            
            if trades:
                trade_sizes = [trade['amount'] for trade in trades if 'amount' in trade]
                
                if trade_sizes:
                    size_mean = np.mean(trade_sizes)
                    size_std = np.std(trade_sizes)
                    size_max = max(trade_sizes)
                    
                    truth_patterns['trade_flow'] = {
                        'size_mean': float(size_mean),
                        'size_std': float(size_std),
                        'size_max': float(size_max),
                        'size_cv': float(size_std / size_mean) if size_mean > 0 else 0
                    }
        
        return truth_patterns
    
    def _detect_manipulation(self, truth_patterns: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect market manipulation by comparing current data to truth patterns.
        
        Parameters:
        - truth_patterns: Truth patterns dictionary
        - market_data: Current market data
        
        Returns:
        - Dictionary with manipulation signals
        """
        if not truth_patterns or not market_data or 'error' in market_data:
            return {}
            
        manipulation_signals = {}
        
        if 'volume_profile' in truth_patterns and 'ohlcv' in market_data and '1h' in market_data['ohlcv']:
            volume_truth = truth_patterns['volume_profile']
            ohlcv = market_data['ohlcv']['1h']
            
            if len(ohlcv) >= 3:
                recent_volumes = [candle['volume'] for candle in ohlcv[-3:]]
                recent_mean = np.mean(recent_volumes)
                
                if recent_mean > volume_truth['volume_mean'] * 3:
                    manipulation_signals['volume_spike'] = {
                        'severity': float(min(recent_mean / volume_truth['volume_mean'] / 3, 1.0)),
                        'description': 'Abnormal volume spike detected'
                    }
                elif recent_mean < volume_truth['volume_mean'] * 0.3:
                    manipulation_signals['volume_dry_up'] = {
                        'severity': float(min((volume_truth['volume_mean'] / recent_mean - 1) / 2, 1.0)),
                        'description': 'Abnormal volume dry-up detected'
                    }
        
        if 'order_book' in truth_patterns and 'order_book' in market_data:
            order_book_truth = truth_patterns['order_book']
            order_book = market_data['order_book']
            
            if 'bids' in order_book and 'asks' in order_book:
                bids = order_book['bids']
                asks = order_book['asks']
                
                if bids and asks:
                    spread = (asks[0][0] - bids[0][0]) / bids[0][0] if bids[0][0] > 0 else 0
                    bid_depth = sum(bid[1] for bid in bids[:10])
                    ask_depth = sum(ask[1] for ask in asks[:10])
                    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
                    
                    if spread > order_book_truth['spread'] * 3:
                        manipulation_signals['spread_widening'] = {
                            'severity': float(min(spread / order_book_truth['spread'] / 3, 1.0)),
                            'description': 'Abnormal spread widening detected'
                        }
                    
                    if abs(imbalance) > abs(order_book_truth['imbalance']) * 3:
                        manipulation_signals['order_book_imbalance'] = {
                            'severity': float(min(abs(imbalance) / abs(order_book_truth['imbalance']) / 3, 1.0)) if abs(order_book_truth['imbalance']) > 0 else 0.5,
                            'description': 'Abnormal order book imbalance detected'
                        }
        
        if 'price_action' in truth_patterns and 'ohlcv' in market_data and '1h' in market_data['ohlcv']:
            price_truth = truth_patterns['price_action']
            ohlcv = market_data['ohlcv']['1h']
            
            if len(ohlcv) >= 3:
                closes = [candle['close'] for candle in ohlcv[-4:]]
                recent_returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                
                if recent_returns:
                    for ret in recent_returns:
                        if abs(ret) > price_truth['return_std'] * 3:
                            manipulation_signals['price_anomaly'] = {
                                'severity': float(min(abs(ret) / price_truth['return_std'] / 3, 1.0)),
                                'description': 'Abnormal price movement detected'
                            }
                            break
        
        if 'trade_flow' in truth_patterns and 'trades' in market_data:
            trade_truth = truth_patterns['trade_flow']
            trades = market_data['trades']
            
            if trades:
                recent_sizes = [trade['amount'] for trade in trades[-10:] if 'amount' in trade]
                
                if recent_sizes:
                    for size in recent_sizes:
                        if size > trade_truth['size_max'] * 2:
                            manipulation_signals['large_trade'] = {
                                'severity': float(min(size / trade_truth['size_max'] / 2, 1.0)),
                                'description': 'Abnormally large trade detected'
                            }
                            break
        
        return manipulation_signals
    
    def _verify_trading_signal(self, signal: Dict[str, Any], manipulation_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a trading signal against detected manipulation.
        
        Parameters:
        - signal: Trading signal to verify
        - manipulation_signals: Detected manipulation signals
        
        Returns:
        - Dictionary with verification results
        """
        if not signal or 'signal' not in signal or signal['signal'] == 'NEUTRAL':
            return {
                'verified': True,
                'confidence': 0.0,
                'manipulated': False,
                'manipulation_types': []
            }
            
        if manipulation_signals:
            severity_sum = sum(m['severity'] for m in manipulation_signals.values())
            severity_avg = severity_sum / len(manipulation_signals) if manipulation_signals else 0
            
            manipulated = severity_avg > 0.5
            
            if manipulated:
                confidence = 1.0 - (1.0 - severity_avg) * 0.5  # Higher severity = higher confidence in manipulation
            else:
                confidence = 0.7  # Default confidence for non-manipulated signals
            
            return {
                'verified': not manipulated,
                'confidence': float(confidence),
                'manipulated': manipulated,
                'manipulation_types': list(manipulation_signals.keys()),
                'severity': float(severity_avg)
            }
        else:
            return {
                'verified': True,
                'confidence': 0.8,  # Default confidence for verified signals
                'manipulated': False,
                'manipulation_types': []
            }
    
    def update_truth_patterns(self, symbol: str) -> None:
        """
        Update the truth patterns for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        """
        current_time = datetime.now()
        
        if symbol in self.truth_patterns and current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        market_data = self._fetch_market_data(symbol)
        
        if not market_data or 'error' in market_data:
            return
            
        truth_patterns = self._extract_truth_patterns(market_data)
        
        if not truth_patterns:
            return
            
        self.truth_patterns[symbol] = {
            'patterns': truth_patterns,
            'timestamp': current_time.isoformat()
        }
        
        self.logger.info(f"Updated truth patterns for {symbol}")
    
    def detect_market_manipulation(self, symbol: str) -> Dict[str, Any]:
        """
        Detect market manipulation for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with manipulation detection results
        """
        try:
            self.update_truth_patterns(symbol)
            
            if symbol not in self.truth_patterns:
                return {
                    'symbol': symbol,
                    'manipulated': False,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            market_data = self._fetch_market_data(symbol)
            
            if not market_data or 'error' in market_data:
                return {
                    'symbol': symbol,
                    'manipulated': False,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            manipulation_signals = self._detect_manipulation(self.truth_patterns[symbol]['patterns'], market_data)
            
            self.manipulation_signals[symbol] = {
                'signals': manipulation_signals,
                'timestamp': datetime.now().isoformat()
            }
            
            manipulated = len(manipulation_signals) > 0
            
            if manipulated:
                severity_sum = sum(m['severity'] for m in manipulation_signals.values())
                severity_avg = severity_sum / len(manipulation_signals)
                confidence = 0.7 + severity_avg * 0.3  # Scale confidence based on severity
            else:
                severity_avg = 0.0
                confidence = 0.7  # Default confidence for non-manipulated markets
            
            return {
                'symbol': symbol,
                'manipulated': manipulated,
                'confidence': float(confidence),
                'manipulation_types': list(manipulation_signals.keys()),
                'severity': float(severity_avg),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market manipulation: {str(e)}")
            return {
                'symbol': symbol,
                'manipulated': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def verify_signal(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a trading signal against market manipulation.
        
        Parameters:
        - symbol: Trading symbol
        - signal: Trading signal to verify
        
        Returns:
        - Dictionary with verification results
        """
        try:
            manipulation_result = self.detect_market_manipulation(symbol)
            
            if 'error' in manipulation_result:
                return {
                    'symbol': symbol,
                    'verified': False,
                    'confidence': 0.0,
                    'error': manipulation_result['error'],
                    'timestamp': datetime.now().isoformat()
                }
            
            manipulation_signals = self.manipulation_signals.get(symbol, {}).get('signals', {})
            
            verification = self._verify_trading_signal(signal, manipulation_signals)
            
            self.verification_results[symbol] = {
                'verification': verification,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'symbol': symbol,
                'verified': verification['verified'],
                'confidence': float(verification['confidence']),
                'manipulated': verification['manipulated'],
                'manipulation_types': verification['manipulation_types'],
                'original_signal': signal.get('signal', 'NEUTRAL'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying signal: {str(e)}")
            return {
                'symbol': symbol,
                'verified': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_trading_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate verified trading signals.
        
        Parameters:
        - symbol: Trading symbol
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with verified trading signal
        """
        try:
            if not market_data or 'signal' not in market_data:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'verified': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            original_signal = market_data.get('signal', 'NEUTRAL')
            original_confidence = market_data.get('confidence', 0.0)
            
            verification = self.verify_signal(symbol, market_data)
            
            if verification['verified'] and original_signal in ['BUY', 'SELL']:
                verified_confidence = original_confidence * verification['confidence']
                
                if verified_confidence >= self.confidence_threshold:
                    return {
                        'symbol': symbol,
                        'signal': original_signal,
                        'confidence': float(verified_confidence),
                        'verified': True,
                        'verification_confidence': float(verification['confidence']),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'symbol': symbol,
                        'signal': 'NEUTRAL',
                        'confidence': float(verified_confidence),
                        'verified': True,
                        'verification_confidence': float(verification['confidence']),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'verified': False,
                    'verification_confidence': float(verification['confidence']),
                    'manipulation_detected': verification['manipulated'],
                    'manipulation_types': verification['manipulation_types'],
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error generating verified trading signal: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Truth Verification Core.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'manipulation_detection_accuracy': float(self.performance['manipulation_detection_accuracy']),
            'false_signal_detection_rate': float(self.performance['false_signal_detection_rate']),
            'verification_speed': float(self.performance['verification_speed']),
            'successful_trades': int(self.performance['successful_trades']),
            'symbols_analyzed': len(self.truth_patterns),
            'timestamp': datetime.now().isoformat()
        }
