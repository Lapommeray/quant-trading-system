"""
Shadow Spread Resonator (SSR)

Detects ultra-micro spread anomalies invisible to humans — used by black-box funds.
Result: Snipe before any visible divergence.
True Edge: This is the cheat code behind 0.001% funds.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks, savgol_filter

class ShadowSpreadResonator:
    """
    Shadow Spread Resonator (SSR) module that detects ultra-micro spread anomalies
    invisible to humans, used by black-box funds.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Shadow Spread Resonator module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('SSR')
        self.spread_anomalies = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(seconds=30)  # More frequent updates for micro anomalies
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'anomaly_detection_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'average_lead_time': 0.0,
            'successful_trades': 0
        }
    
    def _fetch_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker data: {str(e)}")
            return {}
    
    def _fetch_order_book_data(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch order book data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        - limit: Maximum number of orders to fetch
        
        Returns:
        - Order book data
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            return {'bids': [], 'asks': []}
    
    def _fetch_trades_data(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch recent trades data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        - limit: Maximum number of trades to fetch
        
        Returns:
        - List of trades
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"Error fetching trades: {str(e)}")
            return []
    
    def _calculate_micro_spread_metrics(self, order_book: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate micro spread metrics from order book and trades data.
        
        Parameters:
        - order_book: Order book data
        - trades: Recent trades data
        
        Returns:
        - Dictionary with micro spread metrics
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book or not trades:
            return {}
            
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {}
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = spread / mid_price * 100
        
        bid_volume_sum = sum(float(bid[1]) for bid in bids[:10])
        ask_volume_sum = sum(float(ask[1]) for ask in asks[:10])
        
        volume_weighted_bid = sum(float(bid[0]) * float(bid[1]) for bid in bids[:10]) / bid_volume_sum if bid_volume_sum > 0 else best_bid
        volume_weighted_ask = sum(float(ask[0]) * float(ask[1]) for ask in asks[:10]) / ask_volume_sum if ask_volume_sum > 0 else best_ask
        
        volume_weighted_spread = volume_weighted_ask - volume_weighted_bid
        volume_weighted_spread_pct = volume_weighted_spread / mid_price * 100
        
        bid_depth = sum(float(bid[1]) for bid in bids[:20])
        ask_depth = sum(float(ask[1]) for ask in asks[:20])
        
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        if len(trades) >= 2:
            trade_prices = [float(trade['price']) for trade in trades]
            trade_volumes = [float(trade['amount']) for trade in trades]
            trade_sides = [trade['side'] for trade in trades]
            
            buy_volume = sum(vol for vol, side in zip(trade_volumes, trade_sides) if side == 'buy')
            sell_volume = sum(vol for vol, side in zip(trade_volumes, trade_sides) if side == 'sell')
            
            trade_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
            
            price_diffs = np.diff(trade_prices)
            micro_volatility = np.std(price_diffs) / mid_price * 100
            
            spread_crossings = 0
            for i in range(1, len(trades)):
                if (trades[i]['price'] >= best_ask and trades[i-1]['price'] <= best_bid) or \
                   (trades[i]['price'] <= best_bid and trades[i-1]['price'] >= best_ask):
                    spread_crossings += 1
            
            spread_crossing_rate = spread_crossings / len(trades)
            
            effective_spreads = []
            for trade in trades:
                if trade['side'] == 'buy':
                    effective_spread = (float(trade['price']) - mid_price) / mid_price * 100
                else:
                    effective_spread = (mid_price - float(trade['price'])) / mid_price * 100
                
                effective_spreads.append(effective_spread)
            
            avg_effective_spread = np.mean(effective_spreads) if effective_spreads else 0
            
            spread_efficiency = avg_effective_spread / spread_pct if spread_pct > 0 else 0
            
            if len(trades) >= 20:
                if 'datetime' in trades[0]:
                    trade_times = [datetime.fromisoformat(trade['datetime'].replace('Z', '+00:00')) for trade in trades if 'datetime' in trade]
                    if len(trade_times) >= 2:
                        time_diffs = [(trade_times[i] - trade_times[i+1]).total_seconds() for i in range(len(trade_times)-1)]
                        avg_time_between_trades = np.mean(time_diffs) if time_diffs else 0
                        time_variability = np.std(time_diffs) / avg_time_between_trades if avg_time_between_trades > 0 else 0
                    else:
                        avg_time_between_trades = 0
                        time_variability = 0
                else:
                    avg_time_between_trades = 0
                    time_variability = 0
                
                price_reversals = 0
                for i in range(2, len(trade_prices)):
                    if (trade_prices[i] > trade_prices[i-1] and trade_prices[i-1] < trade_prices[i-2]) or \
                       (trade_prices[i] < trade_prices[i-1] and trade_prices[i-1] > trade_prices[i-2]):
                        price_reversals += 1
                
                price_reversal_rate = price_reversals / (len(trade_prices) - 2) if len(trade_prices) > 2 else 0
            else:
                avg_time_between_trades = 0
                time_variability = 0
                price_reversal_rate = 0
        else:
            trade_flow_imbalance = 0
            micro_volatility = 0
            spread_crossing_rate = 0
            avg_effective_spread = 0
            spread_efficiency = 0
            avg_time_between_trades = 0
            time_variability = 0
            price_reversal_rate = 0
        
        return {
            'mid_price': float(mid_price),
            'spread': float(spread),
            'spread_pct': float(spread_pct),
            'volume_weighted_spread': float(volume_weighted_spread),
            'volume_weighted_spread_pct': float(volume_weighted_spread_pct),
            'depth_imbalance': float(depth_imbalance),
            'trade_flow_imbalance': float(trade_flow_imbalance),
            'micro_volatility': float(micro_volatility),
            'spread_crossing_rate': float(spread_crossing_rate),
            'avg_effective_spread': float(avg_effective_spread),
            'spread_efficiency': float(spread_efficiency),
            'avg_time_between_trades': float(avg_time_between_trades),
            'time_variability': float(time_variability),
            'price_reversal_rate': float(price_reversal_rate)
        }
    
    def _detect_spread_anomalies(self, symbol: str, current_metrics: Dict[str, Any], historical_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect spread anomalies by comparing current metrics with historical data.
        
        Parameters:
        - symbol: Trading symbol
        - current_metrics: Current spread metrics
        - historical_metrics: Historical spread metrics
        
        Returns:
        - Dictionary with detected anomalies
        """
        if not current_metrics or not historical_metrics:
            return {
                'anomalies_detected': 0,
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
            
        historical_values = {}
        for metric in current_metrics:
            historical_values[metric] = [h[metric] for h in historical_metrics if metric in h]
        
        anomalies = {}
        
        for metric, values in historical_values.items():
            if not values or metric not in current_metrics:
                continue
                
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value > 0:
                z_score = (current_metrics[metric] - mean_value) / std_value
            else:
                z_score = 0
            
            if abs(z_score) > 2.5:  # Significant deviation
                anomalies[metric] = {
                    'current_value': float(current_metrics[metric]),
                    'mean_value': float(mean_value),
                    'std_value': float(std_value),
                    'z_score': float(z_score),
                    'direction': 'up' if z_score > 0 else 'down'
                }
        
        if anomalies:
            metric_weights = {
                'spread_pct': 0.1,
                'volume_weighted_spread_pct': 0.15,
                'depth_imbalance': 0.2,
                'trade_flow_imbalance': 0.2,
                'micro_volatility': 0.1,
                'spread_crossing_rate': 0.05,
                'avg_effective_spread': 0.05,
                'spread_efficiency': 0.05,
                'time_variability': 0.05,
                'price_reversal_rate': 0.05
            }
            
            weighted_score = 0
            total_weight = 0
            
            for metric, anomaly in anomalies.items():
                if metric in metric_weights:
                    weight = metric_weights[metric]
                    weighted_score += abs(anomaly['z_score']) * weight
                    total_weight += weight
            
            if total_weight > 0:
                anomaly_score = weighted_score / total_weight
            else:
                anomaly_score = 0
            
            direction_signals = []
            
            if 'depth_imbalance' in anomalies:
                direction_signals.append(1 if anomalies['depth_imbalance']['direction'] == 'up' else -1)
            
            if 'trade_flow_imbalance' in anomalies:
                direction_signals.append(1 if anomalies['trade_flow_imbalance']['direction'] == 'up' else -1)
            
            if 'spread_efficiency' in anomalies:
                direction_signals.append(1 if anomalies['spread_efficiency']['direction'] == 'up' else -1)
            
            if direction_signals:
                avg_direction = sum(direction_signals) / len(direction_signals)
                
                if avg_direction > 0.3:
                    direction = 'BUY'
                elif avg_direction < -0.3:
                    direction = 'SELL'
                else:
                    direction = 'NEUTRAL'
            else:
                direction = 'NEUTRAL'
            
            confidence = min(0.7 + anomaly_score * 0.1, 0.99)  # Cap at 0.99
            
            return {
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies,
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'direction': direction
            }
        else:
            return {
                'anomalies_detected': 0,
                'confidence': 0.0,
                'direction': 'NEUTRAL'
            }
    
    def update_spread_metrics(self, symbol: str) -> None:
        """
        Update the spread metrics for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        """
        current_time = datetime.now()
        
        if symbol not in self.spread_anomalies:
            self.spread_anomalies[symbol] = {
                'historical_metrics': [],
                'last_update': current_time - self.update_interval,  # Force update first time
                'anomalies': {}
            }
        
        if current_time - self.spread_anomalies[symbol]['last_update'] < self.update_interval:
            return
            
        self.spread_anomalies[symbol]['last_update'] = current_time
        
        order_book = self._fetch_order_book_data(symbol)
        trades = self._fetch_trades_data(symbol)
        
        if not order_book or 'bids' not in order_book or 'asks' not in order_book or not trades:
            return
            
        metrics = self._calculate_micro_spread_metrics(order_book, trades)
        
        if not metrics:
            return
            
        metrics['timestamp'] = current_time.isoformat()
        
        self.spread_anomalies[symbol]['historical_metrics'].append(metrics)
        
        max_history = 100
        if len(self.spread_anomalies[symbol]['historical_metrics']) > max_history:
            self.spread_anomalies[symbol]['historical_metrics'] = self.spread_anomalies[symbol]['historical_metrics'][-max_history:]
        
        self.logger.info(f"Updated spread metrics for {symbol}")
    
    def detect_resonance(self, symbol: str) -> Dict[str, Any]:
        """
        Detect spread resonance for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with resonance detection results
        """
        try:
            self.update_spread_metrics(symbol)
            
            if symbol not in self.spread_anomalies:
                return {
                    'symbol': symbol,
                    'anomalies_detected': 0,
                    'confidence': 0.0,
                    'direction': 'NEUTRAL',
                    'timestamp': datetime.now().isoformat()
                }
            
            historical_metrics = self.spread_anomalies[symbol]['historical_metrics']
            
            if not historical_metrics or len(historical_metrics) < 10:
                return {
                    'symbol': symbol,
                    'anomalies_detected': 0,
                    'confidence': 0.0,
                    'direction': 'NEUTRAL',
                    'timestamp': datetime.now().isoformat()
                }
            
            current_metrics = historical_metrics[-1]
            
            previous_metrics = historical_metrics[:-1]
            
            anomalies = self._detect_spread_anomalies(symbol, current_metrics, previous_metrics)
            
            self.spread_anomalies[symbol]['anomalies'] = anomalies
            
            return {
                'symbol': symbol,
                'anomalies_detected': anomalies['anomalies_detected'],
                'confidence': anomalies['confidence'],
                'direction': anomalies['direction'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting resonance: {str(e)}")
            return {
                'symbol': symbol,
                'anomalies_detected': 0,
                'confidence': 0.0,
                'direction': 'NEUTRAL',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def resonate(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect spread resonance to generate trading signals.
        
        Parameters:
        - symbol: Trading symbol
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with trading signal
        """
        try:
            resonance = self.detect_resonance(symbol)
            
            signal = resonance['direction']
            confidence = resonance['confidence']
            
            if confidence >= self.confidence_threshold and signal in ['BUY', 'SELL']:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': float(confidence),
                    'anomalies_detected': int(resonance['anomalies_detected']),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error resonating: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Shadow Spread Resonator.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'anomaly_detection_accuracy': float(self.performance['anomaly_detection_accuracy']),
            'prediction_accuracy': float(self.performance['prediction_accuracy']),
            'average_lead_time': float(self.performance['average_lead_time']),
            'successful_trades': int(self.performance['successful_trades']),
            'symbols_analyzed': len(self.spread_anomalies),
            'timestamp': datetime.now().isoformat()
        }
