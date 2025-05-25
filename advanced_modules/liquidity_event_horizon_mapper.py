"""
Liquidity Event Horizon Mapper (LEHM)

AI models the "gravitational pull" of major liquidity zones before price reacts.
Result: You enter where others get liquidated.
True Edge: It's not supply and demand — it's mass and collapse.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d

class LiquidityEventHorizonMapper:
    """
    Liquidity Event Horizon Mapper (LEHM) module that models the "gravitational pull"
    of major liquidity zones before price reacts.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Liquidity Event Horizon Mapper module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('LEHM')
        self.liquidity_zones = {}
        self.event_horizons = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=15)
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'zone_detection_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'average_lead_time': 0.0,
            'successful_trades': 0
        }
    
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
    
    def _fetch_liquidation_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch liquidation data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - List of liquidation events
        """
        
        try:
            trades = self.exchange.fetch_trades(symbol, limit=1000)
            
            if not trades:
                return []
                
            liquidations = []
            
            for i in range(1, len(trades)):
                current_trade = trades[i]
                prev_trade = trades[i-1]
                
                price_change = abs(current_trade['price'] - prev_trade['price']) / prev_trade['price']
                
                if current_trade['amount'] > np.mean([t['amount'] for t in trades]) * 3 and price_change > 0.001:
                    liquidation = {
                        'timestamp': current_trade['timestamp'],
                        'datetime': current_trade['datetime'],
                        'price': current_trade['price'],
                        'amount': current_trade['amount'],
                        'side': current_trade['side'],
                        'price_impact': price_change
                    }
                    
                    liquidations.append(liquidation)
            
            return liquidations
            
        except Exception as e:
            self.logger.error(f"Error fetching liquidation data: {str(e)}")
            return []
    
    def _detect_liquidity_zones(self, order_book: Dict[str, Any], liquidations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect liquidity zones from order book and liquidation data.
        
        Parameters:
        - order_book: Order book data
        - liquidations: Liquidation data
        
        Returns:
        - List of liquidity zones
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return []
            
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return []
            
        bid_prices = np.array([float(bid[0]) for bid in bids])
        bid_volumes = np.array([float(bid[1]) for bid in bids])
        ask_prices = np.array([float(ask[0]) for ask in asks])
        ask_volumes = np.array([float(ask[1]) for ask in asks])
        
        vwap_bids = np.sum(bid_prices * bid_volumes) / np.sum(bid_volumes)
        vwap_asks = np.sum(ask_prices * ask_volumes) / np.sum(ask_volumes)
        
        bid_volume_profile = gaussian_filter1d(bid_volumes, sigma=2)
        ask_volume_profile = gaussian_filter1d(ask_volumes, sigma=2)
        
        bid_peaks = self._find_peaks(bid_volume_profile, threshold=np.mean(bid_volume_profile) * 2)
        ask_peaks = self._find_peaks(ask_volume_profile, threshold=np.mean(ask_volume_profile) * 2)
        
        liquidity_zones = []
        
        for peak_idx in bid_peaks:
            if peak_idx < len(bid_prices):
                zone = {
                    'price': float(bid_prices[peak_idx]),
                    'volume': float(bid_volumes[peak_idx]),
                    'type': 'bid',
                    'strength': float(bid_volumes[peak_idx] / np.mean(bid_volumes)),
                    'distance_from_vwap': float(abs(bid_prices[peak_idx] - vwap_bids) / vwap_bids)
                }
                
                liquidity_zones.append(zone)
        
        for peak_idx in ask_peaks:
            if peak_idx < len(ask_prices):
                zone = {
                    'price': float(ask_prices[peak_idx]),
                    'volume': float(ask_volumes[peak_idx]),
                    'type': 'ask',
                    'strength': float(ask_volumes[peak_idx] / np.mean(ask_volumes)),
                    'distance_from_vwap': float(abs(ask_prices[peak_idx] - vwap_asks) / vwap_asks)
                }
                
                liquidity_zones.append(zone)
        
        if liquidations:
            liquidation_prices = [l['price'] for l in liquidations]
            liquidation_volumes = [l['amount'] for l in liquidations]
            
            clusters = self._cluster_liquidations(liquidation_prices, liquidation_volumes)
            
            for cluster in clusters:
                avg_price = np.mean([liquidation_prices[i] for i in cluster])
                total_volume = np.sum([liquidation_volumes[i] for i in cluster])
                
                zone = {
                    'price': float(avg_price),
                    'volume': float(total_volume),
                    'type': 'liquidation',
                    'strength': float(total_volume / np.mean(liquidation_volumes) if liquidation_volumes else 1.0),
                    'distance_from_vwap': float(abs(avg_price - vwap_bids) / vwap_bids if avg_price < vwap_bids else abs(avg_price - vwap_asks) / vwap_asks)
                }
                
                liquidity_zones.append(zone)
        
        liquidity_zones = sorted(liquidity_zones, key=lambda x: x['strength'], reverse=True)
        
        return liquidity_zones
    
    def _find_peaks(self, data: np.ndarray, threshold: float) -> List[int]:
        """
        Find peaks in data above a threshold.
        
        Parameters:
        - data: Data array
        - threshold: Threshold for peak detection
        
        Returns:
        - List of peak indices
        """
        peaks = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        
        return peaks
    
    def _cluster_liquidations(self, prices: List[float], volumes: List[float]) -> List[List[int]]:
        """
        Cluster liquidations by price proximity.
        
        Parameters:
        - prices: List of liquidation prices
        - volumes: List of liquidation volumes
        
        Returns:
        - List of clusters (each cluster is a list of indices)
        """
        if not prices:
            return []
            
        price_diffs = np.diff(sorted(prices))
        
        threshold = np.mean(price_diffs) * 2
        
        clusters = []
        current_cluster = [0]
        
        for i in range(1, len(prices)):
            if abs(prices[i] - prices[current_cluster[-1]]) < threshold:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_gravitational_pull(self, current_price: float, liquidity_zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate the gravitational pull of liquidity zones on the current price.
        
        Parameters:
        - current_price: Current price
        - liquidity_zones: List of liquidity zones
        
        Returns:
        - List of gravitational pull calculations
        """
        if not liquidity_zones:
            return []
            
        gravitational_pulls = []
        
        for zone in liquidity_zones:
            distance_pct = abs(zone['price'] - current_price) / current_price
            
            if distance_pct > 0:
                pull = zone['strength'] / (distance_pct ** 2)
            else:
                pull = zone['strength'] * 100  # Very high pull for zones at current price
            
            direction = 'up' if zone['price'] > current_price else 'down'
            
            time_to_impact = distance_pct * 100 / (pull + 1e-10)  # Rough estimate in arbitrary units
            
            gravitational_pulls.append({
                'zone_price': float(zone['price']),
                'zone_type': zone['type'],
                'zone_strength': float(zone['strength']),
                'distance_pct': float(distance_pct),
                'pull': float(pull),
                'direction': direction,
                'time_to_impact': float(time_to_impact)
            })
        
        gravitational_pulls = sorted(gravitational_pulls, key=lambda x: x['pull'], reverse=True)
        
        return gravitational_pulls
    
    def _detect_event_horizons(self, current_price: float, gravitational_pulls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect event horizons where price is likely to be pulled towards.
        
        Parameters:
        - current_price: Current price
        - gravitational_pulls: List of gravitational pull calculations
        
        Returns:
        - List of event horizons
        """
        if not gravitational_pulls:
            return []
            
        event_horizons = []
        
        up_pulls = [p for p in gravitational_pulls if p['direction'] == 'up']
        down_pulls = [p for p in gravitational_pulls if p['direction'] == 'down']
        
        net_up_pull = sum(p['pull'] for p in up_pulls)
        net_down_pull = sum(p['pull'] for p in down_pulls)
        
        if net_up_pull > net_down_pull * 1.5:  # Significant upward bias
            dominant_direction = 'up'
            dominant_pulls = up_pulls
        elif net_down_pull > net_up_pull * 1.5:  # Significant downward bias
            dominant_direction = 'down'
            dominant_pulls = down_pulls
        else:
            dominant_direction = 'neutral'
            dominant_pulls = gravitational_pulls[:3]  # Top 3 pulls regardless of direction
        
        for pull in dominant_pulls:
            base_confidence = min(0.5 + pull['pull'] / 100, 0.9)
            
            if pull['zone_type'] == 'liquidation':
                confidence = min(base_confidence + 0.1, 0.99)  # Boost for liquidation zones
            else:
                confidence = base_confidence
            
            if confidence >= 0.7:
                event_horizon = {
                    'price': float(pull['zone_price']),
                    'direction': pull['direction'],
                    'pull': float(pull['pull']),
                    'time_to_impact': float(pull['time_to_impact']),
                    'confidence': float(confidence),
                    'zone_type': pull['zone_type']
                }
                
                event_horizons.append(event_horizon)
        
        return event_horizons
    
    def update_liquidity_zones(self, symbol: str) -> None:
        """
        Update the liquidity zones for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        """
        current_time = datetime.now()
        
        if symbol in self.liquidity_zones and current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        order_book = self._fetch_order_book_data(symbol)
        
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return
            
        liquidations = self._fetch_liquidation_data(symbol)
        
        zones = self._detect_liquidity_zones(order_book, liquidations)
        
        if not zones:
            return
            
        self.liquidity_zones[symbol] = {
            'timestamp': current_time.isoformat(),
            'zones': zones,
            'current_price': float((order_book['bids'][0][0] + order_book['asks'][0][0]) / 2) if order_book['bids'] and order_book['asks'] else 0.0
        }
        
        self.logger.info(f"Updated liquidity zones for {symbol}: {len(zones)} zones detected")
    
    def map_event_horizons(self, symbol: str) -> Dict[str, Any]:
        """
        Map event horizons for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with event horizon mapping results
        """
        try:
            self.update_liquidity_zones(symbol)
            
            if symbol not in self.liquidity_zones:
                return {
                    'symbol': symbol,
                    'event_horizons': [],
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            current_price = self.liquidity_zones[symbol]['current_price']
            zones = self.liquidity_zones[symbol]['zones']
            
            if not zones or current_price == 0:
                return {
                    'symbol': symbol,
                    'event_horizons': [],
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            gravitational_pulls = self._calculate_gravitational_pull(current_price, zones)
            
            if not gravitational_pulls:
                return {
                    'symbol': symbol,
                    'event_horizons': [],
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            event_horizons = self._detect_event_horizons(current_price, gravitational_pulls)
            
            self.event_horizons[symbol] = {
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'event_horizons': event_horizons
            }
            
            if event_horizons:
                confidence = max(horizon['confidence'] for horizon in event_horizons)
            else:
                confidence = 0.0
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'event_horizons': event_horizons,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping event horizons: {str(e)}")
            return {
                'symbol': symbol,
                'event_horizons': [],
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def map_liquidity(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map liquidity event horizons to generate trading signals.
        
        Parameters:
        - symbol: Trading symbol
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with trading signal
        """
        try:
            mapping = self.map_event_horizons(symbol)
            
            if not mapping['event_horizons']:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            current_price = mapping['current_price']
            
            strongest_horizon = max(mapping['event_horizons'], key=lambda x: x['confidence'])
            
            signal = 'NEUTRAL'
            confidence = strongest_horizon['confidence']
            
            if strongest_horizon['direction'] == 'up' and confidence >= self.confidence_threshold:
                signal = 'BUY'
            elif strongest_horizon['direction'] == 'down' and confidence >= self.confidence_threshold:
                signal = 'SELL'
            
            if confidence >= self.confidence_threshold and signal in ['BUY', 'SELL']:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': float(confidence),
                    'target_price': float(strongest_horizon['price']),
                    'current_price': float(current_price),
                    'price_distance_pct': float(abs(strongest_horizon['price'] - current_price) / current_price * 100),
                    'zone_type': strongest_horizon['zone_type'],
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
            self.logger.error(f"Error mapping liquidity: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Liquidity Event Horizon Mapper.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'zone_detection_accuracy': float(self.performance['zone_detection_accuracy']),
            'prediction_accuracy': float(self.performance['prediction_accuracy']),
            'average_lead_time': float(self.performance['average_lead_time']),
            'successful_trades': int(self.performance['successful_trades']),
            'symbols_analyzed': len(self.liquidity_zones),
            'timestamp': datetime.now().isoformat()
        }
