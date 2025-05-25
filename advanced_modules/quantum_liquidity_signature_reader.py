"""
Quantum Liquidity Signature Reader (QLSR)

An AI that detects the unique "liquidity fingerprint" of every major market maker.
Know which whale is active — and what they'll do next — by how the book "feels."
True Edge: Never chase; always reverse-engineer their trap.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

class QuantumLiquiditySignatureReader:
    """
    Quantum Liquidity Signature Reader (QLSR) module that detects the unique
    "liquidity fingerprint" of every major market maker.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Quantum Liquidity Signature Reader module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('QLSR')
        self.signature_database = {}
        self.active_whales = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=15)
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'signature_detection_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'unique_signatures_detected': 0,
            'successful_trades': 0
        }
    
    def _calculate_order_book_fingerprint(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a fingerprint from order book data.
        
        Parameters:
        - order_book: Order book data
        
        Returns:
        - Fingerprint dictionary
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {
                'spread': 0.0,
                'volume_imbalance': 0.0,
                'bid_clustering': 0.0,
                'ask_clustering': 0.0,
                'bid_volume_distribution': 0.0,
                'ask_volume_distribution': 0.0,
                'large_bid_presence': 0.0,
                'large_ask_presence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'hash': hashlib.md5(b'empty_orderbook').hexdigest()
            }
            
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids or not asks:
            return {
                'spread': 0.0,
                'volume_imbalance': 0.0,
                'bid_clustering': 0.0,
                'ask_clustering': 0.0,
                'bid_volume_distribution': 0.0,
                'ask_volume_distribution': 0.0,
                'large_bid_presence': 0.0,
                'large_ask_presence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'hash': hashlib.md5(b'empty_bids_asks').hexdigest()
            }
            
        bid_prices = np.array([float(bid[0]) for bid in bids])
        bid_volumes = np.array([float(bid[1]) for bid in bids])
        ask_prices = np.array([float(ask[0]) for ask in asks])
        ask_volumes = np.array([float(ask[1]) for ask in asks])
        
        spread = float(ask_prices[0] - bid_prices[0]) if len(ask_prices) > 0 and len(bid_prices) > 0 else 0.0
        
        total_bid_volume = np.sum(bid_volumes)
        total_ask_volume = np.sum(ask_volumes)
        volume_imbalance = float((total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-10))
        
        bid_price_diffs = np.diff(bid_prices)
        ask_price_diffs = np.diff(ask_prices)
        
        bid_clustering = float(np.std(bid_price_diffs)) if len(bid_price_diffs) > 0 else 0.0
        ask_clustering = float(np.std(ask_price_diffs)) if len(ask_price_diffs) > 0 else 0.0
        
        bid_volume_distribution = float(np.std(bid_volumes) / (np.mean(bid_volumes) + 1e-10)) if len(bid_volumes) > 0 else 0.0
        ask_volume_distribution = float(np.std(ask_volumes) / (np.mean(ask_volumes) + 1e-10)) if len(ask_volumes) > 0 else 0.0
        
        large_bid_threshold = np.percentile(bid_volumes, 90) if len(bid_volumes) > 10 else np.max(bid_volumes)
        large_ask_threshold = np.percentile(ask_volumes, 90) if len(ask_volumes) > 10 else np.max(ask_volumes)
        
        large_bid_presence = float(np.sum(bid_volumes > large_bid_threshold) / len(bid_volumes)) if len(bid_volumes) > 0 else 0.0
        large_ask_presence = float(np.sum(ask_volumes > large_ask_threshold) / len(ask_volumes)) if len(ask_volumes) > 0 else 0.0
        
        fingerprint = {
            'spread': spread,
            'volume_imbalance': volume_imbalance,
            'bid_clustering': bid_clustering,
            'ask_clustering': ask_clustering,
            'bid_volume_distribution': bid_volume_distribution,
            'ask_volume_distribution': ask_volume_distribution,
            'large_bid_presence': large_bid_presence,
            'large_ask_presence': large_ask_presence,
            'timestamp': datetime.now().isoformat()
        }
        
        fingerprint_str = f"{spread:.8f}_{volume_imbalance:.8f}_{bid_clustering:.8f}_{ask_clustering:.8f}_{bid_volume_distribution:.8f}_{ask_volume_distribution:.8f}_{large_bid_presence:.8f}_{large_ask_presence:.8f}"
        fingerprint_hash = hashlib.md5(fingerprint_str.encode()).hexdigest()
        
        fingerprint['hash'] = fingerprint_hash
        
        return fingerprint
    
    def _fetch_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
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
    
    def _match_fingerprint(self, fingerprint: Dict[str, Any]) -> Tuple[str, float]:
        """
        Match a fingerprint against the signature database.
        
        Parameters:
        - fingerprint: Fingerprint to match
        
        Returns:
        - Tuple of (whale_id, match_score)
        """
        if not fingerprint or not self.signature_database:
            return ('unknown', 0.0)
            
        best_match = 'unknown'
        best_score = 0.0
        
        for whale_id, signatures in self.signature_database.items():
            for signature in signatures:
                score = 0.0
                
                spread_diff = abs(fingerprint['spread'] - signature['spread']) / (signature['spread'] + 1e-10)
                volume_imbalance_diff = abs(fingerprint['volume_imbalance'] - signature['volume_imbalance'])
                bid_clustering_diff = abs(fingerprint['bid_clustering'] - signature['bid_clustering']) / (signature['bid_clustering'] + 1e-10)
                ask_clustering_diff = abs(fingerprint['ask_clustering'] - signature['ask_clustering']) / (signature['ask_clustering'] + 1e-10)
                
                diff_score = spread_diff * 0.3 + volume_imbalance_diff * 0.3 + bid_clustering_diff * 0.2 + ask_clustering_diff * 0.2
                
                similarity_score = 1.0 / (1.0 + diff_score)
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = whale_id
        
        return (best_match, best_score)
    
    def _predict_whale_action(self, whale_id: str, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the next action of a whale based on their signature.
        
        Parameters:
        - whale_id: ID of the whale
        - fingerprint: Current fingerprint
        
        Returns:
        - Dictionary with prediction details
        """
        if whale_id == 'unknown' or whale_id not in self.signature_database:
            return {
                'whale_id': whale_id,
                'action': 'unknown',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
        signatures = self.signature_database[whale_id]
        
        if not signatures or len(signatures) < 2:
            return {
                'whale_id': whale_id,
                'action': 'unknown',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
        volume_imbalances = [sig['volume_imbalance'] for sig in signatures]
        spreads = [sig['spread'] for sig in signatures]
        
        current_imbalance = fingerprint['volume_imbalance']
        current_spread = fingerprint['spread']
        
        if current_imbalance > 0.3 and current_spread < np.mean(spreads) * 0.8:
            action = 'accumulating'  # Likely buying
            confidence = min(0.5 + abs(current_imbalance) * 0.5, 0.99)
        elif current_imbalance < -0.3 and current_spread < np.mean(spreads) * 0.8:
            action = 'distributing'  # Likely selling
            confidence = min(0.5 + abs(current_imbalance) * 0.5, 0.99)
        elif current_spread > np.mean(spreads) * 1.5:
            action = 'trapping'  # Setting a trap
            confidence = min(0.6 + current_spread / (np.mean(spreads) + 1e-10) * 0.2, 0.99)
        else:
            action = 'neutral'
            confidence = 0.5
        
        return {
            'whale_id': whale_id,
            'action': action,
            'confidence': float(confidence),
            'volume_imbalance': float(current_imbalance),
            'spread': float(current_spread),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_signature_database(self, symbols: List[str]) -> None:
        """
        Update the signature database with new order book data.
        
        Parameters:
        - symbols: List of trading symbols
        """
        current_time = datetime.now()
        
        if current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        for symbol in symbols:
            order_book = self._fetch_order_book(symbol)
            
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                continue
                
            fingerprint = self._calculate_order_book_fingerprint(order_book)
            
            if not fingerprint:
                continue
                
            whale_id, match_score = self._match_fingerprint(fingerprint)
            
            if match_score > 0.8:
                if whale_id not in self.signature_database:
                    self.signature_database[whale_id] = []
                
                self.signature_database[whale_id].append(fingerprint)
                
                if len(self.signature_database[whale_id]) > 20:
                    self.signature_database[whale_id] = self.signature_database[whale_id][-20:]
            
            elif match_score < 0.5:
                new_whale_id = f"whale_{len(self.signature_database) + 1}"
                self.signature_database[new_whale_id] = [fingerprint]
                self.performance['unique_signatures_detected'] += 1
            
            self.active_whales[symbol] = {
                'whale_id': whale_id if match_score > 0.8 else f"whale_{len(self.signature_database)}",
                'match_score': float(match_score),
                'fingerprint': fingerprint,
                'timestamp': datetime.now().isoformat()
            }
        
        self.logger.info(f"Updated signature database with {len(symbols)} symbols. Total unique whales: {len(self.signature_database)}")
    
    def detect_liquidity_signature(self, symbol: str) -> Dict[str, Any]:
        """
        Detect liquidity signatures in market data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with signature detection results
        """
        try:
            order_book = self._fetch_order_book(symbol)
            
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return {
                    'symbol': symbol,
                    'whale_id': 'unknown',
                    'match_score': 0.0,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            fingerprint = self._calculate_order_book_fingerprint(order_book)
            
            if not fingerprint:
                return {
                    'symbol': symbol,
                    'whale_id': 'unknown',
                    'match_score': 0.0,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            whale_id, match_score = self._match_fingerprint(fingerprint)
            
            prediction = self._predict_whale_action(whale_id, fingerprint)
            
            self.active_whales[symbol] = {
                'whale_id': whale_id,
                'match_score': float(match_score),
                'fingerprint': fingerprint,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'symbol': symbol,
                'whale_id': whale_id,
                'match_score': float(match_score),
                'fingerprint': {k: float(v) if isinstance(v, (int, float)) else v for k, v in fingerprint.items() if k != 'hash'},
                'prediction': prediction,
                'confidence': float(prediction['confidence']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity signature: {str(e)}")
            return {
                'symbol': symbol,
                'whale_id': 'unknown',
                'match_score': 0.0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def read_signature(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read liquidity signatures from market data to generate trading signals.
        
        Parameters:
        - symbol: Trading symbol
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with trading signal
        """
        try:
            self.update_signature_database([symbol])
            
            signature = self.detect_liquidity_signature(symbol)
            
            signal = 'NEUTRAL'
            confidence = 0.0
            
            if signature['prediction']['action'] == 'accumulating':
                signal = 'BUY'  # Follow the whale
                confidence = signature['prediction']['confidence'] * signature['match_score']
            elif signature['prediction']['action'] == 'distributing':
                signal = 'SELL'  # Follow the whale
                confidence = signature['prediction']['confidence'] * signature['match_score']
            elif signature['prediction']['action'] == 'trapping':
                if signature['fingerprint']['volume_imbalance'] > 0:
                    signal = 'SELL'  # Trap is likely to catch buyers
                else:
                    signal = 'BUY'  # Trap is likely to catch sellers
                confidence = signature['prediction']['confidence'] * signature['match_score'] * 0.8  # Lower confidence for trap reversal
            
            if confidence >= self.confidence_threshold:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': float(confidence),
                    'whale_id': signature['whale_id'],
                    'whale_action': signature['prediction']['action'],
                    'match_score': float(signature['match_score']),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'confidence': float(confidence),
                    'whale_id': signature['whale_id'],
                    'whale_action': signature['prediction']['action'],
                    'match_score': float(signature['match_score']),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error reading signature: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Quantum Liquidity Signature Reader.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'signature_detection_accuracy': float(self.performance['signature_detection_accuracy']),
            'prediction_accuracy': float(self.performance['prediction_accuracy']),
            'unique_signatures_detected': int(self.performance['unique_signatures_detected']),
            'successful_trades': int(self.performance['successful_trades']),
            'active_whales': len(self.active_whales),
            'timestamp': datetime.now().isoformat()
        }
