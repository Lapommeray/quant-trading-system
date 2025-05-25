"""
Arbitrage Synapse Chain (ASC)

Builds a self-healing arbitrage chain across multiple assets and timelines.
Result: No trade is taken unless the entire chain confirms.
True Edge: You profit from the hidden tension between symbols.
"""

import numpy as np
import pandas as pd
import ccxt
import logging
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

class ArbitrageSynapseChain:
    """
    Arbitrage Synapse Chain (ASC) module that builds a self-healing arbitrage chain
    across multiple assets and timelines.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Arbitrage Synapse Chain module.
        
        Parameters:
        - algorithm: Optional algorithm configuration
        """
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.algorithm = algorithm
        self.logger = logging.getLogger('ASC')
        self.arbitrage_chains = {}
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=5)
        self.confidence_threshold = 0.95  # Super high confidence as requested
        
        self.performance = {
            'chain_detection_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'average_profit_per_chain': 0.0,
            'successful_trades': 0
        }
    
    def _fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fetch ticker data for multiple symbols.
        
        Parameters:
        - symbols: List of trading symbols (optional)
        
        Returns:
        - Dictionary with ticker data
        """
        try:
            if symbols:
                tickers = {}
                for symbol in symbols:
                    tickers[symbol] = self.exchange.fetch_ticker(symbol)
                return tickers
            else:
                return self.exchange.fetch_tickers()
        except Exception as e:
            self.logger.error(f"Error fetching tickers: {str(e)}")
            return {}
    
    def _fetch_order_books(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch order book data for multiple symbols.
        
        Parameters:
        - symbols: List of trading symbols
        
        Returns:
        - Dictionary with order book data
        """
        order_books = {}
        
        for symbol in symbols:
            try:
                order_book = self.exchange.fetch_order_book(symbol)
                order_books[symbol] = order_book
            except Exception as e:
                self.logger.error(f"Error fetching order book for {symbol}: {str(e)}")
        
        return order_books
    
    def _build_arbitrage_graph(self, tickers: Dict[str, Dict[str, Any]], order_books: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a directed graph representing arbitrage opportunities.
        
        Parameters:
        - tickers: Dictionary with ticker data
        - order_books: Dictionary with order book data
        
        Returns:
        - Directed graph with arbitrage opportunities
        """
        G = nx.DiGraph()
        
        currencies = set()
        symbol_pairs = {}
        
        for symbol, ticker in tickers.items():
            if 'symbol' not in ticker:
                continue
                
            parts = symbol.split('/')
            
            if len(parts) != 2:
                continue
                
            base, quote = parts
            
            currencies.add(base)
            currencies.add(quote)
            
            symbol_pairs[symbol] = (base, quote)
        
        for currency in currencies:
            G.add_node(currency)
        
        for symbol, (base, quote) in symbol_pairs.items():
            if symbol not in tickers:
                continue
                
            ticker = tickers[symbol]
            
            if 'bid' not in ticker or 'ask' not in ticker:
                continue
                
            bid = ticker['bid']  # Highest buy price
            ask = ticker['ask']  # Lowest sell price
            
            if symbol in order_books and order_books[symbol]['bids'] and order_books[symbol]['asks']:
                bid = float(order_books[symbol]['bids'][0][0])
                ask = float(order_books[symbol]['asks'][0][0])
            
            if bid > 0:
                G.add_edge(quote, base, symbol=symbol, rate=1/ask, type='buy')
            
            if ask > 0:
                G.add_edge(base, quote, symbol=symbol, rate=bid, type='sell')
        
        return G
    
    def _find_arbitrage_cycles(self, G: nx.DiGraph, start_currency: str, max_length: int = 4) -> List[Dict[str, Any]]:
        """
        Find arbitrage cycles in the graph.
        
        Parameters:
        - G: Directed graph with arbitrage opportunities
        - start_currency: Starting currency for cycles
        - max_length: Maximum cycle length
        
        Returns:
        - List of arbitrage cycles
        """
        cycles = []
        
        for length in range(2, max_length + 1):
            for path in nx.simple_cycles(G):
                if len(path) != length:
                    continue
                    
                if start_currency not in path:
                    continue
                    
                while path[0] != start_currency:
                    path = path[1:] + [path[0]]
                
                profit_rate = 1.0
                edges = []
                
                for i in range(len(path)):
                    curr = path[i]
                    next_curr = path[(i + 1) % len(path)]
                    
                    if not G.has_edge(curr, next_curr):
                        profit_rate = 0.0
                        break
                        
                    edge_data = G.get_edge_data(curr, next_curr)
                    profit_rate *= edge_data['rate']
                    
                    edges.append({
                        'from': curr,
                        'to': next_curr,
                        'symbol': edge_data['symbol'],
                        'rate': edge_data['rate'],
                        'type': edge_data['type']
                    })
                
                profit_pct = (profit_rate - 1) * 100
                
                if profit_rate > 1.001:  # At least 0.1% profit
                    cycles.append({
                        'path': path,
                        'profit_rate': float(profit_rate),
                        'profit_pct': float(profit_pct),
                        'edges': edges,
                        'length': len(path)
                    })
        
        cycles = sorted(cycles, key=lambda x: x['profit_pct'], reverse=True)
        
        return cycles
    
    def _validate_arbitrage_chain(self, cycle: Dict[str, Any], order_books: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate an arbitrage chain using order book data.
        
        Parameters:
        - cycle: Arbitrage cycle
        - order_books: Dictionary with order book data
        
        Returns:
        - Validated arbitrage chain
        """
        edges = cycle['edges']
        
        realistic_profit_rate = 1.0
        total_volume = float('inf')
        
        for edge in edges:
            symbol = edge['symbol']
            edge_type = edge['type']
            
            if symbol not in order_books:
                return {
                    'valid': False,
                    'reason': f"Missing order book for {symbol}"
                }
            
            order_book = order_books[symbol]
            
            if not order_book['bids'] or not order_book['asks']:
                return {
                    'valid': False,
                    'reason': f"Empty order book for {symbol}"
                }
            
            if edge_type == 'buy':
                price = float(order_book['asks'][0][0])
                volume = float(order_book['asks'][0][1])
                rate = 1 / price
            else:
                price = float(order_book['bids'][0][0])
                volume = float(order_book['bids'][0][1])
                rate = price
            
            edge['realistic_rate'] = float(rate)
            
            realistic_profit_rate *= rate
            
            total_volume = min(total_volume, volume)
        
        realistic_profit_pct = (realistic_profit_rate - 1) * 100
        
        if realistic_profit_rate > 1.001:  # At least 0.1% profit
            return {
                'valid': True,
                'realistic_profit_rate': float(realistic_profit_rate),
                'realistic_profit_pct': float(realistic_profit_pct),
                'max_volume': float(total_volume)
            }
        else:
            return {
                'valid': False,
                'reason': "Not profitable after considering order book depth",
                'realistic_profit_pct': float(realistic_profit_pct)
            }
    
    def _build_multi_timeframe_confirmation(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Build multi-timeframe confirmation for symbols.
        
        Parameters:
        - symbols: List of trading symbols
        
        Returns:
        - Dictionary with multi-timeframe confirmation
        """
        timeframes = ['1m', '5m', '15m', '1h']
        confirmations = {}
        
        for symbol in symbols:
            symbol_confirmations = {}
            
            for timeframe in timeframes:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=20)
                    
                    if not ohlcv or len(ohlcv) < 10:
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    short_ma = df['close'].rolling(window=5).mean().iloc[-1]
                    long_ma = df['close'].rolling(window=10).mean().iloc[-1]
                    
                    if short_ma > long_ma:
                        trend = 'up'
                    elif short_ma < long_ma:
                        trend = 'down'
                    else:
                        trend = 'neutral'
                    
                    volatility = df['close'].pct_change().std() * 100
                    
                    avg_volume = df['volume'].mean()
                    current_volume = df['volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    symbol_confirmations[timeframe] = {
                        'trend': trend,
                        'volatility': float(volatility),
                        'volume_ratio': float(volume_ratio)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {str(e)}")
            
            if symbol_confirmations:
                confirmations[symbol] = symbol_confirmations
        
        return confirmations
    
    def _calculate_chain_confidence(self, cycle: Dict[str, Any], validation: Dict[str, Any], confirmations: Dict[str, Any]) -> float:
        """
        Calculate confidence for an arbitrage chain.
        
        Parameters:
        - cycle: Arbitrage cycle
        - validation: Validation results
        - confirmations: Multi-timeframe confirmations
        
        Returns:
        - Confidence score
        """
        if not validation['valid']:
            return 0.0
            
        profit_pct = validation['realistic_profit_pct']
        
        if profit_pct < 0.1:
            base_confidence = 0.0
        elif profit_pct < 0.5:
            base_confidence = 0.7
        elif profit_pct < 1.0:
            base_confidence = 0.8
        else:
            base_confidence = 0.9
        
        if not confirmations:
            return base_confidence
            
        confirmation_score = 0.0
        total_weight = 0.0
        
        timeframe_weights = {
            '1m': 0.2,
            '5m': 0.3,
            '15m': 0.3,
            '1h': 0.2
        }
        
        for edge in cycle['edges']:
            symbol = edge['symbol']
            
            if symbol not in confirmations:
                continue
                
            symbol_score = 0.0
            symbol_weight = 0.0
            
            for timeframe, weight in timeframe_weights.items():
                if timeframe not in confirmations[symbol]:
                    continue
                    
                tf_confirmation = confirmations[symbol][timeframe]
                
                if (edge['type'] == 'buy' and tf_confirmation['trend'] == 'up') or \
                   (edge['type'] == 'sell' and tf_confirmation['trend'] == 'down'):
                    symbol_score += weight
                
                symbol_weight += weight
            
            if symbol_weight > 0:
                confirmation_score += symbol_score / symbol_weight
                total_weight += 1.0
        
        if total_weight > 0:
            avg_confirmation = confirmation_score / total_weight
            
            adjusted_confidence = base_confidence * (0.8 + avg_confirmation * 0.2)
        else:
            adjusted_confidence = base_confidence
        
        return min(adjusted_confidence, 0.99)
    
    def update_arbitrage_chains(self, base_currency: str = 'USDT', symbols: Optional[List[str]] = None) -> None:
        """
        Update the arbitrage chains.
        
        Parameters:
        - base_currency: Base currency for arbitrage
        - symbols: List of trading symbols (optional)
        """
        current_time = datetime.now()
        
        if current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        tickers = self._fetch_tickers(symbols)
        
        if not tickers:
            return
            
        ticker_symbols = list(tickers.keys())
        
        order_books = self._fetch_order_books(ticker_symbols)
        
        if not order_books:
            return
            
        G = self._build_arbitrage_graph(tickers, order_books)
        
        if not G.nodes():
            return
            
        cycles = self._find_arbitrage_cycles(G, base_currency)
        
        if not cycles:
            return
            
        cycle_symbols = set()
        for cycle in cycles:
            for edge in cycle['edges']:
                cycle_symbols.add(edge['symbol'])
        
        confirmations = self._build_multi_timeframe_confirmation(list(cycle_symbols))
        
        validated_chains = []
        
        for cycle in cycles:
            validation = self._validate_arbitrage_chain(cycle, order_books)
            
            if validation['valid']:
                confidence = self._calculate_chain_confidence(cycle, validation, confirmations)
                
                validated_chain = {
                    'path': cycle['path'],
                    'edges': cycle['edges'],
                    'profit_pct': float(validation['realistic_profit_pct']),
                    'max_volume': float(validation['max_volume']),
                    'confidence': float(confidence),
                    'timestamp': current_time.isoformat()
                }
                
                validated_chains.append(validated_chain)
        
        self.arbitrage_chains = {
            'base_currency': base_currency,
            'chains': validated_chains,
            'timestamp': current_time.isoformat()
        }
        
        self.logger.info(f"Updated arbitrage chains: {len(validated_chains)} valid chains found")
    
    def find_arbitrage(self, base_currency: str = 'USDT') -> Dict[str, Any]:
        """
        Find arbitrage opportunities.
        
        Parameters:
        - base_currency: Base currency for arbitrage
        
        Returns:
        - Dictionary with arbitrage opportunities
        """
        try:
            self.update_arbitrage_chains(base_currency)
            
            if not self.arbitrage_chains or 'chains' not in self.arbitrage_chains:
                return {
                    'base_currency': base_currency,
                    'chains_found': 0,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            chains = self.arbitrage_chains['chains']
            
            if not chains:
                return {
                    'base_currency': base_currency,
                    'chains_found': 0,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            best_chain = max(chains, key=lambda x: x['confidence'])
            
            return {
                'base_currency': base_currency,
                'chains_found': len(chains),
                'best_chain': best_chain,
                'confidence': float(best_chain['confidence']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage: {str(e)}")
            return {
                'base_currency': base_currency,
                'chains_found': 0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def chain_arbitrage(self, base_currency: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on arbitrage chains.
        
        Parameters:
        - base_currency: Base currency for arbitrage
        - market_data: Market data dictionary
        
        Returns:
        - Dictionary with trading signal
        """
        try:
            arbitrage = self.find_arbitrage(base_currency)
            
            if arbitrage['chains_found'] == 0 or 'best_chain' not in arbitrage:
                return {
                    'base_currency': base_currency,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            best_chain = arbitrage['best_chain']
            confidence = best_chain['confidence']
            
            if confidence >= self.confidence_threshold:
                return {
                    'base_currency': base_currency,
                    'signal': 'ARBITRAGE',
                    'confidence': float(confidence),
                    'profit_pct': float(best_chain['profit_pct']),
                    'path': best_chain['path'],
                    'edges': best_chain['edges'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'base_currency': base_currency,
                    'signal': 'NEUTRAL',
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error in chain arbitrage: {str(e)}")
            return {
                'base_currency': base_currency,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Arbitrage Synapse Chain.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            'chain_detection_accuracy': float(self.performance['chain_detection_accuracy']),
            'prediction_accuracy': float(self.performance['prediction_accuracy']),
            'average_profit_per_chain': float(self.performance['average_profit_per_chain']),
            'successful_trades': int(self.performance['successful_trades']),
            'timestamp': datetime.now().isoformat()
        }
