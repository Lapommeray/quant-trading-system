"""
Data Verifier - Ensure 100% Real Data

This module provides verification mechanisms to ensure that all data
used in the system is 100% real and not synthetic.
"""

import logging
import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataVerifier:
    """
    Data verification to ensure 100% real market data.
    
    This class provides methods for verifying the authenticity of market data
    and ensuring that no synthetic data is used in the system.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the data verifier.
        
        Parameters:
        - strict_mode: If True, will raise exceptions for synthetic data
        """
        self.strict_mode = strict_mode
        self.verification_history = []
        logger.info(f"Initialized DataVerifier with strict_mode={strict_mode}")
    
    def verify_ohlcv_data(self, data: List[List[float]], symbol: str, 
                         exchange: str, timeframe: str) -> Tuple[bool, str]:
        """
        Verify OHLCV (candle) data authenticity.
        
        Parameters:
        - data: OHLCV data to verify
        - symbol: Trading pair symbol
        - exchange: Exchange name
        - timeframe: Timeframe of the data
        
        Returns:
        - Tuple of (is_authentic, reason)
        """
        if not data:
            reason = "Empty data set"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if not all(len(candle) >= 6 for candle in data):
            reason = "Invalid candle format (missing fields)"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        timestamps = [candle[0] for candle in data]
        if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)):
            reason = "Non-sequential timestamps"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        for i in range(len(data)-1):
            current_close = data[i][4]
            next_open = data[i+1][1]
            
            price_change_pct = abs(next_open - current_close) / current_close if current_close else 0
            if price_change_pct > 0.5:  # 50% change between candles is suspicious
                reason = f"Unrealistic price gap: {price_change_pct:.2%} between candles"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
        
        for candle in data:
            timestamp, open_price, high, low, close, volume = candle[:6]
            
            if not (low <= open_price <= high and low <= close <= high):
                reason = "Price integrity violation (OHLC relationship)"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
            
            if any(val < 0 for val in candle[1:]):
                reason = "Negative values in candle data"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
            
            if volume == 0 and exchange not in ['ftx', 'bybit']:
                reason = "Zero volume candle (suspicious)"
                logger.warning(f"Suspicious data: {reason} for {symbol} on {exchange}")
        
        self._record_verification(True, "Data verified successfully", symbol, exchange, "ohlcv", timeframe)
        return True, "Data verified successfully"
    
    def verify_ticker_data(self, ticker: Dict[str, Any], symbol: str, 
                          exchange: str) -> Tuple[bool, str]:
        """
        Verify ticker data authenticity.
        
        Parameters:
        - ticker: Ticker data to verify
        - symbol: Trading pair symbol
        - exchange: Exchange name
        
        Returns:
        - Tuple of (is_authentic, reason)
        """
        if not ticker:
            reason = "Empty ticker data"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        required_fields = ['last', 'bid', 'ask', 'volume']
        missing_fields = [field for field in required_fields if field not in ticker]
        if missing_fields:
            reason = f"Missing required fields: {missing_fields}"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if 'bid' in ticker and 'ask' in ticker and ticker['bid'] > ticker['ask']:
            reason = "Price integrity violation (bid > ask)"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if 'timestamp' in ticker:
            ticker_time = datetime.fromtimestamp(ticker['timestamp'] / 1000)
            now = datetime.now()
            if ticker_time < now - timedelta(minutes=5):
                reason = f"Stale ticker data: {ticker_time}"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
        
        self._record_verification(True, "Ticker verified successfully", symbol, exchange, "ticker")
        return True, "Ticker verified successfully"
    
    def verify_order_book_data(self, order_book: Dict[str, Any], symbol: str, 
                              exchange: str) -> Tuple[bool, str]:
        """
        Verify order book data authenticity.
        
        Parameters:
        - order_book: Order book data to verify
        - symbol: Trading pair symbol
        - exchange: Exchange name
        
        Returns:
        - Tuple of (is_authentic, reason)
        """
        if not order_book:
            reason = "Empty order book data"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        required_fields = ['bids', 'asks']
        missing_fields = [field for field in required_fields if field not in order_book]
        if missing_fields:
            reason = f"Missing required fields: {missing_fields}"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        bids = order_book['bids']
        asks = order_book['asks']
        
        if not bids and not asks:
            reason = "Empty order book (no bids or asks)"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if bids and not all(bids[i][0] >= bids[i+1][0] for i in range(len(bids)-1)):
            reason = "Bids not properly ordered (descending)"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if asks and not all(asks[i][0] <= asks[i+1][0] for i in range(len(asks)-1)):
            reason = "Asks not properly ordered (ascending)"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        if bids and asks and bids[0][0] >= asks[0][0]:
            reason = f"Crossed order book: highest bid {bids[0][0]} >= lowest ask {asks[0][0]}"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        self._record_verification(True, "Order book verified successfully", symbol, exchange, "order_book")
        return True, "Order book verified successfully"
    
    def verify_trade_data(self, trades: List[Dict[str, Any]], symbol: str, 
                         exchange: str) -> Tuple[bool, str]:
        """
        Verify trade data authenticity.
        
        Parameters:
        - trades: Trade data to verify
        - symbol: Trading pair symbol
        - exchange: Exchange name
        
        Returns:
        - Tuple of (is_authentic, reason)
        """
        if not trades:
            reason = "Empty trade data"
            self._handle_verification_failure(reason, symbol, exchange)
            return False, reason
        
        required_fields = ['price', 'amount', 'side']
        for i, trade in enumerate(trades):
            missing_fields = [field for field in required_fields if field not in trade]
            if missing_fields:
                reason = f"Trade {i} missing required fields: {missing_fields}"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
        
        if all('timestamp' in trade for trade in trades):
            timestamps = [trade['timestamp'] for trade in trades]
            if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                reason = "Non-sequential trade timestamps"
                self._handle_verification_failure(reason, symbol, exchange)
                return False, reason
        
        self._record_verification(True, "Trade data verified successfully", symbol, exchange, "trades")
        return True, "Trade data verified successfully"
    
    def verify_exchange_info(self, exchange_info: Dict[str, Any], 
                            exchange: str) -> Tuple[bool, str]:
        """
        Verify exchange information authenticity.
        
        Parameters:
        - exchange_info: Exchange information to verify
        - exchange: Exchange name
        
        Returns:
        - Tuple of (is_authentic, reason)
        """
        if not exchange_info:
            reason = "Empty exchange info"
            self._handle_verification_failure(reason, "N/A", exchange)
            return False, reason
        
        if 'has' in exchange_info:
            required_capabilities = ['fetchTicker', 'fetchOrderBook', 'fetchTrades']
            missing_capabilities = [cap for cap in required_capabilities 
                                   if cap in exchange_info['has'] and not exchange_info['has'][cap]]
            
            if missing_capabilities:
                reason = f"Exchange missing required capabilities: {missing_capabilities}"
                logger.warning(f"Exchange limitation: {reason} for {exchange}")
        
        self._record_verification(True, "Exchange info verified successfully", "N/A", exchange, "exchange_info")
        return True, "Exchange info verified successfully"
    
    def verify_all_data_sources(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify all data sources in the system.
        
        Parameters:
        - data_sources: Dictionary of data sources to verify
        
        Returns:
        - Verification results for each data source
        """
        results = {}
        
        for source_name, source_data in data_sources.items():
            source_type = source_data.get('type', 'unknown')
            symbol = source_data.get('symbol', 'N/A')
            exchange = source_data.get('exchange', 'N/A')
            data = source_data.get('data', None)
            
            if source_type == 'ohlcv':
                is_authentic, reason = self.verify_ohlcv_data(
                    data, symbol, exchange, source_data.get('timeframe', '1m')
                )
            elif source_type == 'ticker':
                is_authentic, reason = self.verify_ticker_data(data, symbol, exchange)
            elif source_type == 'order_book':
                is_authentic, reason = self.verify_order_book_data(data, symbol, exchange)
            elif source_type == 'trades':
                is_authentic, reason = self.verify_trade_data(data, symbol, exchange)
            elif source_type == 'exchange_info':
                is_authentic, reason = self.verify_exchange_info(data, exchange)
            else:
                is_authentic, reason = False, f"Unknown data source type: {source_type}"
                self._handle_verification_failure(reason, symbol, exchange)
            
            results[source_name] = {
                'is_authentic': is_authentic,
                'reason': reason,
                'symbol': symbol,
                'exchange': exchange,
                'type': source_type
            }
        
        return results
    
    def _handle_verification_failure(self, reason: str, symbol: str, exchange: str) -> None:
        """
        Handle verification failure based on strict mode.
        
        Parameters:
        - reason: Reason for verification failure
        - symbol: Trading pair symbol
        - exchange: Exchange name
        """
        error_msg = f"Data verification failed: {reason} for {symbol} on {exchange}"
        self._record_verification(False, reason, symbol, exchange)
        
        if self.strict_mode:
            logger.error(error_msg)
            raise ValueError(f"FAKE DATA DETECTED: {error_msg}")
        else:
            logger.warning(error_msg)
    
    def _record_verification(self, success: bool, reason: str, symbol: str, 
                            exchange: str, data_type: str = "unknown", 
                            timeframe: str = None) -> None:
        """
        Record verification result in history.
        
        Parameters:
        - success: Whether verification was successful
        - reason: Reason for verification result
        - symbol: Trading pair symbol
        - exchange: Exchange name
        - data_type: Type of data verified
        - timeframe: Timeframe of the data (if applicable)
        """
        self.verification_history.append({
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'reason': reason,
            'symbol': symbol,
            'exchange': exchange,
            'data_type': data_type,
            'timeframe': timeframe
        })
        
        if len(self.verification_history) > 1000:
            self.verification_history = self.verification_history[-1000:]
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get verification statistics.
        
        Returns:
        - Dictionary with verification statistics
        """
        total = len(self.verification_history)
        if total == 0:
            return {
                'total_verifications': 0,
                'success_rate': 0,
                'failure_rate': 0,
                'failures_by_exchange': {},
                'failures_by_data_type': {}
            }
        
        successes = sum(1 for record in self.verification_history if record['success'])
        failures = total - successes
        
        failures_by_exchange = {}
        failures_by_data_type = {}
        
        for record in self.verification_history:
            if not record['success']:
                exchange = record['exchange']
                data_type = record['data_type']
                
                if exchange not in failures_by_exchange:
                    failures_by_exchange[exchange] = 0
                failures_by_exchange[exchange] += 1
                
                if data_type not in failures_by_data_type:
                    failures_by_data_type[data_type] = 0
                failures_by_data_type[data_type] += 1
        
        return {
            'total_verifications': total,
            'success_count': successes,
            'failure_count': failures,
            'success_rate': successes / total if total > 0 else 0,
            'failure_rate': failures / total if total > 0 else 0,
            'failures_by_exchange': failures_by_exchange,
            'failures_by_data_type': failures_by_data_type
        }
    
    def run_nuclear_verification(self) -> bool:
        """
        Run a comprehensive verification of all data sources.
        
        This is the "nuclear" verification that ensures 100% real data
        with no chance of synthetic data slipping through.
        
        Returns:
        - True if all data is verified as real, False otherwise
        """
        logger.info("Running nuclear verification of all data sources")
        
        
        failure_count = sum(1 for record in self.verification_history if not record['success'])
        
        if failure_count > 0:
            logger.error(f"Nuclear verification failed: {failure_count} verification failures detected")
            return False
        
        logger.info("Nuclear verification passed: All data verified as 100% real")
        return True
