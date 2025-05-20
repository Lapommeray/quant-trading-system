"""
Interactive Brokers Executor

This module provides an execution interface for the Interactive Brokers API to be used with the QMP Overrider system.
It handles order execution, position management, and account information.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

try:
    from ib_insync import *
except ImportError:
    logging.error("ib_insync not installed. Please install it with 'pip install ib_insync'")

class IBExecutor:
    """
    Interactive Brokers Executor
    
    Provides an execution interface for the Interactive Brokers API to be used with the QMP Overrider system.
    It handles order execution, position management, and account information.
    """
    
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        """
        Initialize Interactive Brokers Executor
        
        Parameters:
        - host: IB Gateway/TWS host
        - port: IB Gateway/TWS port
        - client_id: Client ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self.ib = IB()
        self.logger = self._setup_logger()
        self.logger.info("Initializing Interactive Brokers Executor")
        
        self.order_history = []
        self.position_cache = {}
        self.account_cache = None
        self.last_cache_update = None
        
        self.connected = False
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("IBExecutor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect(self):
        """
        Connect to Interactive Brokers
        
        Returns:
        - True if successful, False otherwise
        """
        if self.connected:
            return True
        
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            self.logger.info(f"Connected to IB: {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to IB: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from Interactive Brokers
        """
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IB")
    
    def ensure_connected(self):
        """
        Ensure connection to Interactive Brokers
        
        Returns:
        - True if connected, False otherwise
        """
        if not self.connected:
            return self.connect()
        
        return True
    
    def get_account(self, force_refresh=False):
        """
        Get account information
        
        Parameters:
        - force_refresh: Force refresh of account cache
        
        Returns:
        - Account information
        """
        if not self.ensure_connected():
            return None
        
        if not force_refresh and self.account_cache and self.last_cache_update:
            if (datetime.now() - self.last_cache_update).total_seconds() < 60:
                return self.account_cache
        
        try:
            account_values = self.ib.accountValues()
            
            account_data = {}
            for value in account_values:
                if value.currency == 'USD':
                    account_data[value.tag] = value.value
            
            self.account_cache = account_data
            self.last_cache_update = datetime.now()
            
            return account_data
        except Exception as e:
            self.logger.error(f"Error getting account: {e}")
            return None
    
    def get_positions(self, force_refresh=False):
        """
        Get positions
        
        Parameters:
        - force_refresh: Force refresh of position cache
        
        Returns:
        - Positions
        """
        if not self.ensure_connected():
            return None
        
        if not force_refresh and self.position_cache and self.last_cache_update:
            if (datetime.now() - self.last_cache_update).total_seconds() < 60:
                return list(self.position_cache.values())
        
        try:
            positions = self.ib.positions()
            
            position_data = []
            for position in positions:
                position_data.append({
                    "symbol": position.contract.symbol,
                    "exchange": position.contract.exchange,
                    "currency": position.contract.currency,
                    "position": position.position,
                    "avg_cost": position.avgCost
                })
            
            self.position_cache = {p["symbol"]: p for p in position_data}
            self.last_cache_update = datetime.now()
            
            return position_data
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return None
    
    def get_position(self, symbol):
        """
        Get position for a specific symbol
        
        Parameters:
        - symbol: Symbol to get position for
        
        Returns:
        - Position information
        """
        positions = self.get_positions()
        
        if positions:
            for position in positions:
                if position["symbol"] == symbol:
                    return position
        
        return None
    
    def create_contract(self, symbol, sec_type="STK", exchange="SMART", currency="USD"):
        """
        Create contract
        
        Parameters:
        - symbol: Symbol
        - sec_type: Security type (STK, FUT, OPT, etc.)
        - exchange: Exchange
        - currency: Currency
        
        Returns:
        - Contract
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        return contract
    
    def place_market_order(self, symbol, qty, side):
        """
        Place market order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (BUY or SELL)
        
        Returns:
        - Order information
        """
        if not self.ensure_connected():
            return None
        
        contract = self.create_contract(symbol)
        
        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, abs(qty))
        
        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give IB time to process
            
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "market",
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "order_id": trade.order.orderId
            }
            
            self.order_history.append(order_data)
            self.logger.info(f"Placed market order: {symbol} {side} {qty}")
            
            return order_data
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol, qty, side, limit_price):
        """
        Place limit order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (BUY or SELL)
        - limit_price: Limit price
        
        Returns:
        - Order information
        """
        if not self.ensure_connected():
            return None
        
        contract = self.create_contract(symbol)
        
        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = LimitOrder(action, abs(qty), limit_price)
        
        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give IB time to process
            
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "limit",
                "limit_price": limit_price,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "order_id": trade.order.orderId
            }
            
            self.order_history.append(order_data)
            self.logger.info(f"Placed limit order: {symbol} {side} {qty} @ {limit_price}")
            
            return order_data
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_stop_order(self, symbol, qty, side, stop_price):
        """
        Place stop order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (BUY or SELL)
        - stop_price: Stop price
        
        Returns:
        - Order information
        """
        if not self.ensure_connected():
            return None
        
        contract = self.create_contract(symbol)
        
        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = StopOrder(action, abs(qty), stop_price)
        
        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give IB time to process
            
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "stop",
                "stop_price": stop_price,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "order_id": trade.order.orderId
            }
            
            self.order_history.append(order_data)
            self.logger.info(f"Placed stop order: {symbol} {side} {qty} @ {stop_price}")
            
            return order_data
        except Exception as e:
            self.logger.error(f"Error placing stop order: {e}")
            return None
    
    def get_orders(self):
        """
        Get orders
        
        Returns:
        - Orders
        """
        if not self.ensure_connected():
            return None
        
        try:
            trades = self.ib.trades()
            
            order_data = []
            for trade in trades:
                order_data.append({
                    "symbol": trade.contract.symbol,
                    "exchange": trade.contract.exchange,
                    "currency": trade.contract.currency,
                    "action": trade.order.action,
                    "qty": trade.order.totalQuantity,
                    "order_type": trade.order.orderType,
                    "limit_price": trade.order.lmtPrice,
                    "stop_price": trade.order.auxPrice,
                    "status": trade.orderStatus.status,
                    "filled": trade.orderStatus.filled,
                    "remaining": trade.orderStatus.remaining,
                    "avg_fill_price": trade.orderStatus.avgFillPrice,
                    "order_id": trade.order.orderId
                })
            
            return order_data
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel order
        
        Parameters:
        - order_id: Order ID to cancel
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.ensure_connected():
            return False
        
        try:
            trades = self.ib.trades()
            
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    self.logger.info(f"Cancelled order: {order_id}")
                    return True
            
            self.logger.warning(f"Order not found: {order_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def cancel_all_orders(self):
        """
        Cancel all orders
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.ensure_connected():
            return False
        
        try:
            self.ib.reqGlobalCancel()
            self.logger.info("Cancelled all orders")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return False
    
    def close_position(self, symbol):
        """
        Close position
        
        Parameters:
        - symbol: Symbol to close position for
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.ensure_connected():
            return False
        
        position = self.get_position(symbol)
        
        if not position:
            self.logger.warning(f"No position found for {symbol}")
            return False
        
        try:
            contract = self.create_contract(symbol)
            
            action = "SELL" if position["position"] > 0 else "BUY"
            qty = abs(position["position"])
            
            order = MarketOrder(action, qty)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Give IB time to process
            
            self.logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def close_all_positions(self):
        """
        Close all positions
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.ensure_connected():
            return False
        
        positions = self.get_positions()
        
        if not positions:
            self.logger.warning("No positions found")
            return False
        
        success = True
        
        for position in positions:
            if not self.close_position(position["symbol"]):
                success = False
        
        return success
    
    def get_market_data(self, symbol, data_type="TRADES"):
        """
        Get market data
        
        Parameters:
        - symbol: Symbol to get market data for
        - data_type: Data type (TRADES, BID_ASK, etc.)
        
        Returns:
        - Market data
        """
        if not self.ensure_connected():
            return None
        
        contract = self.create_contract(symbol)
        
        try:
            if data_type == "TRADES":
                self.ib.reqMktData(contract, "", False, False)
            elif data_type == "BID_ASK":
                self.ib.reqMktData(contract, "233", False, False)
            
            self.ib.sleep(1)  # Give IB time to process
            
            ticker = self.ib.ticker(contract)
            
            if data_type == "TRADES":
                return {
                    "symbol": symbol,
                    "last": ticker.last,
                    "last_size": ticker.lastSize,
                    "volume": ticker.volume,
                    "high": ticker.high,
                    "low": ticker.low,
                    "close": ticker.close,
                    "time": ticker.time
                }
            elif data_type == "BID_ASK":
                return {
                    "symbol": symbol,
                    "bid": ticker.bid,
                    "bid_size": ticker.bidSize,
                    "ask": ticker.ask,
                    "ask_size": ticker.askSize,
                    "time": ticker.time
                }
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
        finally:
            self.ib.cancelMktData(contract)
    
    def get_historical_data(self, symbol, duration="1 D", bar_size="1 min", what_to_show="TRADES"):
        """
        Get historical data
        
        Parameters:
        - symbol: Symbol to get historical data for
        - duration: Duration (1 D, 1 W, 1 M, etc.)
        - bar_size: Bar size (1 min, 5 mins, 1 hour, 1 day, etc.)
        - what_to_show: What to show (TRADES, BID_ASK, etc.)
        
        Returns:
        - Historical data
        """
        if not self.ensure_connected():
            return None
        
        contract = self.create_contract(symbol)
        
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True
            )
            
            df = util.df(bars)
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
