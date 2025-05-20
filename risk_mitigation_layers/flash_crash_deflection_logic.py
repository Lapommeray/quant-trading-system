"""
Flash Crash Deflection Logic

Auto-places buy limits at -10% gaps for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class FlashCrashDeflectionLogic:
    """
    Auto-places buy limits at -10% gaps.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Flash Crash Deflection Logic.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("FlashCrashDeflectionLogic")
        self.logger.setLevel(logging.INFO)
        
        self.minor_crash_threshold = -0.05  # 5% drop
        self.major_crash_threshold = -0.10  # 10% drop
        self.severe_crash_threshold = -0.15  # 15% drop
        
        self.ultra_fast_window = 5
        self.fast_window = 15
        self.medium_window = 60
        
        self.limit_order_levels = {
            "minor": [0.03, 0.05, 0.07],  # 3%, 5%, 7% down from pre-crash
            "major": [0.07, 0.10, 0.13],  # 7%, 10%, 13% down from pre-crash
            "severe": [0.10, 0.15, 0.20]  # 10%, 15%, 20% down from pre-crash
        }
        
        self.limit_order_sizes = {
            "minor": [0.5, 0.3, 0.2],  # 50%, 30%, 20% of normal size
            "major": [0.4, 0.4, 0.2],  # 40%, 40%, 20% of normal size
            "severe": [0.3, 0.4, 0.3]  # 30%, 40%, 30% of normal size
        }
        
        self.market_data = {}
        
        self.active_orders = {}
        
        self.crash_detection = {
            "crash_detected": False,
            "crash_type": None,
            "crash_time": None,
            "pre_crash_prices": {},
            "crash_magnitude": {},
            "recovery_prices": {}
        }
        
        self.crash_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(minutes=1)  # More frequent updates for crash detection
        
    def update(self, current_time, market_data=None):
        """
        Update the flash crash deflection logic with latest data.
        
        Parameters:
        - current_time: Current datetime
        - market_data: Market data (optional)
        
        Returns:
        - Dictionary containing deflection results
        """
        if current_time - self.last_update < self.update_frequency and market_data is None:
            return {
                "crash_detection": self.crash_detection,
                "active_orders": self.active_orders
            }
            
        if market_data is not None:
            self._update_market_data(market_data)
        else:
            self._update_market_data_internal()
        
        self._detect_flash_crash(current_time)
        
        self._manage_deflection_orders(current_time)
        
        self.last_update = current_time
        
        return {
            "crash_detection": self.crash_detection,
            "active_orders": self.active_orders
        }
        
    def _update_market_data(self, market_data):
        """
        Update market data.
        
        Parameters:
        - market_data: Market data
        """
        self.market_data = market_data
        
    def _update_market_data_internal(self):
        """
        Update market data internally.
        """
        
        self.market_data = {
            "SPY": {
                "current_price": 420.0,
                "price_5m_ago": 422.0,
                "price_15m_ago": 423.0,
                "price_60m_ago": 425.0,
                "daily_open": 424.0,
                "previous_close": 423.5,
                "volume": 5000000,
                "average_volume": 3000000
            },
            "QQQ": {
                "current_price": 330.0,
                "price_5m_ago": 332.0,
                "price_15m_ago": 333.0,
                "price_60m_ago": 335.0,
                "daily_open": 334.0,
                "previous_close": 333.5,
                "volume": 4000000,
                "average_volume": 2500000
            },
            "BTCUSD": {
                "current_price": 38000.0,
                "price_5m_ago": 38200.0,
                "price_15m_ago": 38300.0,
                "price_60m_ago": 38500.0,
                "daily_open": 38400.0,
                "previous_close": 38350.0,
                "volume": 2000,
                "average_volume": 1500
            }
        }
        
    def _detect_flash_crash(self, current_time):
        """
        Detect flash crash.
        
        Parameters:
        - current_time: Current datetime
        """
        if self.crash_detection["crash_detected"]:
            self._check_crash_recovery(current_time)
            return
        
        crash_symbols = {}
        
        for symbol, data in self.market_data.items():
            change_5m = (data["current_price"] / data["price_5m_ago"]) - 1.0 if data["price_5m_ago"] > 0 else 0.0
            change_15m = (data["current_price"] / data["price_15m_ago"]) - 1.0 if data["price_15m_ago"] > 0 else 0.0
            change_60m = (data["current_price"] / data["price_60m_ago"]) - 1.0 if data["price_60m_ago"] > 0 else 0.0
            
            if change_5m <= self.severe_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "severe",
                    "window": self.ultra_fast_window,
                    "magnitude": change_5m,
                    "reference_price": data["price_5m_ago"]
                }
            elif change_5m <= self.major_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "major",
                    "window": self.ultra_fast_window,
                    "magnitude": change_5m,
                    "reference_price": data["price_5m_ago"]
                }
            elif change_5m <= self.minor_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "minor",
                    "window": self.ultra_fast_window,
                    "magnitude": change_5m,
                    "reference_price": data["price_5m_ago"]
                }
            
            elif change_15m <= self.severe_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "severe",
                    "window": self.fast_window,
                    "magnitude": change_15m,
                    "reference_price": data["price_15m_ago"]
                }
            elif change_15m <= self.major_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "major",
                    "window": self.fast_window,
                    "magnitude": change_15m,
                    "reference_price": data["price_15m_ago"]
                }
            elif change_15m <= self.minor_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "minor",
                    "window": self.fast_window,
                    "magnitude": change_15m,
                    "reference_price": data["price_15m_ago"]
                }
            
            elif change_60m <= self.severe_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "severe",
                    "window": self.medium_window,
                    "magnitude": change_60m,
                    "reference_price": data["price_60m_ago"]
                }
            elif change_60m <= self.major_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "major",
                    "window": self.medium_window,
                    "magnitude": change_60m,
                    "reference_price": data["price_60m_ago"]
                }
            elif change_60m <= self.minor_crash_threshold:
                crash_symbols[symbol] = {
                    "type": "minor",
                    "window": self.medium_window,
                    "magnitude": change_60m,
                    "reference_price": data["price_60m_ago"]
                }
        
        if len(crash_symbols) >= 2:  # At least 2 symbols crashing
            crash_type = "minor"
            for symbol, crash in crash_symbols.items():
                if crash["type"] == "severe":
                    crash_type = "severe"
                    break
                elif crash["type"] == "major" and crash_type != "severe":
                    crash_type = "major"
            
            self.crash_detection["crash_detected"] = True
            self.crash_detection["crash_type"] = crash_type
            self.crash_detection["crash_time"] = current_time
            self.crash_detection["pre_crash_prices"] = {symbol: crash["reference_price"] for symbol, crash in crash_symbols.items()}
            self.crash_detection["crash_magnitude"] = {symbol: crash["magnitude"] for symbol, crash in crash_symbols.items()}
            self.crash_detection["recovery_prices"] = {}
            
            self.logger.warning(f"Flash crash detected: {crash_type} crash in {len(crash_symbols)} symbols")
            
            self._place_deflection_orders(crash_symbols)
            
            self.crash_history.append({
                "date": current_time.strftime("%Y-%m-%d"),
                "time": current_time.strftime("%H:%M:%S"),
                "type": crash_type,
                "symbols": list(crash_symbols.keys()),
                "magnitudes": self.crash_detection["crash_magnitude"]
            })
        
    def _check_crash_recovery(self, current_time):
        """
        Check if crash has ended.
        
        Parameters:
        - current_time: Current datetime
        """
        crash_duration = (current_time - self.crash_detection["crash_time"]).total_seconds() / 60.0
        
        min_duration = 15.0
        
        if crash_duration < min_duration:
            return
        
        recovery_detected = True
        
        for symbol, pre_crash_price in self.crash_detection["pre_crash_prices"].items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol]["current_price"]
                
                recovery_pct = (current_price / pre_crash_price) - 1.0
                
                self.crash_detection["recovery_prices"][symbol] = current_price
                
                if recovery_pct < self.crash_detection["crash_magnitude"][symbol]:
                    recovery_detected = False
                    break
        
        if recovery_detected:
            self.logger.info("Flash crash recovery detected")
            
            self.crash_detection["crash_detected"] = False
            self.crash_detection["crash_type"] = None
            self.crash_detection["crash_time"] = None
            self.crash_detection["pre_crash_prices"] = {}
            self.crash_detection["crash_magnitude"] = {}
            self.crash_detection["recovery_prices"] = {}
            
            self._cancel_deflection_orders()
        
    def _place_deflection_orders(self, crash_symbols):
        """
        Place deflection orders.
        
        Parameters:
        - crash_symbols: Dictionary of symbols in crash
        """
        crash_type = self.crash_detection["crash_type"]
        
        levels = self.limit_order_levels[crash_type]
        sizes = self.limit_order_sizes[crash_type]
        
        for symbol, crash in crash_symbols.items():
            reference_price = crash["reference_price"]
            
            for i, level in enumerate(levels):
                limit_price = reference_price * (1.0 - level)
                
                order_size = sizes[i]
                
                order_id = f"{symbol}_L{i+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                self.active_orders[order_id] = {
                    "symbol": symbol,
                    "type": "LIMIT",
                    "side": "BUY",
                    "price": limit_price,
                    "size": order_size,
                    "level": i + 1,
                    "status": "ACTIVE",
                    "placed_time": datetime.now()
                }
                
                self.logger.info(f"Placed deflection order: {symbol} BUY LIMIT {order_size} @ {limit_price} (Level {i+1})")
        
    def _cancel_deflection_orders(self):
        """
        Cancel all deflection orders.
        """
        for order_id, order in self.active_orders.items():
            if order["status"] == "ACTIVE":
                order["status"] = "CANCELLED"
                
                self.logger.info(f"Cancelled deflection order: {order['symbol']} {order['side']} {order['type']} {order['size']} @ {order['price']}")
        
        self.active_orders = {order_id: order for order_id, order in self.active_orders.items() if order["status"] != "CANCELLED"}
        
    def _manage_deflection_orders(self, current_time):
        """
        Manage deflection orders.
        
        Parameters:
        - current_time: Current datetime
        """
        if not self.crash_detection["crash_detected"]:
            return
        
        filled_orders = []
        
        for order_id, order in self.active_orders.items():
            if order["status"] != "ACTIVE":
                continue
                
            symbol = order["symbol"]
            limit_price = order["price"]
            
            if symbol in self.market_data:
                current_price = self.market_data[symbol]["current_price"]
                
                if current_price <= limit_price:
                    order["status"] = "FILLED"
                    order["filled_time"] = current_time
                    order["filled_price"] = current_price
                    
                    filled_orders.append(order_id)
                    
                    self.logger.info(f"Deflection order filled: {symbol} {order['side']} {order['type']} {order['size']} @ {current_price}")
        
        for order_id in filled_orders:
            order = self.active_orders[order_id]
            
            for other_id, other_order in self.active_orders.items():
                if (other_id != order_id and 
                    other_order["symbol"] == order["symbol"] and 
                    other_order["level"] > order["level"] and
                    other_order["status"] == "ACTIVE"):
                    other_order["status"] = "CANCELLED"
                    
                    self.logger.info(f"Cancelled deflection order: {other_order['symbol']} {other_order['side']} {other_order['type']} {other_order['size']} @ {other_order['price']}")
        
    def is_crash_detected(self):
        """
        Check if crash is detected.
        
        Returns:
        - Boolean indicating if crash is detected
        """
        return self.crash_detection["crash_detected"]
        
    def get_crash_detection(self):
        """
        Get crash detection status.
        
        Returns:
        - Crash detection status
        """
        return self.crash_detection
        
    def get_active_orders(self):
        """
        Get active deflection orders.
        
        Returns:
        - Active deflection orders
        """
        return {order_id: order for order_id, order in self.active_orders.items() if order["status"] == "ACTIVE"}
        
    def get_crash_history(self):
        """
        Get crash history.
        
        Returns:
        - Crash history
        """
        return self.crash_history
