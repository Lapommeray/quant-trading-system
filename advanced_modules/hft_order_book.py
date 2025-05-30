import numpy as np
import pandas as pd
import logging
import warnings
from sortedcontainers import SortedDict
from datetime import datetime

def _warn_about_enhanced_features():
    warnings.warn(
        "Consider using EnhancedLimitOrderBook for institutional features like trade imbalance tracking and VPIN calculation. "
        "Basic LimitOrderBook will continue to be supported.",
        FutureWarning,
        stacklevel=3
    )

class LimitOrderBook:
    """
    High-frequency trading limit order book implementation
    """
    def __init__(self):
        self.bids = SortedDict()  # price -> volume
        self.asks = SortedDict()  # price -> volume
        self.bid_orders = {}  # order_id -> (price, volume)
        self.ask_orders = {}  # order_id -> (price, volume)
        self.trades = []
        self.next_order_id = 1
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def add_order(self, price, volume, is_bid):
        """
        Add a limit order to the book
        """
        order_id = self.next_order_id
        self.next_order_id += 1
        
        if is_bid:
            self.bids[price] = self.bids.get(price, 0) + volume
            self.bid_orders[order_id] = (price, volume)
        else:
            self.asks[price] = self.asks.get(price, 0) + volume
            self.ask_orders[order_id] = (price, volume)
            
        return order_id
        
    def cancel_order(self, order_id):
        """
        Cancel an existing order
        """
        if order_id in self.bid_orders:
            price, volume = self.bid_orders[order_id]
            self.bids[price] -= volume
            if self.bids[price] <= 0:
                del self.bids[price]
            del self.bid_orders[order_id]
            return True
        elif order_id in self.ask_orders:
            price, volume = self.ask_orders[order_id]
            self.asks[price] -= volume
            if self.asks[price] <= 0:
                del self.asks[price]
            del self.ask_orders[order_id]
            return True
        else:
            return False
            
    def match_orders(self):
        """
        Match orders and execute trades
        """
        trades = []
        
        while self.bids and self.asks:
            best_bid_price = self.bids.keys()[-1]
            best_ask_price = self.asks.keys()[0]
            
            if best_bid_price >= best_ask_price:
                best_bid_volume = self.bids[best_bid_price]
                best_ask_volume = self.asks[best_ask_price]
                
                trade_volume = min(best_bid_volume, best_ask_volume)
                
                trade = {
                    'timestamp': datetime.now(),
                    'price': (best_bid_price + best_ask_price) / 2,  # Mid-price
                    'volume': trade_volume
                }
                trades.append(trade)
                self.trades.append(trade)
                
                self.bids[best_bid_price] -= trade_volume
                self.asks[best_ask_price] -= trade_volume
                
                if self.bids[best_bid_price] <= 0:
                    del self.bids[best_bid_price]
                if self.asks[best_ask_price] <= 0:
                    del self.asks[best_ask_price]
                    
                for order_id, (price, volume) in list(self.bid_orders.items()):
                    if price == best_bid_price:
                        remaining_volume = max(0, volume - trade_volume)
                        if remaining_volume > 0:
                            self.bid_orders[order_id] = (price, remaining_volume)
                        else:
                            del self.bid_orders[order_id]
                        break
                        
                for order_id, (price, volume) in list(self.ask_orders.items()):
                    if price == best_ask_price:
                        remaining_volume = max(0, volume - trade_volume)
                        if remaining_volume > 0:
                            self.ask_orders[order_id] = (price, remaining_volume)
                        else:
                            del self.ask_orders[order_id]
                        break
            else:
                break
                
        return trades
        
    def get_order_book_snapshot(self):
        """
        Get a snapshot of the current order book
        """
        bid_levels = [(price, volume) for price, volume in self.bids.items()]
        ask_levels = [(price, volume) for price, volume in self.asks.items()]
        
        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'timestamp': datetime.now()
        }
        
    def get_mid_price(self):
        """
        Get the mid-price
        """
        if not self.bids or not self.asks:
            return None
            
        best_bid = self.bids.keys()[-1] if self.bids else 0
        best_ask = self.asks.keys()[0] if self.asks else 0
        
        if best_bid == 0 or best_ask == 0:
            return None
            
        return (best_bid + best_ask) / 2
        
    def get_spread(self):
        """
        Get the bid-ask spread
        """
        if not self.bids or not self.asks:
            return None
            
        best_bid = self.bids.keys()[-1] if self.bids else 0
        best_ask = self.asks.keys()[0] if self.asks else 0
        
        if best_bid == 0 or best_ask == 0:
            return None
            
        return best_ask - best_bid
        
    def get_volume_at_price(self, price, is_bid):
        """
        Get the volume at a specific price level
        """
        if is_bid:
            return self.bids.get(price, 0)
        else:
            return self.asks.get(price, 0)
            
    def get_order_imbalance(self):
        """
        Calculate order imbalance
        """
        if not self.bids or not self.asks:
            return 0
            
        best_bid = self.bids.keys()[-1] if self.bids else 0
        best_ask = self.asks.keys()[0] if self.asks else 0
        
        if best_bid == 0 or best_ask == 0:
            return 0
            
        bid_volume = self.bids[best_bid]
        ask_volume = self.asks[best_ask]
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume
        
    def simulate_market_order(self, volume, is_buy):
        """
        Simulate a market order and calculate the execution price
        """
        if is_buy:
            book = self.asks
            is_empty = not self.asks
        else:
            book = self.bids
            is_empty = not self.bids
            
        if is_empty:
            return None
            
        remaining_volume = volume
        total_cost = 0
        
        for price, level_volume in sorted(book.items()):
            if remaining_volume <= 0:
                break
                
            executed_volume = min(remaining_volume, level_volume)
            total_cost += executed_volume * price
            remaining_volume -= executed_volume
            
        if remaining_volume > 0:
            return None
            
        average_price = total_cost / volume
        return average_price
        
    def simulate_limit_order(self, price, volume, is_buy):
        """
        Simulate a limit order and calculate the execution
        """
        if is_buy:
            if not self.asks:
                return 0, price  # No execution, order added to book
                
            executed_volume = 0
            total_cost = 0
            
            for ask_price, ask_volume in sorted(self.asks.items()):
                if ask_price > price or executed_volume >= volume:
                    break
                    
                executable_volume = min(volume - executed_volume, ask_volume)
                executed_volume += executable_volume
                total_cost += executable_volume * ask_price
                
            if executed_volume == 0:
                return 0, price  # No execution, order added to book
                
            average_price = total_cost / executed_volume if executed_volume > 0 else price
            return executed_volume, average_price
        else:
            if not self.bids:
                return 0, price  # No execution, order added to book
                
            executed_volume = 0
            total_revenue = 0
            
            for bid_price, bid_volume in sorted(self.bids.items(), reverse=True):
                if bid_price < price or executed_volume >= volume:
                    break
                    
                executable_volume = min(volume - executed_volume, bid_volume)
                executed_volume += executable_volume
                total_revenue += executable_volume * bid_price
                
            if executed_volume == 0:
                return 0, price  # No execution, order added to book
                
            average_price = total_revenue / executed_volume if executed_volume > 0 else price
            return executed_volume, average_price
            
    def clear(self):
        """
        Clear the order book
        """
        self.bids.clear()
        self.asks.clear()
        self.bid_orders.clear()
        self.ask_orders.clear()


class EnhancedLimitOrderBook(LimitOrderBook):
    """
    Enhanced Limit Order Book with institutional features.
    Extends the existing LimitOrderBook with trade imbalance tracking and VPIN calculation.
    """
    
    def __init__(self, tick_size=0.01, enable_institutional_features=False):
        super().__init__()
        self.tick_size = tick_size
        self.enable_institutional_features = enable_institutional_features
        
        if enable_institutional_features:
            self.trade_imbalance = 0
            self.price_impact = []
            self.mid_price_history = []
            self.spread_history = []
            self.buy_volumes = []
            self.sell_volumes = []
    
    def add_order(self, price, volume, is_bid):
        """
        Add a limit order to the book with enhanced tracking
        """
        _warn_about_enhanced_features()
        order_id = super().add_order(price, volume, is_bid)
        
        if self.enable_institutional_features:
            if is_bid:
                self.trade_imbalance += volume
            else:
                self.trade_imbalance -= volume
                
        return order_id
        
    def process_trade(self, price, volume, side):
        """
        Process a trade with institutional metrics
        
        Parameters:
        - price: Trade price
        - volume: Trade volume
        - side: 'buy' or 'sell'
        
        Returns:
        - Trade impact metrics
        """
        if not self.enable_institutional_features:
            return None
            
        # Update trade imbalance
        if side == 'buy':
            self.trade_imbalance += volume
            self.buy_volumes.append(volume)
        else:
            self.trade_imbalance -= volume
            self.sell_volumes.append(volume)
        
        # Calculate price impact
        impact = None
        if self.bids and self.asks:
            current_mid = (self.bids.keys()[-1] + self.asks.keys()[0]) / 2
            if hasattr(self, 'last_mid') and self.last_mid:
                impact = (current_mid - self.last_mid) / self.last_mid
                self.price_impact.append(impact)
            self.last_mid = current_mid
            
            self.mid_price_history.append(current_mid)
            self.spread_history.append(self.asks.keys()[0] - self.bids.keys()[-1])
        
        trade = {
            'timestamp': datetime.now(),
            'price': price,
            'volume': volume,
            'side': side,
            'impact': impact
        }
        self.trades.append(trade)
        
        return impact
    
    def calculate_vpin(self, bucket_volume=10000, window=50):
        """
        Calculate Volume-synchronized Probability of Informed Trading
        
        Parameters:
        - bucket_volume: Volume per bucket
        - window: Number of buckets to consider
        
        Returns:
        - VPIN value between 0 (normal) and 1 (toxic flow)
        """
        if not self.enable_institutional_features or len(self.buy_volumes) < window or len(self.sell_volumes) < window:
            return np.nan
            
        buy_vol = np.sum(self.buy_volumes[-window:])
        sell_vol = np.sum(self.sell_volumes[-window:])
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            return np.nan
            
        return np.abs(buy_vol - sell_vol) / total_vol
    
    def get_order_book_snapshot(self):
        """
        Get an enhanced snapshot of the current order book
        """
        snapshot = super().get_order_book_snapshot()
        
        if self.enable_institutional_features:
            snapshot.update({
                'trade_imbalance': self.trade_imbalance,
                'vpin': self.calculate_vpin(),
                'price_impact': np.mean(self.price_impact[-10:]) if len(self.price_impact) >= 10 else None
            })
            
        return snapshot
