from sortedcontainers import SortedDict
import numpy as np

class LimitOrderBook:
    def __init__(self):
        self.bids = SortedDict()  # price -> volume
        self.asks = SortedDict()
        self.trade_history = []
    
    def add_order(self, price, volume, is_bid):
        book = self.bids if is_bid else self.asks
        book[price] = book.get(price, 0) + volume
    
    def cancel_order(self, price, volume, is_bid):
        book = self.bids if is_bid else self.asks
        if price in book:
            book[price] = max(0, book[price] - volume)
            if book[price] == 0:
                del book[price]
    
    def match_orders(self):
        trades = []
        while self.bids and self.asks and self.bids.peekitem(-1)[0] >= self.asks.peekitem(0)[0]:
            best_bid_price, best_bid_vol = self.bids.peekitem(-1)
            best_ask_price, best_ask_vol = self.asks.peekitem(0)
            
            trade_volume = min(best_bid_vol, best_ask_vol)
            trade_price = (best_bid_price + best_ask_price) / 2
            
            trades.append((trade_price, trade_volume))
            
            if best_bid_vol == trade_volume:
                del self.bids[best_bid_price]
            else:
                self.bids[best_bid_price] -= trade_volume
                
            if best_ask_vol == trade_volume:
                del self.asks[best_ask_price]
            else:
                self.asks[best_ask_price] -= trade_volume
        
        self.trade_history.extend(trades)
        return trades
