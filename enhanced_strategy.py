from market_microstructure import LimitOrderBook, VPINCalculator
from statistical_arbitrage import AdvancedCointegration
from execution import VWAPExecution, OptimalExecution
from execution.advanced.smart_routing import InstitutionalOptimalExecution, AdvancedVWAPExecution
import pandas as pd
import numpy as np

class Strategy:
    """Base Strategy class for backtesting framework"""
    def __init__(self):
        self.data = None
        self.position = Position()
        self._indicators = {}
        
    def I(self, indicator_func):
        """Indicator wrapper function"""
        if callable(indicator_func):
            result = indicator_func
        else:
            result = indicator_func
        return result
        
    def buy(self, size=1):
        """Execute buy order"""
        self.position.size += size
        
    def sell(self, size=1):
        """Execute sell order"""
        self.position.size -= size
        
    def next(self):
        """Override this method in your strategy"""
        pass

class Position:
    """Position tracking"""
    def __init__(self):
        self.size = 0

class EnhancedStrategy(Strategy):
    def __init__(self, use_institutional=False):
        super().__init__()
        self.use_institutional = use_institutional
        
    def init(self):
        self.lob = LimitOrderBook()
        self.vpin = VPINCalculator()
        self.coint = AdvancedCointegration()
        
        if self.use_institutional:
            self.institutional_execution = InstitutionalOptimalExecution()
            self.advanced_vwap = AdvancedVWAPExecution
        
        self.rsi = self.I(self._calculate_rsi)
        self.order_flow = self.I(self._calculate_order_flow)
    
    def _calculate_rsi(self, period=14):
        """Calculate RSI indicator"""
        if not hasattr(self.data, 'Close'):
            return pd.Series(50, index=range(len(self.data)))
        
        delta = self.data.Close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_order_flow(self):
        if not hasattr(self.data, 'Trades'):
            return pd.Series(np.zeros(len(self.data.Close)), 
                           index=self.data.Close.index)
        
        ofi = []
        for trade in self.data.Trades.itertuples():
            self.vpin.add_trade(trade.price, trade.quantity, trade.side)
            ofi.append(self.vpin.calculate())
        return pd.Series(ofi, index=self.data.Trades.index)
    
    def next(self):
        if len(self.data) > self.coint.lookback:
            hedge_ratio = self.coint.hedge_ratio_estimation(
                self.data[['Close', 'Volume']].iloc[-self.coint.lookback:])
            
            if self.use_institutional and len(self.data.columns) > 2:
                institutional_hedge = self.coint.johansen_test(
                    self.data[['Close', 'Volume']].iloc[-self.coint.lookback:])
            
        if self.position.size == 0 and self.vpin.calculate() > 0.7:
            if self.use_institutional:
                optimal_schedule = self.institutional_execution.solve_institutional(
                    1000, 24)  # 24 time periods
            else:
                schedule = VWAPExecution(self.data.Volume).get_schedule(
                    1000, self.data.index[-1], self.data.index[-1] + pd.Timedelta('1h'))
