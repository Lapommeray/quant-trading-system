import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedBacktester:
    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.results = {}
    
    def add_strategy(self, strategy_class, **kwargs):
        self.cerebro.addstrategy(strategy_class, **kwargs)
    
    def add_data(self, data_feed):
        self.cerebro.adddata(data_feed)
    
    def set_cash(self, cash=10000):
        self.cerebro.broker.setcash(cash)
    
    def add_sizer(self, sizer_class=bt.sizers.FixedSize, stake=1):
        self.cerebro.addsizer(sizer_class, stake=stake)
    
    def run_backtest(self):
        print('Starting Portfolio Value: %.2f' % self.cerebro.broker.getvalue())
        
        results = self.cerebro.run()
        
        print('Final Portfolio Value: %.2f' % self.cerebro.broker.getvalue())
        
        return results
    
    def plot_results(self):
        self.cerebro.plot()

class QuantumStrategy(bt.Strategy):
    def __init__(self):
        self.quantum_signal = None
    
    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.close[-1]:
                self.buy()
        else:
            if self.data.close[0] < self.data.close[-1]:
                self.sell()
