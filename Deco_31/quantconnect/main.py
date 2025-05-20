
from AlgorithmImports import *

class QMPGodMode(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        self.AddEquity("SPY", Resolution.Minute)
        self.SetBenchmark("SPY")

    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 1)
