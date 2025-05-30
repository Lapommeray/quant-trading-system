import pandas as pd

class OrderFlowImbalance:
    def __init__(self, window=100):
        self.window = window

    def calculate(self, trades):
        if not isinstance(trades, pd.DataFrame):
            raise ValueError("Requires tick data DataFrame")
        trades['dollar_volume'] = trades['price'] * trades['quantity']
        buys = trades[trades['side'] == 1]
        sells = trades[trades['side'] == -1]
        buy_vol = buys['dollar_volume'].rolling(self.window).sum()
        sell_vol = sells['dollar_volume'].rolling(self.window).sum()
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        return imbalance
