import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class ML_RSI:
    def __init__(self, window=14, lookahead=5):
        self.window = window
        self.lookahead = lookahead
        self.model = GradientBoostingRegressor(n_estimators=100)

    def calculate(self, prices, rsi_values):
        X = []
        y = []
        for i in range(self.window, len(prices)-self.lookahead):
            features = [
                rsi_values[i],
                prices[i]/prices[i-self.window] - 1,
                (prices[i] - prices[i-self.window:i].min()) / 
                (prices[i-self.window:i].max() - prices[i-self.window:i].min()),
                (rsi_values[i] - rsi_values[i-self.window:i].min()) / 
                (rsi_values[i-self.window:i].max() - rsi_values[i-self.window:i].min())
            ]
            X.append(features)
            y.append(prices[i+self.lookahead]/prices[i] - 1)
        self.model.fit(X[:-self.lookahead], y[:-self.lookahead])
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=prices.index[self.window:-self.lookahead])
