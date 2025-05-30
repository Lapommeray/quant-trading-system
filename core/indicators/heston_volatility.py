import numpy as np
import pandas as pd
from scipy.optimize import minimize

class HestonVolatility:
    def __init__(self, lookback=30, risk_free=0.01):
        self.lookback = lookback
        self.r = risk_free

    def heston_objective(self, params, returns):
        kappa, theta, xi, rho, v0 = params
        n = len(returns)
        v = np.zeros(n)
        v[0] = v0
        ll = 0

        for t in range(1, n):
            v[t] = np.abs(v[t-1] + kappa*(theta - v[t-1])/252 + xi*np.sqrt(v[t-1]/252)*returns[t-1])
            ll += -0.5*(np.log(2*np.pi) + np.log(v[t]/252) + returns[t]**2/(v[t]/252))
        return -ll

    def calculate(self, close_prices):
        returns = np.log(close_prices/close_prices.shift(1)).dropna()
        init_params = [3.0, 0.04, 0.1, -0.7, 0.04]
        bounds = ((0.1, 10), (0.001, 0.5), (0.01, 0.5), (-0.99, 0.99), (0.001, 0.5))
        res = minimize(self.heston_objective, init_params, args=(returns[-self.lookback:]),
                      bounds=bounds, method='L-BFGS-B')
        kappa, theta, xi, rho, v0 = res.x
        return pd.Series(np.sqrt(v0)*np.sqrt(252), index=close_prices.index[-len(close_prices):])
