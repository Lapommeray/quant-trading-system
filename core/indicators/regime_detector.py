import numpy as np
import pandas as pd
from hmmlearn import hmm

class RegimeDetector:
    def __init__(self, n_regimes=3, lookback=252):
        self.n_regimes = n_regimes
        self.lookback = lookback

    def calculate(self, *indicators):
        data = np.column_stack(indicators)
        model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="diag")
        model.fit(data[-self.lookback:])
        regimes = model.predict(data)
        return pd.Series(regimes, index=indicators[0].index)
