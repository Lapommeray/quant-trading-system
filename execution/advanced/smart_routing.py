import cvxpy as cp
import numpy as np
import pandas as pd
import warnings

class InstitutionalOptimalExecution:
    """Enhanced optimal execution with volatility and liquidity modeling"""
    def __init__(self, volatility_forecast=0.2, liquidity_profile=1.0):
        self.vol = volatility_forecast
        self.liq = liquidity_profile

    def solve_institutional(self, target_shares, time_horizon):
        """Solve optimal execution with institutional cost model"""
        x = cp.Variable(time_horizon)
        cost = cp.sum(
            0.1 * cp.abs(x) +                    # Linear impact
            0.3 * cp.square(x) / self.liq +      # Quadratic impact  
            0.5 * self.vol * cp.abs(x)           # Volatility penalty
        )
        prob = cp.Problem(cp.Minimize(cost), [cp.sum(x) == target_shares])
        prob.solve(solver=cp.ECOS)
        return x.value if prob.status == cp.OPTIMAL else np.zeros(time_horizon)

class AdvancedVWAPExecution:
    """Enhanced VWAP execution with institutional features"""
    def __init__(self, historical_volumes, market_impact_model=None):
        self.volume_profile = self._calculate_profile(historical_volumes)
        self.impact_model = market_impact_model
        
    def _calculate_profile(self, volumes):
        if hasattr(volumes, 'index') and hasattr(volumes.index, 'time'):
            return volumes.groupby(volumes.index.time).mean()
        else:
            return volumes
    
    def get_institutional_schedule(self, target_quantity, start_time, end_time, risk_aversion=1.0):
        """Enhanced VWAP scheduling with risk management"""
        time_slices = pd.date_range(start_time, end_time, freq='5min').time
        
        if hasattr(self.volume_profile, 'loc'):
            available_volume = sum(self.volume_profile.loc[t] for t in time_slices 
                               if t in self.volume_profile.index)
            
            schedule = {}
            for t in time_slices:
                if t in self.volume_profile.index:
                    base_allocation = target_quantity * (self.volume_profile.loc[t]/available_volume)
                    risk_adjusted = base_allocation * (1 - risk_aversion * 0.1)
                    schedule[t] = max(0, risk_adjusted)
            return schedule
        else:
            equal_allocation = target_quantity / len(time_slices)
            return {t: equal_allocation for t in time_slices}
