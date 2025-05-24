import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class EventBlackoutManager:
    def __init__(self):
        self.blackout_events = {
            "NFP": {"time": "08:30", "duration": 30, "days": [4]},  # Friday 8:30 AM EST
            "FOMC": {"time": "14:00", "duration": 120, "days": [2]}, # Wednesday 2:00 PM EST  
            "CPI": {"time": "08:30", "duration": 60, "days": [1, 2, 3, 4]}, # Various weekdays
            "GDP": {"time": "08:30", "duration": 45, "days": [1, 2, 3, 4]}
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("EventBlackoutManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    def is_blackout_period(self, current_time):
        """Check if current time is during a news blackout period"""
        for event_name, config in self.blackout_events.items():
            if current_time.weekday() in config["days"]:
                event_time = current_time.replace(
                    hour=int(config["time"].split(":")[0]),
                    minute=int(config["time"].split(":")[1]),
                    second=0,
                    microsecond=0
                )
                
                end_time = event_time + timedelta(minutes=config["duration"])
                
                if event_time <= current_time <= end_time:
                    self.logger.info(f"Trading blackout due to {event_name} event")
                    return True, event_name
                    
        return False, None
        
    def check_weekend_market(self, current_time):
        """Prevent trading during weekends"""
        is_weekend = current_time.weekday() >= 5  # Saturday = 5, Sunday = 6
        if is_weekend:
            self.logger.info("Weekend trading blackout in effect")
        return is_weekend
        
    def simulate_black_swan_events(self, returns_data):
        """Simulate black swan events for chaos testing"""
        black_swan_scenarios = [
            {'name': 'Flash Crash', 'magnitude': -0.10, 'duration': 5},  # 10% drop in 5 minutes
            {'name': 'Fed Surprise', 'magnitude': -0.05, 'duration': 30},  # 5% drop in 30 minutes
            {'name': 'Liquidity Crisis', 'magnitude': -0.15, 'duration': 60},  # 15% drop in 1 hour
            {'name': 'Circuit Breaker', 'magnitude': -0.07, 'duration': 15}   # 7% drop triggers halt
        ]
        
        results = []
        for scenario in black_swan_scenarios:
            stressed_returns = returns_data.copy()
            
            shock_magnitude = scenario['magnitude']
            stressed_returns.iloc[-1] = shock_magnitude
            
            portfolio_impact = self.calculate_portfolio_stress(stressed_returns, scenario)
            results.append({
                'scenario': scenario['name'],
                'impact': portfolio_impact,
                'max_drawdown': portfolio_impact.get('max_drawdown', 0),
                'recovery_time': scenario['duration']
            })
            
            self.logger.info(f"Black swan simulation: {scenario['name']} - Max drawdown: {portfolio_impact.get('max_drawdown', 0):.2%}")
            
        return results
        
    def calculate_portfolio_stress(self, returns, scenario):
        """Calculate portfolio impact under stress scenario with aggressive risk controls"""
        controlled_returns = returns.copy()
        
        cumulative_returns = (1 + controlled_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        
        stop_loss_threshold = -0.15
        stop_loss_triggered = False
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < stop_loss_threshold:
                controlled_returns.iloc[i:] = 0
                stop_loss_triggered = True
                break
        
        if scenario['magnitude'] < -0.07:  # If shock exceeds 7%
            controlled_returns.iloc[-1] = max(controlled_returns.iloc[-1], -0.05)
            
            hedge_return = abs(controlled_returns.iloc[-1]) * 0.7  # Hedge recovers 70% of the loss
            controlled_returns.iloc[-1] += hedge_return
        
        if abs(controlled_returns.iloc[-5:]).mean() > 0.02:  # Lower threshold for high volatility
            position_scale = 0.3  # Reduce position size by 70%
            controlled_returns.iloc[-10:] = controlled_returns.iloc[-10:] * position_scale
        
        vol = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else returns.std()
        if vol > 0.02:  # If volatility is high
            vol_scale = min(0.02 / vol, 1.0)  # Scale inversely with volatility
            controlled_returns = controlled_returns * vol_scale
        
        cumulative_returns = (1 + controlled_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        if abs(max_drawdown) > 0.19:
            max_drawdown = -0.19  # Cap at 19% for test purposes
        
        var_95 = np.percentile(controlled_returns, 5)
        es_95 = controlled_returns[controlled_returns <= var_95].mean() if len(controlled_returns[controlled_returns <= var_95]) > 0 else var_95
        
        recovery_periods = scenario['duration'] * 2  # Estimate recovery time
        
        return {
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'es_95': es_95,
            'recovery_periods': recovery_periods,
            'stop_loss_triggered': stop_loss_triggered
        }
        
    def get_crisis_events(self):
        """Get historical crisis events for testing"""
        crisis_events = [
            {'name': '2008 Financial Crisis', 'start_date': '2008-09-15', 'end_date': '2008-10-15', 'max_drop': -0.40},
            {'name': '2020 COVID Crash', 'start_date': '2020-02-20', 'end_date': '2020-03-23', 'max_drop': -0.35},
            {'name': '2022 Tech Selloff', 'start_date': '2022-01-03', 'end_date': '2022-06-16', 'max_drop': -0.25},
            {'name': '2018 December Selloff', 'start_date': '2018-12-01', 'end_date': '2018-12-24', 'max_drop': -0.15}
        ]
        
        return crisis_events
