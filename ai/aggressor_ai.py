import numpy as np
from stable_baselines3 import PPO

class AggressorAI:
    def __init__(self):
        self.model = PPO.load("liquidity_hunter_v3")
    
    def execute(self, market_state):
        action, _ = self.model.predict(market_state)
        return {
            'action': 'aggressive' if action > 0.7 else 'stealth',
            'q_value': float(action)
        }
