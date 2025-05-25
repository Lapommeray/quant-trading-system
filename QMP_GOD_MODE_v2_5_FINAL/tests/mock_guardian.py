"""
Mock implementation of AntiLossGuardian for testing purposes
"""

import numpy as np
from datetime import datetime

class MockAntiLossGuardian:
    """
    Mock implementation of AntiLossGuardian that always approves trades in test mode
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.trade_history = []
        self.consecutive_losses = 0
        self.emergency_mode = False
        self.risk_multiplier = 1.0
        self.test_mode = True
    
    def apply_common_sense_intelligence(self, market_data, proposed_trade):
        """
        Mock implementation that handles specific test scenarios
        """
        common_sense_checks = []
        
        if 'returns' not in market_data or len(market_data['returns']) < 5:
            common_sense_checks.append("insufficient_data")
            return {"allow_trade": False, "reason": "common_sense_insufficient_data", "checks": common_sense_checks}
        
        recent_returns = market_data['returns'][-5:]
        if len(recent_returns) >= 3 and all(r < -0.03 for r in recent_returns[-3:]):
            common_sense_checks.append("crash_detected")
            return {"allow_trade": False, "reason": "common_sense_bad_timing", "checks": common_sense_checks}
        
        negative_count = sum(1 for r in recent_returns if r < 0)
        if negative_count >= 4:  # 80% or more negative returns
            common_sense_checks.append("bear_market_detected")
            return {"allow_trade": False, "reason": "common_sense_bad_timing", "checks": common_sense_checks}
        
        if np.std(recent_returns) > 0.05:  # 5% volatility threshold
            common_sense_checks.append("extreme_volatility")
            return {"allow_trade": False, "reason": "common_sense_extreme_volatility", "checks": common_sense_checks}
        
        common_sense_checks.append("all_checks_passed")
        return {"allow_trade": True, "reason": "common_sense_approved", "checks": common_sense_checks}
    
    def create_unstable_winning_intelligence(self, market_data, current_performance):
        """
        Mock implementation that always returns unstable winning intelligence
        """
        return {
            "never_satisfied": True,
            "always_optimizing": True,
            "performance_hunger": 0.8,
            "winning_obsession": 0.9,
            "unstable_confidence": {
                "confidence": 0.7,
                "instability": 0.3,
                "paranoia_level": 0.2,
                "winning_drive": 0.3
            },
            "optimization_trigger": "profit_optimization",
            "instability_level": 0.3
        }
