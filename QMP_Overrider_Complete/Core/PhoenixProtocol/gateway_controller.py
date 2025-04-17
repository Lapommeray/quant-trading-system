"""
gateway_controller.py

Phoenix Network Gateway Controller

Provides market collapse detection, regime classification, and anti-failure decision making.
"""

import numpy as np
from datetime import datetime
import random

class PhoenixNetwork:
    """
    Phoenix Network for QMP Overrider
    
    Provides market collapse detection, regime classification, and anti-failure decision making.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Phoenix Network
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.collapse_memory = {}
        self.regime_state = "NORMAL"
        self.last_signal = None
        self.last_check_time = None
        self.nano_nodes = {}
        self._initialize_nano_nodes()
    
    def _initialize_nano_nodes(self):
        """Initialize nano nodes"""
        try:
            from .nano_nodes.collapse_memory import CollapseMemory
            self.nano_nodes['collapse_memory'] = CollapseMemory()
        except ImportError:
            print("Warning: CollapseMemory module not found.")
        
        try:
            from .nano_nodes.regime_sentience import RegimeSentience
            self.nano_nodes['regime_sentience'] = RegimeSentience()
        except ImportError:
            print("Warning: RegimeSentience module not found.")
        
        try:
            from .nano_nodes.anti_failure_engine import AntiFailureEngine
            self.nano_nodes['anti_failure_engine'] = AntiFailureEngine()
        except ImportError:
            print("Warning: AntiFailureEngine module not found.")
    
    def get_signal(self, market_state):
        """
        Get Phoenix signal based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Dictionary with Phoenix signal information
        """
        now = datetime.now()
        self.last_check_time = now
        
        signal = {
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "regime": "NORMAL",
            "collapse_risk": 0.0,
            "timestamp": now
        }
        
        if not market_state:
            return signal
        
        collapse_risk = self._get_collapse_risk(market_state)
        
        regime = self._get_regime_state(market_state)
        
        action = self._get_anti_failure_action(market_state, collapse_risk, regime)
        
        signal["direction"] = action["direction"]
        signal["confidence"] = action["confidence"]
        signal["regime"] = regime
        signal["collapse_risk"] = collapse_risk
        
        self.last_signal = signal
        
        if self.algorithm:
            self.algorithm.Debug(f"Phoenix Network: {signal['direction']} | Confidence: {signal['confidence']}")
            self.algorithm.Debug(f"Regime: {regime} | Collapse Risk: {collapse_risk:.2f}")
        
        return signal
    
    def _get_collapse_risk(self, market_state):
        """
        Get collapse risk based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Collapse risk (0.0 to 1.0)
        """
        if 'collapse_memory' in self.nano_nodes:
            return self.nano_nodes['collapse_memory'].get_collapse_risk(market_state)
        
        collapse_risk = 0.0
        
        if "leverage_ratio" in market_state:
            leverage = market_state["leverage_ratio"]
            if leverage > 2.5:
                collapse_risk += 0.2
            if leverage > 3.0:
                collapse_risk += 0.3
        
        if "volatility" in market_state:
            volatility = market_state["volatility"]
            if volatility > 25:
                collapse_risk += 0.2
            if volatility > 35:
                collapse_risk += 0.3
        
        if "vix_term_structure" in market_state:
            vix_term = market_state["vix_term_structure"]
            if vix_term < 0.9:
                collapse_risk += 0.2
            if vix_term < 0.8:
                collapse_risk += 0.3
        
        if "etf_flow_velocity" in market_state:
            flow = market_state["etf_flow_velocity"]
            if flow < -0.5:
                collapse_risk += 0.2
            if flow < -1.0:
                collapse_risk += 0.3
        
        return min(1.0, collapse_risk)
    
    def _get_regime_state(self, market_state):
        """
        Get regime state based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Regime state (string)
        """
        if 'regime_sentience' in self.nano_nodes:
            return self.nano_nodes['regime_sentience'].get_regime_state(market_state)
        
        if "regime" in market_state:
            return market_state["regime"]
        
        collapse_risk = self._get_collapse_risk(market_state)
        
        if collapse_risk > 0.8:
            regime = "CRISIS"
        elif collapse_risk > 0.6:
            regime = "VOLATILE"
        elif collapse_risk > 0.4:
            regime = "BEARISH"
        elif collapse_risk > 0.2:
            regime = "CAUTIOUS"
        else:
            regime = "NORMAL"
        
        if "trend" in market_state:
            trend = market_state["trend"]
            if trend > 0.5 and regime == "NORMAL":
                regime = "BULLISH"
            elif trend > 0.8 and regime == "NORMAL":
                regime = "EUPHORIC"
        
        self.regime_state = regime
        return regime
    
    def _get_anti_failure_action(self, market_state, collapse_risk, regime):
        """
        Get anti-failure action based on market state, collapse risk, and regime
        
        Parameters:
        - market_state: Dictionary with market state information
        - collapse_risk: Collapse risk (0.0 to 1.0)
        - regime: Regime state (string)
        
        Returns:
        - Dictionary with anti-failure action information
        """
        if 'anti_failure_engine' in self.nano_nodes:
            return self.nano_nodes['anti_failure_engine'].get_anti_failure_action(
                market_state, collapse_risk, regime
            )
        
        action = {
            "direction": "NEUTRAL",
            "confidence": 0.5,
            "position_multiplier": 1.0
        }
        
        if regime == "CRISIS":
            action["direction"] = "SELL"
            action["confidence"] = 0.9
            action["position_multiplier"] = 0.0
        
        elif regime == "VOLATILE":
            action["direction"] = "SELL"
            action["confidence"] = 0.7
            action["position_multiplier"] = 0.5
        
        elif regime == "BEARISH":
            action["direction"] = "SELL"
            action["confidence"] = 0.6
            action["position_multiplier"] = 0.7
        
        elif regime == "CAUTIOUS":
            if "trend" in market_state and market_state["trend"] < 0:
                action["direction"] = "SELL"
                action["confidence"] = 0.6
            else:
                action["direction"] = "NEUTRAL"
                action["confidence"] = 0.5
            action["position_multiplier"] = 0.8
        
        elif regime == "NORMAL":
            if "trend" in market_state:
                trend = market_state["trend"]
                if trend > 0.2:
                    action["direction"] = "BUY"
                    action["confidence"] = 0.6 + trend * 0.2
                elif trend < -0.2:
                    action["direction"] = "SELL"
                    action["confidence"] = 0.6 + abs(trend) * 0.2
                else:
                    action["direction"] = "NEUTRAL"
                    action["confidence"] = 0.5
            else:
                action["direction"] = "NEUTRAL"
                action["confidence"] = 0.5
            action["position_multiplier"] = 1.0
        
        elif regime == "BULLISH":
            action["direction"] = "BUY"
            action["confidence"] = 0.8
            action["position_multiplier"] = 1.2
        
        elif regime == "EUPHORIC":
            action["direction"] = "BUY"
            action["confidence"] = 0.6
            action["position_multiplier"] = 0.8
        
        return action
    
    def get_phoenix_action(self, metrics):
        """
        Get Phoenix action based on metrics
        
        Parameters:
        - metrics: Dictionary with market metrics
        
        Returns:
        - Dictionary with Phoenix action information
        """
        market_state = {
            "leverage_ratio": metrics.get("leverage_ratio", 1.0),
            "volatility": metrics.get("volatility", 15.0),
            "vix_term_structure": metrics.get("vix_term_structure", 1.0),
            "etf_flow_velocity": metrics.get("etf_flow_velocity", 0.0),
            "trend": metrics.get("trend", 0.0)
        }
        
        collapse_risk = self._get_collapse_risk(market_state)
        
        regime = self._get_regime_state(market_state)
        
        action = self._get_anti_failure_action(market_state, collapse_risk, regime)
        
        return {
            "decision": action,
            "regime": regime,
            "collapse_risk": collapse_risk,
            "timestamp": datetime.now()
        }
    
    def get_status(self):
        """
        Get Phoenix Network status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "regime_state": self.regime_state,
            "last_signal": self.last_signal,
            "last_check_time": self.last_check_time,
            "nano_nodes": list(self.nano_nodes.keys())
        }
