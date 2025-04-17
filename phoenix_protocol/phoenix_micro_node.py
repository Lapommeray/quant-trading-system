"""
phoenix_micro_node.py

Minimal integration interface for Phoenix Protocol.
Feeds survival actions into a main trading system like QMP or Aurora.

This module provides a simple interface to access Phoenix Protocol's
market survival actions and regime detection capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class PhoenixProtocol:
    """
    Phoenix Protocol integration for QMP Overrider
    
    Provides market collapse detection, regime classification,
    and anti-failure decision making capabilities.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Phoenix Protocol
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.last_metrics = {}
        self.last_result = {}
        
    def get_phoenix_action(self, current_market_metrics):
        """
        Get Phoenix Protocol's survival action based on current market metrics
        
        Parameters:
        - current_market_metrics: Dictionary containing market metrics
          {
              "leverage_ratio": float,
              "volatility": float,
              "vix_term_structure": float,
              "etf_flow_velocity": float
          }
        
        Returns:
        - Dictionary with Phoenix Protocol's analysis and decision
          {
              "collapse_score": float,
              "regime_status": str,
              "failure_mode": str,
              "decision": {
                  "action": str,
                  "position_multiplier": float,
                  "notes": str
              }
          }
        """
        self.last_metrics = current_market_metrics
        
        collapse_info = self._detect_collapse_pattern(current_market_metrics)
        
        regime_status = self._classify_regime(
            current_market_metrics["volatility"],
            current_market_metrics["vix_term_structure"],
            current_market_metrics["etf_flow_velocity"]
        )
        
        decision = self._anti_failure_decision(regime_status, collapse_info["score"])
        
        result = {
            "collapse_match": collapse_info["match"],
            "collapse_score": collapse_info["score"],
            "regime_status": regime_status,
            "failure_mode": collapse_info["failure_mode"],
            "decision": decision
        }
        
        self.last_result = result
        
        if self.algorithm:
            self.algorithm.Debug(f"Phoenix Protocol: {regime_status} regime, collapse score {collapse_info['score']:.2f}")
            self.algorithm.Debug(f"Phoenix Action: {decision['action']} (multiplier: {decision['position_multiplier']:.2f})")
        
        return result
    
    def _detect_collapse_pattern(self, metrics):
        """
        Detect market collapse patterns based on current metrics
        
        This is a simplified implementation of the collapse memory module.
        In a full implementation, this would compare current metrics to
        historical collapse patterns from 1929, 1987, 2000, 2008, etc.
        """
        leverage_factor = max(0, min(1, metrics["leverage_ratio"] / 3.0))
        volatility_factor = max(0, min(1, metrics["volatility"] / 40.0))
        term_structure_factor = max(0, min(1, (1.0 - metrics["vix_term_structure"]) * 2))
        flow_factor = max(0, min(1, (metrics["etf_flow_velocity"] * -1) / 2.0))
        
        score = (
            leverage_factor * 0.3 +
            volatility_factor * 0.3 +
            term_structure_factor * 0.2 +
            flow_factor * 0.2
        )
        
        if score > 0.8:
            match = "1929_CRASH"
            failure_mode = "LIQUIDITY_COLLAPSE"
        elif score > 0.6:
            match = "2008_CRASH"
            failure_mode = "CREDIT_SEIZURE"
        elif score > 0.4:
            match = "2000_BUBBLE"
            failure_mode = "VALUATION_RESET"
        elif score > 0.2:
            match = "2018_CORRECTION"
            failure_mode = "MOMENTUM_REVERSAL"
        else:
            match = "NORMAL_MARKET"
            failure_mode = "NONE"
        
        return {
            "score": score,
            "match": match,
            "failure_mode": failure_mode
        }
    
    def _classify_regime(self, volatility, vix_term_structure, etf_flow_velocity):
        """
        Classify the current market regime
        
        This is a simplified implementation of the regime sentience module.
        In a full implementation, this would use machine learning to
        classify the current market regime based on multiple factors.
        """
        if volatility > 30:
            if etf_flow_velocity < -1.0:
                return "CRISIS"
            else:
                return "VOLATILE"
        elif volatility < 10:
            if vix_term_structure > 1.1:
                return "COMPLACENT"
            else:
                return "LOW_VOL"
        elif etf_flow_velocity > 1.0:
            return "BULLISH"
        elif etf_flow_velocity < -0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _anti_failure_decision(self, regime, collapse_score):
        """
        Generate anti-failure decision based on regime and collapse score
        
        This is a simplified implementation of the anti-failure engine.
        In a full implementation, this would use more sophisticated
        decision-making algorithms to determine the optimal action.
        """
        if regime == "CRISIS":
            action = "REDUCE_ALL"
            multiplier = 0.0
            notes = "Crisis regime detected - move to cash"
        elif regime == "VOLATILE" and collapse_score > 0.5:
            action = "REDUCE_RISK"
            multiplier = 0.25
            notes = "High volatility with collapse risk - reduce exposure"
        elif regime == "BEARISH" and collapse_score > 0.3:
            action = "HEDGE"
            multiplier = 0.5
            notes = "Bearish trend with moderate collapse risk - add hedges"
        elif regime == "COMPLACENT" and collapse_score > 0.2:
            action = "CAUTION"
            multiplier = 0.75
            notes = "Complacency detected with low collapse risk - reduce leverage"
        elif regime == "BULLISH" and collapse_score < 0.2:
            action = "FULL_RISK"
            multiplier = 1.0
            notes = "Bullish trend with minimal collapse risk - full exposure"
        else:
            action = "NORMAL"
            multiplier = 0.8
            notes = "Normal market conditions - standard exposure"
        
        return {
            "action": action,
            "position_multiplier": multiplier,
            "notes": notes
        }
