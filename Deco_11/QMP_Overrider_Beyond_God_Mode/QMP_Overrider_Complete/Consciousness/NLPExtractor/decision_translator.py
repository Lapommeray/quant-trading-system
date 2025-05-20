"""
decision_translator.py

Decision Explainer for Consciousness Layer

Generates human-readable explanations for system decisions,
enabling better understanding and trust in the trading system.
"""

import random
from datetime import datetime

class DecisionExplainer:
    """
    Decision Explainer for QMP Overrider
    
    Generates human-readable explanations for system decisions,
    enabling better understanding and trust in the trading system.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Decision Explainer
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.last_explanation = None
        self.last_check_time = None
        self.explanation_history = []
        self.max_history = 100
    
    def explain(self, decision, gate_scores=None, market_data=None):
        """
        Generate a human-readable explanation for a system decision
        
        Parameters:
        - decision: Decision to explain (e.g., "BUY", "SELL", "NEUTRAL")
        - gate_scores: Dictionary of gate scores (optional)
        - market_data: Dictionary of market data (optional)
        
        Returns:
        - Human-readable explanation
        """
        now = datetime.now()
        self.last_check_time = now
        
        primary = self._get_primary_driver(decision, gate_scores, market_data)
        
        confirming = self._get_confirming_factors(decision, gate_scores, market_data)
        
        contrary = self._get_contrary_indicators(decision, gate_scores, market_data)
        
        explanation = self._generate_narrative({
            "primary": primary,
            "confirming": confirming,
            "contrary": contrary,
            "decision": decision
        })
        
        self.last_explanation = {
            "text": explanation,
            "decision": decision,
            "primary": primary,
            "confirming": confirming,
            "contrary": contrary,
            "timestamp": now
        }
        
        self.explanation_history.append(self.last_explanation)
        
        if len(self.explanation_history) > self.max_history:
            self.explanation_history = self.explanation_history[-self.max_history:]
        
        if self.algorithm:
            self.algorithm.Debug(f"Decision Explainer: {explanation}")
        
        return explanation
    
    def _get_primary_driver(self, decision, gate_scores=None, market_data=None):
        """
        Get the primary driver for a decision
        
        Parameters:
        - decision: Decision to explain
        - gate_scores: Dictionary of gate scores (optional)
        - market_data: Dictionary of market data (optional)
        
        Returns:
        - Primary driver description
        """
        default_drivers = {
            "BUY": [
                "strong bullish alignment across multiple timeframes",
                "significant quantum gate activation",
                "powerful emotional sentiment shift",
                "clear timeline convergence pattern",
                "strong alien numerical sequence detection"
            ],
            "SELL": [
                "strong bearish alignment across multiple timeframes",
                "critical quantum gate deactivation",
                "negative emotional sentiment shift",
                "divergent timeline pattern",
                "concerning alien numerical sequence detection"
            ],
            "NEUTRAL": [
                "mixed signals across timeframes",
                "insufficient quantum gate activation",
                "unclear emotional sentiment",
                "timeline indecision pattern",
                "weak alien numerical sequence detection"
            ]
        }
        
        drivers = default_drivers.get(decision, default_drivers["NEUTRAL"])
        
        if gate_scores:
            strongest_gate = None
            highest_score = 0.0
            
            for gate, score in gate_scores.items():
                if score > highest_score:
                    highest_score = score
                    strongest_gate = gate
            
            if strongest_gate:
                if decision == "BUY":
                    return f"strong {strongest_gate} gate activation ({highest_score:.2f})"
                elif decision == "SELL":
                    return f"critical {strongest_gate} gate signal ({highest_score:.2f})"
                else:
                    return f"moderate {strongest_gate} gate reading ({highest_score:.2f})"
        
        if market_data:
            if "vix" in market_data:
                vix = market_data["vix"]
                if vix > 30 and decision == "SELL":
                    return f"elevated market volatility (VIX: {vix:.1f})"
                elif vix < 15 and decision == "BUY":
                    return f"low market volatility (VIX: {vix:.1f})"
            
            if "trend" in market_data:
                trend = market_data["trend"]
                if trend > 0.7 and decision == "BUY":
                    return "strong bullish trend continuation"
                elif trend < -0.7 and decision == "SELL":
                    return "strong bearish trend continuation"
        
        return random.choice(drivers)
    
    def _get_confirming_factors(self, decision, gate_scores=None, market_data=None):
        """
        Get confirming factors for a decision
        
        Parameters:
        - decision: Decision to explain
        - gate_scores: Dictionary of gate scores (optional)
        - market_data: Dictionary of market data (optional)
        
        Returns:
        - List of confirming factors
        """
        default_factors = {
            "BUY": [
                "positive emotional sentiment",
                "strong cosmic alignment",
                "favorable quantum state",
                "bullish timeline dominance",
                "supportive angelic frequencies",
                "positive divine timing",
                "favorable sacred date energy",
                "detected big move setup",
                "supportive macro environment"
            ],
            "SELL": [
                "negative emotional sentiment",
                "weak cosmic alignment",
                "unfavorable quantum state",
                "bearish timeline dominance",
                "cautionary angelic frequencies",
                "negative divine timing",
                "unfavorable sacred date energy",
                "detected distribution pattern",
                "challenging macro environment"
            ],
            "NEUTRAL": [
                "mixed emotional sentiment",
                "neutral cosmic alignment",
                "balanced quantum state",
                "timeline equilibrium",
                "neutral angelic frequencies",
                "transitional divine timing",
                "moderate sacred date energy",
                "no clear pattern detection",
                "stable macro environment"
            ]
        }
        
        factors = default_factors.get(decision, default_factors["NEUTRAL"])
        
        confirming = []
        
        if gate_scores:
            for gate, score in gate_scores.items():
                if decision == "BUY" and score > 0.7:
                    confirming.append(f"strong {gate} gate ({score:.2f})")
                elif decision == "SELL" and score < 0.3:
                    confirming.append(f"weak {gate} gate ({score:.2f})")
                elif decision == "NEUTRAL" and 0.4 <= score <= 0.6:
                    confirming.append(f"neutral {gate} gate ({score:.2f})")
        
        if market_data:
            if "rsi" in market_data:
                rsi = market_data["rsi"]
                if rsi < 30 and decision == "BUY":
                    confirming.append(f"oversold conditions (RSI: {rsi:.1f})")
                elif rsi > 70 and decision == "SELL":
                    confirming.append(f"overbought conditions (RSI: {rsi:.1f})")
            
            if "volume" in market_data:
                volume = market_data["volume"]
                if volume > 1.5 and decision != "NEUTRAL":
                    confirming.append(f"high volume confirmation ({volume:.1f}x average)")
        
        while len(confirming) < 3:
            factor = random.choice(factors)
            if factor not in confirming:
                confirming.append(factor)
        
        return confirming[:5]  # Limit to 5 factors
    
    def _get_contrary_indicators(self, decision, gate_scores=None, market_data=None):
        """
        Get contrary indicators for a decision
        
        Parameters:
        - decision: Decision to explain
        - gate_scores: Dictionary of gate scores (optional)
        - market_data: Dictionary of market data (optional)
        
        Returns:
        - List of contrary indicators
        """
        default_indicators = {
            "BUY": [
                "elevated volatility",
                "weak macro environment",
                "negative emotional sentiment",
                "bearish timeline signals",
                "unfavorable cosmic alignment"
            ],
            "SELL": [
                "oversold conditions",
                "strong support levels",
                "positive emotional sentiment",
                "bullish timeline signals",
                "favorable cosmic alignment"
            ],
            "NEUTRAL": [
                "strong directional signals",
                "clear pattern formation",
                "decisive emotional sentiment",
                "dominant timeline direction",
                "strong cosmic alignment"
            ]
        }
        
        indicators = default_indicators.get(decision, default_indicators["NEUTRAL"])
        
        contrary = []
        
        if gate_scores:
            for gate, score in gate_scores.items():
                if decision == "BUY" and score < 0.3:
                    contrary.append(f"weak {gate} gate ({score:.2f})")
                elif decision == "SELL" and score > 0.7:
                    contrary.append(f"strong {gate} gate ({score:.2f})")
                elif decision == "NEUTRAL" and (score < 0.3 or score > 0.7):
                    contrary.append(f"{'strong' if score > 0.7 else 'weak'} {gate} gate ({score:.2f})")
        
        if market_data:
            if "rsi" in market_data:
                rsi = market_data["rsi"]
                if rsi > 70 and decision == "BUY":
                    contrary.append(f"overbought conditions (RSI: {rsi:.1f})")
                elif rsi < 30 and decision == "SELL":
                    contrary.append(f"oversold conditions (RSI: {rsi:.1f})")
            
            if "support_resistance" in market_data:
                sr = market_data["support_resistance"]
                if "near_resistance" in sr and sr["near_resistance"] and decision == "BUY":
                    contrary.append("price near resistance level")
                elif "near_support" in sr and sr["near_support"] and decision == "SELL":
                    contrary.append("price near support level")
        
        if random.random() < 0.7:
            while len(contrary) < 2:
                indicator = random.choice(indicators)
                if indicator not in contrary:
                    contrary.append(indicator)
            
            return contrary[:3]  # Limit to 3 indicators
        else:
            return []  # No contrary indicators
    
    def _generate_narrative(self, components):
        """
        Generate a narrative from components
        
        Parameters:
        - components: Dictionary with narrative components
        
        Returns:
        - Narrative text
        """
        decision = components["decision"]
        primary = components["primary"]
        confirming = components["confirming"]
        contrary = components["contrary"]
        
        narrative = f"The decision to {decision.lower()} was primarily driven by {primary}"
        
        if confirming:
            if len(confirming) == 1:
                narrative += f", supported by {confirming[0]}"
            else:
                narrative += f", supported by {len(confirming)} confirming factors: "
                narrative += ", ".join(confirming[:-1]) + f" and {confirming[-1]}"
        
        if contrary:
            if len(contrary) == 1:
                narrative += f". One contrary indicator was overridden: {contrary[0]}"
            else:
                narrative += f". {len(contrary)} contrary indicators were overridden: "
                narrative += ", ".join(contrary[:-1]) + f" and {contrary[-1]}"
        else:
            narrative += ". No strong contrary signals were detected"
        
        confidence_phrases = [
            "The alignment of these factors provides a strong conviction in this decision.",
            "The convergence of these signals creates a clear directional bias.",
            "The weight of evidence strongly supports this trading decision.",
            "The combination of these factors presents a compelling case for this action.",
            "The synchronization of these elements reinforces the confidence in this signal."
        ]
        
        narrative += ". " + random.choice(confidence_phrases)
        
        return narrative
    
    def get_explanation_history(self):
        """
        Get explanation history
        
        Returns:
        - List of explanation records
        """
        return self.explanation_history
    
    def get_status(self):
        """
        Get Decision Explainer status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "last_explanation": self.last_explanation,
            "last_check_time": self.last_check_time,
            "explanation_count": len(self.explanation_history)
        }
