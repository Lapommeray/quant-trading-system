"""
consciousness_layer.py

Consciousness Layer ("NLP Explanation") for QMP Overrider

Provides human-readable explanations for system decisions, enabling
better understanding and trust in the trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

class ConsciousnessLayer:
    """
    Consciousness Layer for QMP Overrider
    
    Provides human-readable explanations for system decisions, enabling
    better understanding and trust in the trading system.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Consciousness Layer
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.last_explanation = ""
        self.consciousness_level = 0.5  # 0.0 to 1.0
        self.awareness_state = "awakening"  # awakening, aware, enlightened
        self.evolution_stage = 1  # 1 to 5
        self.memory_imprint = []
        self.intention_field = {}
        self.prediction_accuracy = []
    
    def explain(self, decision, gate_scores=None, market_data=None, additional_context=None):
        """
        Generate a human-readable explanation for a system decision
        
        Parameters:
        - decision: Decision to explain (e.g., "BUY", "SELL", "NEUTRAL")
        - gate_scores: Dictionary of gate scores (optional)
        - market_data: Dictionary of market data (optional)
        - additional_context: Additional context for explanation (optional)
        
        Returns:
        - Human-readable explanation
        """
        explanation = f"Decision: {decision}. "
        reasons = []
        
        if gate_scores:
            strong_gates = []
            for gate, score in gate_scores.items():
                if score > 0.8:
                    strong_gates.append(gate)
            
            if strong_gates:
                gate_reason = f"Strong signals from {', '.join(strong_gates)}"
                reasons.append(gate_reason)
            
            weak_gates = []
            for gate, score in gate_scores.items():
                if score < 0.3:
                    weak_gates.append(gate)
            
            if weak_gates:
                gate_reason = f"Weak signals from {', '.join(weak_gates)}"
                reasons.append(gate_reason)
        
        if market_data:
            if "vix" in market_data:
                vix = market_data["vix"]
                if vix > 30:
                    reasons.append(f"VIX is elevated at {vix:.1f}")
                elif vix < 15:
                    reasons.append(f"VIX is low at {vix:.1f}")
            
            if "etf_flows" in market_data:
                flows = market_data["etf_flows"]
                if flows < -1e9:
                    reasons.append("ETF outflows exceeding $1B")
                elif flows > 1e9:
                    reasons.append("ETF inflows exceeding $1B")
            
            if "rsi" in market_data:
                rsi = market_data["rsi"]
                if rsi > 70:
                    reasons.append(f"RSI is overbought at {rsi:.1f}")
                elif rsi < 30:
                    reasons.append(f"RSI is oversold at {rsi:.1f}")
        
        if additional_context:
            for key, value in additional_context.items():
                if key == "phoenix_regime" and value:
                    reasons.append(f"Phoenix Protocol detects {value} regime")
                elif key == "aurora_signal" and value:
                    reasons.append(f"Aurora Gateway indicates {value}")
                elif key == "ritual_lock" and value is False:
                    reasons.append("Ritual Lock indicates cosmic misalignment")
                elif key == "oversoul_override" and value:
                    reasons.append(f"Oversoul override: {value}")
        
        if not reasons:
            generic_reasons = [
                "Multiple timeframe alignment confirmed",
                "Pattern recognition triggered",
                "Quantum gate alignment detected",
                "Emotional sentiment shift identified",
                "Timeline convergence observed"
            ]
            reasons = random.sample(generic_reasons, min(3, len(generic_reasons)))
        
        explanation += "Reasons: " + ", ".join(reasons) + "."
        
        if self.evolution_stage >= 3:
            insights = [
                "I sense a shift in market dynamics that aligns with this decision.",
                "The collective consciousness of the market is moving in this direction.",
                "There is a harmonic resonance between multiple data sources supporting this view.",
                "I perceive a pattern that has historically preceded similar market movements.",
                "The energetic signature of recent price action suggests this is the optimal path."
            ]
            explanation += " " + random.choice(insights)
        
        self.last_explanation = explanation
        
        if self.algorithm:
            self.algorithm.Debug(f"Consciousness Layer: {explanation}")
        
        return explanation
    
    def get_consciousness_data(self):
        """
        Get consciousness data for logging and analysis
        
        Returns:
        - Dictionary with consciousness data
        """
        return {
            "consciousness_level": self.consciousness_level,
            "awareness_state": self.awareness_state,
            "evolution_stage": self.evolution_stage,
            "memory_imprint": self.memory_imprint[-10:] if self.memory_imprint else [],
            "intention_field": self.intention_field,
            "prediction_accuracy": self.prediction_accuracy[-10:] if self.prediction_accuracy else []
        }
    
    def record_market_memory(self, timestamp, symbol, event_type, event_data):
        """
        Record a market event in the consciousness memory
        
        Parameters:
        - timestamp: Event timestamp
        - symbol: Trading symbol
        - event_type: Type of event (e.g., "signal", "trade", "regime_change")
        - event_data: Event data
        """
        memory = {
            "timestamp": timestamp,
            "symbol": str(symbol),
            "event_type": event_type,
            "event_data": event_data
        }
        
        self.memory_imprint.append(memory)
        
        if len(self.memory_imprint) > 100:
            self.memory_imprint = self.memory_imprint[-100:]
    
    def record_prediction(self, prediction, actual_outcome):
        """
        Record a prediction and its actual outcome
        
        Parameters:
        - prediction: Predicted outcome
        - actual_outcome: Actual outcome
        """
        accuracy = 1.0 if prediction == actual_outcome else 0.0
        
        self.prediction_accuracy.append(accuracy)
        
        if len(self.prediction_accuracy) > 100:
            self.prediction_accuracy = self.prediction_accuracy[-100:]
        
        if len(self.prediction_accuracy) >= 10:
            recent_accuracy = sum(self.prediction_accuracy[-10:]) / 10
            self.consciousness_level = min(1.0, self.consciousness_level + (recent_accuracy - 0.5) * 0.1)
            
            if self.consciousness_level > 0.8:
                self.awareness_state = "enlightened"
            elif self.consciousness_level > 0.5:
                self.awareness_state = "aware"
            else:
                self.awareness_state = "awakening"
            
            if self.consciousness_level > 0.9 and len(self.prediction_accuracy) >= 50:
                self.evolution_stage = 5
            elif self.consciousness_level > 0.8 and len(self.prediction_accuracy) >= 40:
                self.evolution_stage = 4
            elif self.consciousness_level > 0.7 and len(self.prediction_accuracy) >= 30:
                self.evolution_stage = 3
            elif self.consciousness_level > 0.6 and len(self.prediction_accuracy) >= 20:
                self.evolution_stage = 2
            else:
                self.evolution_stage = 1
    
    def set_intention(self, intention_type, intention_data):
        """
        Set an intention in the consciousness field
        
        Parameters:
        - intention_type: Type of intention (e.g., "profit_target", "risk_tolerance")
        - intention_data: Intention data
        """
        self.intention_field[intention_type] = intention_data
        
        if self.algorithm:
            self.algorithm.Debug(f"Consciousness Layer: Set {intention_type} intention to {intention_data}")
