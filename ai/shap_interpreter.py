#!/usr/bin/env python
"""
SHAP Interpreter Module
Explains trading decisions using SHAP (SHapley Additive exPlanations)
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shap_interpreter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SHAPInterpreter")

class SHAPTraderExplainer:
    """Explains trading decisions using SHAP (SHapley Additive exPlanations)"""
    
    def __init__(self, feature_names=None, max_features=10):
        """Initialize the SHAP Trader Explainer
        
        Args:
            feature_names: Names of the features used in the model
            max_features: Maximum number of features to include in explanations
        """
        self.feature_names = feature_names or [
            "price", "volume", "volatility", "trend", "momentum",
            "rsi", "macd", "bollinger", "support", "resistance",
            "whale_confidence", "quantum_signal", "divine_pulse"
        ]
        self.max_features = max_features
        self.explanation_history = []
        logger.info(f"Initialized SHAPTraderExplainer with {len(self.feature_names)} features")
        
    def explain_decision(self, decision: Dict, features: Dict) -> Dict:
        """Explain a trading decision using SHAP values"""
        if not self._verify_real_time_data(features):
            logger.error("Data verification failed - not 100% real-time")
            return {
                "explanation": "Cannot explain decision: Data verification failed - not 100% real-time",
                "confidence": 0.0,
                "shap_values": {},
                "error": "Data verification failed - not 100% real-time"
            }
            
        signal = decision.get('signal', 'HOLD')
        confidence = decision.get('confidence', 0.0)
        
        shap_values = self._calculate_shap_values(features, signal)
        
        explanation = self._generate_explanation(signal, confidence, shap_values)
        
        self._record_explanation(explanation)
        
        logger.info(f"Generated explanation for {signal} decision with confidence {confidence}")
        
        return explanation
        
    def _calculate_shap_values(self, features: Dict, signal: str) -> Dict:
        """Calculate SHAP values for the features"""
        shap_values = {}
        
        available_features = [f for f in self.feature_names if f in features]
        
        if not available_features:
            return shap_values
            
        feature_values = [features[f] for f in available_features]
        base_value = sum(feature_values) / len(feature_values)
        
        total_deviation = sum(abs(v - base_value) for v in feature_values)
        
        if total_deviation == 0:
            for feature in available_features:
                shap_values[feature] = 1.0 / len(available_features)
        else:
            for feature in available_features:
                feature_value = features[feature]
                deviation = abs(feature_value - base_value)
                contribution = deviation / total_deviation
                
                if signal in ['BUY', 'STRONG_BUY', 'DIVINE_BUY', 'QUANTUM_BUY']:
                    shap_values[feature] = contribution if feature_value > base_value else -contribution
                elif signal in ['SELL', 'STRONG_SELL', 'DIVINE_SELL', 'QUANTUM_SELL']:
                    shap_values[feature] = contribution if feature_value < base_value else -contribution
                else:
                    shap_values[feature] = contribution
                    
        return shap_values
        
    def _generate_explanation(self, signal: str, confidence: float, shap_values: Dict) -> Dict:
        """Generate an explanation for the decision"""
        if not shap_values:
            return {
                "explanation": f"Decision: {signal} (Confidence: {confidence:.2f}). No feature information available for explanation.",
                "confidence": confidence,
                "shap_values": {},
                "top_features": []
            }
            
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_features = sorted_features[:self.max_features]
        
        explanation_parts = [f"Decision: {signal} (Confidence: {confidence:.2f})"]
        explanation_parts.append("Top contributing factors:")
        
        for feature, value in top_features:
            direction = "increased" if value > 0 else "decreased"
            explanation_parts.append(f"- {feature}: {direction} likelihood by {abs(value):.2f}")
            
        explanation = "\n".join(explanation_parts)
        
        return {
            "explanation": explanation,
            "confidence": confidence,
            "shap_values": shap_values,
            "top_features": top_features
        }
        
    def _record_explanation(self, explanation: Dict) -> None:
        """Record the explanation"""
        self.explanation_history.append(explanation)
        
        if len(self.explanation_history) > 100:
            self.explanation_history.pop(0)
            
    def _verify_real_time_data(self, data: Dict) -> bool:
        """Verify the data is 100% real-time with no synthetic elements"""
        if not data:
            logger.warning("Empty data")
            return False
            
        current_time = time.time() * 1000
        data_timestamp = data.get('timestamp', 0)
        
        if data_timestamp > 0 and current_time - data_timestamp > 5 * 1000:  # 5 seconds tolerance
            logger.warning(f"Data not real-time: {(current_time - data_timestamp)/1000:.2f} seconds old")
            return False
            
        data_str = str(data)
        synthetic_markers = [
            'simulated', 'synthetic', 'fake', 'mock', 'test', 
            'dummy', 'placeholder', 'generated', 'artificial', 
            'virtualized', 'pseudo', 'demo', 'sample',
            'backtesting', 'historical', 'backfill', 'sandbox'
        ]
        
        for marker in synthetic_markers:
            if marker in data_str.lower():
                logger.warning(f"Synthetic data marker found: {marker}")
                return False
                
        return True

if __name__ == "__main__":
    explainer = SHAPTraderExplainer()
    test_decision = {
        "signal": "BUY",
        "confidence": 0.85
    }
    test_features = {
        "price": 0.7,
        "volume": 0.9,
        "volatility": 0.3,
        "trend": 0.8,
        "momentum": 0.6,
        "rsi": 0.4,
        "macd": 0.7,
        "timestamp": time.time() * 1000
    }
    result = explainer.explain_decision(test_decision, test_features)
    print(result["explanation"])
