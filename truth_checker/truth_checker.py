"""
truth_checker.py

Cross-Validation Engine ("Truth Checker") for QMP Overrider

Compares signals from Aurora, Phoenix, and QMP for agreement or conflict,
providing a higher-level decision mechanism that ensures signal consistency.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TruthChecker:
    """
    Truth Checker for QMP Overrider
    
    Compares signals from Aurora, Phoenix, and QMP for agreement or conflict,
    providing a higher-level decision mechanism that ensures signal consistency.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Truth Checker
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.signals = {}
        self.last_resolution = {}
        
    def add_signal(self, source, signal, confidence=1.0, metadata=None):
        """
        Add a signal to the Truth Checker
        
        Parameters:
        - source: Signal source name (e.g., "aurora", "phoenix", "qmp")
        - signal: Signal direction ("BUY", "SELL", "NEUTRAL", etc.)
        - confidence: Signal confidence (0.0-1.0)
        - metadata: Additional signal metadata (optional)
        """
        self.signals[source] = {
            "signal": signal,
            "confidence": confidence,
            "metadata": metadata if metadata else {},
            "timestamp": datetime.now()
        }
        
        if self.algorithm:
            self.algorithm.Debug(f"Truth Checker: Added {signal} signal from {source} with confidence {confidence:.2f}")
    
    def resolve_signal(self):
        """
        Resolve all signals to determine the final decision
        
        Returns:
        - Dictionary with resolved signal information
          {
              "signal": str,
              "confidence": float,
              "agreement": str,
              "sources": list,
              "notes": str
          }
        """
        if not self.signals:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "agreement": "NONE",
                "sources": [],
                "notes": "No signals available"
            }
        
        directions = {}
        confidences = {}
        
        for source, data in self.signals.items():
            signal = data["signal"]
            confidence = data["confidence"]
            
            if signal not in directions:
                directions[signal] = []
                confidences[signal] = []
            
            directions[signal].append(source)
            confidences[signal].append(confidence)
        
        if len(directions) == 1:
            signal = list(directions.keys())[0]
            agreement = "FULL"
            confidence = sum(confidences[signal]) / len(confidences[signal])
            sources = directions[signal]
            notes = f"All {len(sources)} sources agree on {signal}"
            
            if signal == "BUY" and confidence > 0.8:
                signal = "STRONG_BUY"
            elif signal == "SELL" and confidence > 0.8:
                signal = "STRONG_SELL"
        elif len(directions) == len(self.signals):
            agreement = "NONE"
            
            best_signal = None
            best_confidence = 0.0
            
            for signal, confs in confidences.items():
                avg_conf = sum(confs) / len(confs)
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_signal = signal
            
            signal = best_signal
            confidence = best_confidence
            sources = directions[signal]
            notes = f"No agreement, using {signal} with highest confidence"
        else:
            agreement = "PARTIAL"
            
            best_signal = None
            most_sources = 0
            
            for sig, srcs in directions.items():
                if len(srcs) > most_sources:
                    most_sources = len(srcs)
                    best_signal = sig
            
            signal = best_signal
            confidence = sum(confidences[signal]) / len(confidences[signal])
            sources = directions[signal]
            notes = f"{len(sources)} of {len(self.signals)} sources agree on {signal}"
        
        result = {
            "signal": signal,
            "confidence": confidence,
            "agreement": agreement,
            "sources": sources,
            "notes": notes
        }
        
        self.last_resolution = result
        
        if self.algorithm:
            self.algorithm.Debug(f"Truth Checker: Resolved signal {signal} with {agreement} agreement")
        
        return result
    
    def clear_signals(self):
        """
        Clear all signals from the Truth Checker
        """
        self.signals = {}
