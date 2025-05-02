"""
signal_triangulation.py

Truth Validator Signal Triangulation

Compares signals from QMP, Phoenix, and Aurora to ensure signal consistency
and provide a higher-level decision mechanism.
"""

import numpy as np
from datetime import datetime

class TruthValidator:
    """
    Truth Validator for QMP Overrider
    
    Compares signals from QMP, Phoenix, and Aurora to ensure signal consistency
    and provide a higher-level decision mechanism.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Truth Validator
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.signals = {}
        self.last_resolution = None
        self.last_check_time = None
        self.tolerance = 0.15
    
    def add_signal(self, source, direction, confidence):
        """
        Add a signal from a source
        
        Parameters:
        - source: Signal source (e.g., "qmp", "phoenix", "aurora")
        - direction: Signal direction (e.g., "BUY", "SELL", "NEUTRAL")
        - confidence: Signal confidence (0.0 to 1.0)
        
        Returns:
        - True if successful, False otherwise
        """
        if not source or not direction:
            return False
        
        confidence = max(0.0, min(1.0, confidence))
        
        self.signals[source] = {
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
        
        return True
    
    def resolve_signal(self):
        """
        Resolve signals from all sources
        
        Returns:
        - Dictionary with resolved signal information
        """
        now = datetime.now()
        self.last_check_time = now
        
        resolution = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "agreement": "NONE",
            "sources": [],
            "timestamp": now
        }
        
        if not self.signals:
            return resolution
        
        phoenix_sig = self._get_phoenix_reading()
        aurora_sig = self._get_aurora_reading()
        qmp_sig = self._get_qmp_reading()
        
        if not phoenix_sig and not aurora_sig and not qmp_sig:
            return resolution
        
        triangulation = self._triangulate(phoenix_sig, aurora_sig, qmp_sig)
        
        resolution["signal"] = triangulation["signal"]
        resolution["confidence"] = triangulation["confidence"]
        resolution["agreement"] = triangulation["agreement"]
        resolution["sources"] = triangulation["sources"]
        
        self.last_resolution = resolution
        
        if self.algorithm:
            self.algorithm.Debug(f"Truth Validator: {resolution['signal']} | Confidence: {resolution['confidence']}")
            self.algorithm.Debug(f"Agreement: {resolution['agreement']} | Sources: {resolution['sources']}")
        
        return resolution
    
    def _get_phoenix_reading(self):
        """
        Get Phoenix signal
        
        Returns:
        - Dictionary with Phoenix signal information
        """
        return self.signals.get("phoenix")
    
    def _get_aurora_reading(self):
        """
        Get Aurora signal
        
        Returns:
        - Dictionary with Aurora signal information
        """
        return self.signals.get("aurora")
    
    def _get_qmp_reading(self):
        """
        Get QMP signal
        
        Returns:
        - Dictionary with QMP signal information
        """
        return self.signals.get("qmp")
    
    def _triangulate(self, phoenix_sig, aurora_sig, qmp_sig):
        """
        Triangulate signals from Phoenix, Aurora, and QMP
        
        Parameters:
        - phoenix_sig: Phoenix signal
        - aurora_sig: Aurora signal
        - qmp_sig: QMP signal
        
        Returns:
        - Dictionary with triangulated signal information
        """
        directions = {}
        confidences = {}
        sources = {}
        
        if phoenix_sig:
            direction = phoenix_sig["direction"]
            confidence = phoenix_sig["confidence"]
            
            if direction not in directions:
                directions[direction] = 0
                confidences[direction] = []
                sources[direction] = []
            
            directions[direction] += 1
            confidences[direction].append(confidence)
            sources[direction].append("phoenix")
        
        if aurora_sig:
            direction = aurora_sig["direction"]
            confidence = aurora_sig["confidence"]
            
            if direction not in directions:
                directions[direction] = 0
                confidences[direction] = []
                sources[direction] = []
            
            directions[direction] += 1
            confidences[direction].append(confidence)
            sources[direction].append("aurora")
        
        if qmp_sig:
            direction = qmp_sig["direction"]
            confidence = qmp_sig["confidence"]
            
            if direction not in directions:
                directions[direction] = 0
                confidences[direction] = []
                sources[direction] = []
            
            directions[direction] += 1
            confidences[direction].append(confidence)
            sources[direction].append("qmp")
        
        best_direction = None
        most_signals = 0
        
        for direction, count in directions.items():
            if count > most_signals:
                most_signals = count
                best_direction = direction
        
        if best_direction is None or (most_signals == 1 and len(directions) > 1):
            avg_confidences = {}
            for direction, confs in confidences.items():
                if confs:
                    avg_confidences[direction] = sum(confs) / len(confs)
                else:
                    avg_confidences[direction] = 0.0
            
            best_direction = max(avg_confidences, key=avg_confidences.get)
        
        if confidences.get(best_direction):
            confidence = sum(confidences[best_direction]) / len(confidences[best_direction])
            
            if directions[best_direction] > 1:
                confidence = min(0.95, confidence * (1.0 + 0.1 * (directions[best_direction] - 1)))
        else:
            confidence = 0.0
        
        total_signals = sum(directions.values())
        
        if total_signals == 0:
            agreement = "NONE"
        elif directions.get(best_direction, 0) == total_signals:
            agreement = "FULL"
        elif directions.get(best_direction, 0) >= total_signals - 1:
            agreement = "STRONG"
        elif directions.get(best_direction, 0) > total_signals / 2:
            agreement = "MAJORITY"
        else:
            agreement = "WEAK"
        
        return {
            "signal": best_direction,
            "confidence": confidence,
            "agreement": agreement,
            "sources": sources.get(best_direction, [])
        }
    
    def validate(self, consensus):
        """
        Validate a consensus signal
        
        Parameters:
        - consensus: Dictionary with consensus signal information
        
        Returns:
        - True if valid, False otherwise
        """
        if not consensus:
            return False
        
        phoenix_sig = self._get_phoenix_reading()
        aurora_sig = self._get_aurora_reading()
        qmp_sig = self._get_qmp_reading()
        
        if not phoenix_sig and not aurora_sig and not qmp_sig:
            return False
        
        signals = []
        
        if phoenix_sig:
            phoenix_val = 1.0 if phoenix_sig["direction"] == "BUY" else (-1.0 if phoenix_sig["direction"] == "SELL" else 0.0)
            signals.append(phoenix_val * phoenix_sig["confidence"])
        
        if aurora_sig:
            aurora_val = 1.0 if aurora_sig["direction"] == "BUY" else (-1.0 if aurora_sig["direction"] == "SELL" else 0.0)
            signals.append(aurora_val * aurora_sig["confidence"])
        
        if qmp_sig:
            qmp_val = 1.0 if qmp_sig["direction"] == "BUY" else (-1.0 if qmp_sig["direction"] == "SELL" else 0.0)
            signals.append(qmp_val * qmp_sig["confidence"])
        
        if signals:
            avg = sum(signals) / len(signals)
            
            return all(abs(s - avg) < abs(avg) * self.tolerance for s in signals)
        
        return False
    
    def get_status(self):
        """
        Get Truth Validator status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "signals": self.signals,
            "last_resolution": self.last_resolution,
            "last_check_time": self.last_check_time,
            "tolerance": self.tolerance
        }
