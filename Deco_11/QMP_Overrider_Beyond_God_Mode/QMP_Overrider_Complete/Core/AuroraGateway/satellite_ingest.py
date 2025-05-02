"""
satellite_ingest.py

Aurora Gateway Satellite Ingest

Provides advanced signal fusion and market intelligence by integrating
multiple data sources and analysis methods.
"""

import numpy as np
from datetime import datetime
import random

class AuroraGateway:
    """
    Aurora Gateway for QMP Overrider
    
    Provides advanced signal fusion and market intelligence by integrating
    multiple data sources and analysis methods.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Aurora Gateway
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.modules = {}
        self.signals = {}
        self.last_fusion = None
        self.last_check_time = None
        self._load_modules()
    
    def _load_modules(self):
        """Load specialized intelligence modules"""
        try:
            from .modules.tartarian_module import TartarianModule
            self.modules['tartarian'] = TartarianModule()
        except ImportError:
            print("Warning: TartarianModule not found.")
        
        try:
            from .modules.atlantean_module import AtlanteanModule
            self.modules['atlantean'] = AtlanteanModule()
        except ImportError:
            print("Warning: AtlanteanModule not found.")
        
        try:
            from .modules.fed_module import FedModule
            self.modules['fed'] = FedModule()
        except ImportError:
            print("Warning: FedModule not found.")
        
        try:
            from .modules.pyramid_module import PyramidModule
            self.modules['pyramid'] = PyramidModule()
        except ImportError:
            print("Warning: PyramidModule not found.")
        
        try:
            from .modules.alien_module import AlienModule
            self.modules['alien'] = AlienModule()
        except ImportError:
            print("Warning: AlienModule not found.")
        
        try:
            from .modules.vatican_module import VaticanModule
            self.modules['vatican'] = VaticanModule()
        except ImportError:
            print("Warning: VaticanModule not found.")
    
    def get_signal(self, market_state):
        """
        Get Aurora signal based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Dictionary with Aurora signal information
        """
        now = datetime.now()
        self.last_check_time = now
        
        signal = {
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "modules": {},
            "timestamp": now
        }
        
        if not market_state:
            return signal
        
        self._collect_signals(market_state)
        
        fusion = self._fuse_signals()
        
        signal["direction"] = fusion["signal"]
        signal["confidence"] = fusion["confidence"]
        signal["modules"] = fusion["modules"]
        
        self.last_fusion = fusion
        
        if self.algorithm:
            self.algorithm.Debug(f"Aurora Gateway: {signal['direction']} | Confidence: {signal['confidence']}")
            self.algorithm.Debug(f"Active Modules: {len(fusion['modules'])}")
        
        return signal
    
    def _collect_signals(self, market_state):
        """
        Collect signals from all modules
        
        Parameters:
        - market_state: Dictionary with market state information
        """
        self.signals = {}
        
        for name, module in self.modules.items():
            try:
                if hasattr(module, "get_signal"):
                    signal = module.get_signal(market_state)
                    self.signals[name] = signal
                else:
                    self.signals[name] = self._simulate_module_signal(name, market_state)
            except Exception as e:
                print(f"Error collecting signal from {name}: {e}")
    
    def _simulate_module_signal(self, module_name, market_state):
        """
        Simulate a module signal
        
        Parameters:
        - module_name: Module name
        - market_state: Dictionary with market state information
        
        Returns:
        - Dictionary with simulated signal information
        """
        if "timestamp" in market_state:
            seed = int(market_state["timestamp"].timestamp())
            random.seed(seed)
        
        direction = "NEUTRAL"
        confidence = 0.5
        
        if "trend" in market_state:
            trend = market_state["trend"]
            if trend > 0.2:
                direction = "BUY"
                confidence = 0.5 + trend * 0.3
            elif trend < -0.2:
                direction = "SELL"
                confidence = 0.5 + abs(trend) * 0.3
        
        if module_name == "tartarian":
            if "volatility" in market_state:
                volatility = market_state["volatility"]
                if volatility > 25:
                    direction = "SELL"
                    confidence = min(0.9, 0.5 + volatility / 100)
        
        elif module_name == "atlantean":
            if "liquidity" in market_state:
                liquidity = market_state["liquidity"]
                if liquidity < 0.5:
                    direction = "SELL"
                    confidence = min(0.9, 0.5 + (1 - liquidity) * 0.5)
        
        elif module_name == "fed":
            if "interest_rate" in market_state:
                rate = market_state["interest_rate"]
                if rate > 4.0:
                    direction = "SELL"
                    confidence = min(0.9, 0.5 + (rate - 4.0) * 0.1)
                elif rate < 2.0:
                    direction = "BUY"
                    confidence = min(0.9, 0.5 + (2.0 - rate) * 0.1)
        
        elif module_name == "pyramid":
            if "cycle_phase" in market_state:
                phase = market_state["cycle_phase"]
                if phase == "expansion":
                    direction = "BUY"
                    confidence = 0.8
                elif phase == "contraction":
                    direction = "SELL"
                    confidence = 0.8
        
        elif module_name == "alien":
            if "unusual_patterns" in market_state:
                unusual = market_state["unusual_patterns"]
                if unusual > 0.7:
                    direction = "SELL"
                    confidence = min(0.9, 0.5 + unusual * 0.3)
        
        elif module_name == "vatican":
            if "global_events" in market_state:
                events = market_state["global_events"]
                if events > 0.7:
                    direction = "SELL"
                    confidence = min(0.9, 0.5 + events * 0.3)
        
        confidence = max(0.1, min(0.9, confidence + random.uniform(-0.1, 0.1)))
        
        return {
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
    
    def _fuse_signals(self):
        """
        Fuse signals from all modules
        
        Returns:
        - Dictionary with fused signal information
        """
        if not self.signals:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "modules": {}
            }
        
        directions = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        confidences = {"BUY": [], "SELL": [], "NEUTRAL": []}
        modules = {"BUY": [], "SELL": [], "NEUTRAL": []}
        
        for name, signal in self.signals.items():
            direction = signal["direction"]
            confidence = signal["confidence"]
            
            directions[direction] += 1
            confidences[direction].append(confidence)
            modules[direction].append(name)
        
        best_direction = max(directions, key=directions.get)
        
        if directions[best_direction] == 0 or (best_direction == "NEUTRAL" and (directions["BUY"] > 0 or directions["SELL"] > 0)):
            avg_confidences = {}
            for direction, confs in confidences.items():
                if confs:
                    avg_confidences[direction] = sum(confs) / len(confs)
                else:
                    avg_confidences[direction] = 0.0
            
            best_direction = max(avg_confidences, key=avg_confidences.get)
        
        if confidences[best_direction]:
            confidence = sum(confidences[best_direction]) / len(confidences[best_direction])
            
            if directions[best_direction] > 1:
                confidence = min(0.95, confidence * (1.0 + 0.1 * (directions[best_direction] - 1)))
        else:
            confidence = 0.5
        
        module_info = {}
        for name, signal in self.signals.items():
            module_info[name] = {
                "direction": signal["direction"],
                "confidence": signal["confidence"],
                "agrees_with_fusion": signal["direction"] == best_direction
            }
        
        return {
            "signal": best_direction,
            "confidence": confidence,
            "modules": module_info
        }
    
    def get_status(self):
        """
        Get Aurora Gateway status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "modules": list(self.modules.keys()),
            "signals": self.signals,
            "last_fusion": self.last_fusion,
            "last_check_time": self.last_check_time
        }
