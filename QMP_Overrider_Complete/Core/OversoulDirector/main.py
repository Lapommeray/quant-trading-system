"""
main.py

OversoulDirector Main Module

Central routing intelligence for the QMP Overrider system.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np

from signal_router import SignalRouter
from module_activator import ModuleActivator
from priority_engine import PriorityEngine

class OversoulDirector:
    """
    OversoulDirector for QMP Overrider
    
    Central routing intelligence that manages all modules and provides
    meta-control over the entire system.
    """
    
    def __init__(self):
        """Initialize the OversoulDirector"""
        self.modules = {}
        self.signal_router = SignalRouter(self)
        self.module_activator = ModuleActivator(self)
        self.priority_engine = PriorityEngine(self)
        self.priority_matrix = self.priority_engine.get_priority_matrix()
        self.last_market_state = None
        self.last_consensus = None
        self.last_decision = None
        self.initialized = False
    
    def initialize(self, mode="full"):
        """
        Initialize the OversoulDirector and load all modules
        
        Parameters:
        - mode: Initialization mode ("full", "lite", "backtest")
        
        Returns:
        - True if successful, False otherwise
        """
        print(f"Initializing OversoulDirector in {mode} mode...")
        
        try:
            if mode == "full":
                self._load_all_modules()
            elif mode == "lite":
                self._load_core_modules()
            elif mode == "backtest":
                self._load_backtest_modules()
            else:
                print(f"Unknown mode: {mode}")
                return False
            
            self.priority_engine.load_priority_matrix()
            self.priority_matrix = self.priority_engine.get_priority_matrix()
            
            self.initialized = True
            print("OversoulDirector initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing OversoulDirector: {e}")
            return False
    
    def _load_all_modules(self):
        """Load all modules"""
        self._load_core_modules()
        self._load_advanced_modules()
        
        try:
            from Optimization.AgentLab.darwinian_ga import StrategyEvolver
            self.modules['darwin'] = StrategyEvolver()
        except ImportError:
            print("Warning: StrategyEvolver module not found.")
        
        try:
            from Optimization.EventProbability.market_tomography import EventProbabilityEngine
            self.modules['event_probability'] = EventProbabilityEngine()
        except ImportError:
            print("Warning: EventProbabilityEngine module not found.")
        
        try:
            from Consciousness.NLPExtractor.decision_translator import DecisionExplainer
            self.modules['consciousness'] = DecisionExplainer()
        except ImportError:
            print("Warning: DecisionExplainer module not found.")
        
        try:
            from Consciousness.MetaMonitor.integrity_checker import MetaMonitor
            self.modules['meta_monitor'] = MetaMonitor()
        except ImportError:
            print("Warning: MetaMonitor module not found.")
    
    def _load_core_modules(self):
        """Load core modules"""
        try:
            from Core.PhoenixProtocol.gateway_controller import PhoenixNetwork
            self.modules['phoenix'] = PhoenixNetwork()
        except ImportError:
            print("Warning: PhoenixNetwork module not found.")
        
        try:
            from Core.AuroraGateway.satellite_ingest import AuroraGateway
            self.modules['aurora'] = AuroraGateway()
        except ImportError:
            print("Warning: AuroraGateway module not found.")
        
        try:
            from Validation.TruthChecker.signal_triangulation import TruthValidator
            self.modules['truth'] = TruthValidator()
        except ImportError:
            print("Warning: TruthValidator module not found.")
        
        try:
            from Validation.RitualLock.solar_aligner import CosmicSynchronizer
            self.modules['ritual'] = CosmicSynchronizer()
        except ImportError:
            print("Warning: CosmicSynchronizer module not found.")
    
    def _load_backtest_modules(self):
        """Load modules for backtesting"""
        self._load_core_modules()
        
        try:
            from Optimization.AgentLab.darwinian_ga import StrategyEvolver
            self.modules['darwin'] = StrategyEvolver()
        except ImportError:
            print("Warning: StrategyEvolver module not found.")
            
    def _load_advanced_modules(self):
        """Load advanced modules from CERN, Neuralink, and Quantum Consciousness"""
        try:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'advanced_modules'))
            from module_interface import AdvancedModuleInterface
            print("Advanced Module Interface loaded successfully.")
        except ImportError:
            print("Warning: Advanced Module Interface not found.")
            return
            
        try:
            from cern_physics.hadron_collider_market_analyzer import HadronColliderMarketAnalyzer
            self.modules['hadron_collider'] = HadronColliderMarketAnalyzer()
            print("CERN Hadron Collider Market Analyzer loaded successfully.")
        except ImportError:
            print("Warning: CERN Hadron Collider Market Analyzer module not found.")
            
        try:
            from neuralink_bci.neural_spike_pattern_recognizer import NeuralSpikePatternRecognizer
            self.modules['neural_spike'] = NeuralSpikePatternRecognizer()
            print("Neuralink Neural Spike Pattern Recognizer loaded successfully.")
        except ImportError:
            print("Warning: Neuralink Neural Spike Pattern Recognizer module not found.")
            
        try:
            from quantum_consciousness.quantum_entanglement_market_correlator import QuantumEntanglementMarketCorrelator
            self.modules['quantum_entanglement'] = QuantumEntanglementMarketCorrelator()
            print("Quantum Entanglement Market Correlator loaded successfully.")
        except ImportError:
            print("Warning: Quantum Entanglement Market Correlator module not found.")
            
        try:
            from never_loss_intelligence.temporal_probability_calculator import TemporalProbabilityCalculator
            self.modules['temporal_probability'] = TemporalProbabilityCalculator()
            print("Temporal Probability Calculator loaded successfully.")
        except ImportError:
            print("Warning: Temporal Probability Calculator module not found.")
    
    def route_signal(self, market_state):
        """
        Route a signal through the system based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Dictionary with routing results
        """
        if not self.initialized:
            print("OversoulDirector not initialized.")
            return None
        
        self.last_market_state = market_state
        
        active_modules = self.module_activator.determine_active_modules(market_state)
        
        signals = {}
        
        if 'phoenix' in active_modules and active_modules['phoenix'] > 0.5:
            if 'phoenix' in self.modules:
                phoenix_signal = self.modules['phoenix'].get_signal(market_state)
                signals['phoenix'] = phoenix_signal
        
        if 'aurora' in active_modules and active_modules['aurora'] > 0.5:
            if 'aurora' in self.modules:
                aurora_signal = self.modules['aurora'].get_signal(market_state)
                signals['aurora'] = aurora_signal
        
        if 'qmp' in active_modules and active_modules['qmp'] > 0.5:
            if 'qmp_signal' in market_state:
                signals['qmp'] = market_state['qmp_signal']
                
        for module_name in ['hadron_collider', 'neural_spike', 'quantum_entanglement', 'temporal_probability']:
            if module_name in active_modules and active_modules[module_name] > 0.5:
                if module_name in self.modules:
                    module_signal = self.modules[module_name].get_signal(market_state)
                    signals[module_name] = module_signal
        
        consensus = None
        if 'truth' in active_modules and active_modules['truth'] > 0.5:
            if 'truth' in self.modules:
                for source, signal in signals.items():
                    self.modules['truth'].add_signal(source, signal['direction'], signal['confidence'])
                
                consensus = self.modules['truth'].resolve_signal()
        else:
            consensus = self._get_simple_consensus(signals)
        
        self.last_consensus = consensus
        
        if consensus is None:
            consensus = {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "agreement": "NONE",
                "sources": [],
                "notes": "No consensus available"
            }
        
        if 'ritual' in active_modules and active_modules['ritual'] > 0.5:
            if 'ritual' in self.modules:
                is_aligned = self.modules['ritual'].is_aligned(consensus['signal'])
                
                if not is_aligned:
                    consensus['signal'] = "NEUTRAL"
                    consensus['notes'] = "Ritual Lock prevented trade due to cosmic misalignment"
        
        if 'darwin' in active_modules and active_modules['darwin'] > 0.5:
            if 'darwin' in self.modules:
                consensus = self.modules['darwin'].optimize(consensus)
        
        if 'consciousness' in active_modules and active_modules['consciousness'] > 0.5:
            if 'consciousness' in self.modules:
                explanation = self.modules['consciousness'].explain(
                    consensus['signal'],
                    market_state.get('gate_scores'),
                    market_state
                )
                
                consensus['explanation'] = explanation
        
        self.last_decision = consensus
        
        return consensus
    
    def _get_simple_consensus(self, signals):
        """
        Get a simple consensus from signals
        
        Parameters:
        - signals: Dictionary with signals from different modules
        
        Returns:
        - Dictionary with consensus information
        """
        if not signals:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "agreement": "NONE",
                "sources": [],
                "notes": "No signals available"
            }
        
        directions = {}
        confidences = {}
        
        for source, signal in signals.items():
            direction = signal['direction']
            confidence = signal['confidence']
            
            if direction not in directions:
                directions[direction] = []
                confidences[direction] = []
            
            directions[direction].append(source)
            confidences[direction].append(confidence)
        
        best_direction = None
        most_sources = 0
        
        for direction, sources in directions.items():
            if len(sources) > most_sources:
                most_sources = len(sources)
                best_direction = direction
        
        if best_direction:
            avg_confidence = sum(confidences[best_direction]) / len(confidences[best_direction])
            
            if len(directions) == 1:
                agreement = "FULL"
                notes = f"All {len(signals)} sources agree on {best_direction}"
            elif len(directions[best_direction]) == len(signals) - 1:
                agreement = "STRONG"
                notes = f"{len(directions[best_direction])} of {len(signals)} sources agree on {best_direction}"
            elif len(directions[best_direction]) > len(signals) / 2:
                agreement = "MAJORITY"
                notes = f"Majority ({len(directions[best_direction])} of {len(signals)}) sources agree on {best_direction}"
            else:
                agreement = "WEAK"
                notes = f"Weak agreement ({len(directions[best_direction])} of {len(signals)}) on {best_direction}"
            
            return {
                "signal": best_direction,
                "confidence": avg_confidence,
                "agreement": agreement,
                "sources": directions[best_direction],
                "notes": notes
            }
        else:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "agreement": "NONE",
                "sources": [],
                "notes": "No consensus reached"
            }
    
    def get_modules(self):
        """
        Get all loaded modules
        
        Returns:
        - Dictionary with loaded modules
        """
        return self.modules
    
    def get_active_modules(self):
        """
        Get active modules
        
        Returns:
        - Dictionary with active modules and their activation levels
        """
        return self.module_activator.get_active_modules()
    
    def get_priority_matrix(self):
        """
        Get the priority matrix
        
        Returns:
        - Dictionary with module priorities
        """
        return self.priority_matrix
    
    def get_last_decision(self):
        """
        Get the last decision
        
        Returns:
        - Dictionary with last decision information
        """
        return self.last_decision
    
    def get_status(self):
        """
        Get the OversoulDirector status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "modules": list(self.modules.keys()),
            "active_modules": self.module_activator.get_active_modules(),
            "priority_matrix": self.priority_matrix,
            "last_market_state": self.last_market_state,
            "last_consensus": self.last_consensus,
            "last_decision": self.last_decision
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="OversoulDirector for QMP Overrider")
    parser.add_argument("--mode", choices=["full", "lite", "backtest"], default="full",
                        help="Initialization mode")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy the system")
    parser.add_argument("--live", action="store_true",
                        help="Run in live mode")
    
    args = parser.parse_args()
    
    oversoul = OversoulDirector()
    
    if not oversoul.initialize(args.mode):
        print("Failed to initialize OversoulDirector.")
        return 1
    
    if args.deploy:
        print("Deploying OversoulDirector...")
        
        if args.live:
            print("Running in live mode...")
        else:
            print("Running in simulation mode...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
