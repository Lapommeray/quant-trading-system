"""
QMP Overrider Complete - Main Entry Point

This is the main entry point for the QMP Overrider Complete trading system.
It initializes all components and starts the system in the specified mode.
"""

import os
import sys
import argparse
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.OversoulDirector.main import OversoulDirector
from Validation.TruthChecker.signal_triangulation import TruthValidator
from Validation.RitualLock.solar_aligner import CosmicSynchronizer
from Optimization.AgentLab.darwinian_ga import StrategyEvolver
from Consciousness.NLPExtractor.decision_translator import DecisionExplainer
from Consciousness.MetaMonitor.anomaly_reflector import AnomalyReflector
from Integrations.QuantConnect.qc_bridge import QCBridge
from Administration.config_manager import ConfigManager
from Administration.dependency_check import DependencyChecker
from Administration.audit_logger import AuditLogger

class QMPOverriderComplete:
    """
    QMP Overrider Complete
    
    Main class for the QMP Overrider Complete trading system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the QMP Overrider Complete system
        
        Parameters:
        - config_path: Path to configuration file (optional)
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        self.logger = AuditLogger(config_path)
        self.logger.initialize()
        
        self.dependency_checker = DependencyChecker(config_path)
        
        self.oversoul = None
        self.truth_validator = None
        self.cosmic_synchronizer = None
        self.strategy_evolver = None
        self.decision_explainer = None
        self.anomaly_reflector = None
        self.qc_bridge = None
        
        self.initialized = False
        self.running = False
        self.start_time = None
        self.mode = None
    
    def initialize(self, mode="full"):
        """
        Initialize the QMP Overrider Complete system
        
        Parameters:
        - mode: System mode (full, lite, backtest)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        self.mode = mode
        
        self.logger.log_system("Checking dependencies...")
        
        check_results = self.dependency_checker.check_dependencies()
        
        if not check_results["all_required_satisfied"]:
            self.logger.log_error(
                f"Missing required dependencies: {check_results['missing_packages']}",
                "ERROR"
            )
            return False
        
        self.logger.log_system("Initializing components...")
        
        try:
            self.oversoul = OversoulDirector()
            
            self.truth_validator = TruthValidator()
            
            self.cosmic_synchronizer = CosmicSynchronizer()
            
            self.strategy_evolver = StrategyEvolver()
            
            self.decision_explainer = DecisionExplainer()
            
            self.anomaly_reflector = AnomalyReflector()
            
            self.qc_bridge = QCBridge()
            
            self.oversoul.set_truth_validator(self.truth_validator)
            self.oversoul.set_cosmic_synchronizer(self.cosmic_synchronizer)
            self.oversoul.set_strategy_evolver(self.strategy_evolver)
            self.oversoul.set_decision_explainer(self.decision_explainer)
            self.oversoul.set_anomaly_reflector(self.anomaly_reflector)
            
            self.initialized = True
            
            self.logger.log_system("System initialized successfully")
            
            return True
        except Exception as e:
            self.logger.log_error(
                f"Error initializing system: {e}",
                "ERROR",
                sys.exc_info()
            )
            return False
    
    def start(self):
        """
        Start the QMP Overrider Complete system
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
        
        if self.running:
            return True
        
        self.start_time = datetime.now()
        
        self.logger.log_system(f"Starting system in {self.mode} mode...")
        
        try:
            if self.mode == "full":
                self.oversoul.start()
                self.qc_bridge.start()
            elif self.mode == "lite":
                self.oversoul.start(lite_mode=True)
                self.qc_bridge.start()
            elif self.mode == "backtest":
                self.qc_bridge.start_backtest()
            else:
                self.logger.log_error(f"Invalid mode: {self.mode}", "ERROR")
                return False
            
            self.running = True
            
            self.logger.log_system(f"System started in {self.mode} mode")
            
            return True
        except Exception as e:
            self.logger.log_error(
                f"Error starting system: {e}",
                "ERROR",
                sys.exc_info()
            )
            return False
    
    def stop(self):
        """
        Stop the QMP Overrider Complete system
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.running:
            return True
        
        self.logger.log_system("Stopping system...")
        
        try:
            if self.oversoul:
                self.oversoul.stop()
            
            if self.qc_bridge:
                self.qc_bridge.stop()
            
            self.running = False
            
            self.logger.log_system("System stopped")
            
            return True
        except Exception as e:
            self.logger.log_error(
                f"Error stopping system: {e}",
                "ERROR",
                sys.exc_info()
            )
            return False
    
    def get_status(self):
        """
        Get system status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "running": self.running,
            "mode": self.mode,
            "start_time": self.start_time,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else None,
            "oversoul": self.oversoul.get_status() if self.oversoul else None,
            "truth_validator": self.truth_validator.get_status() if self.truth_validator else None,
            "cosmic_synchronizer": self.cosmic_synchronizer.get_status() if self.cosmic_synchronizer else None,
            "strategy_evolver": self.strategy_evolver.get_status() if self.strategy_evolver else None,
            "anomaly_reflector": self.anomaly_reflector.get_status() if self.anomaly_reflector else None,
            "qc_bridge": self.qc_bridge.get_status() if self.qc_bridge else None
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="QMP Overrider Complete")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lite", "backtest"], help="System mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--deploy", action="store_true", help="Deploy system")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--status", action="store_true", help="Get system status")
    parser.add_argument("--stop", action="store_true", help="Stop system")
    
    args = parser.parse_args()
    
    system = QMPOverriderComplete(args.config)
    
    if args.status:
        status = system.get_status()
        print(json.dumps(status, indent=4, default=str))
    elif args.stop:
        if system.stop():
            print("System stopped successfully")
        else:
            print("Error stopping system")
            sys.exit(1)
    else:
        if not system.initialize(args.mode):
            print("Error initializing system")
            sys.exit(1)
        
        if not system.start():
            print("Error starting system")
            sys.exit(1)
        
        print(f"System started in {args.mode} mode")
        
        if args.deploy and args.live:
            print("System deployed in live mode")
        elif args.deploy:
            print("System deployed")

if __name__ == "__main__":
    main()
