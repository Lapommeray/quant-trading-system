"""
QMP Overrider Complete - Integration Verification

This script verifies the integration of all components in the QMP Overrider Complete system.
It runs a series of tests to ensure that all components are properly connected and functioning.
"""

import os
import sys
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

class IntegrationVerifier:
    """
    Integration Verifier for QMP Overrider Complete
    
    Verifies the integration of all components in the QMP Overrider Complete system.
    """
    
    def __init__(self):
        """Initialize the Integration Verifier"""
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
    
    def run_all_tests(self):
        """
        Run all integration tests
        
        Returns:
        - Dictionary with test results
        """
        self.test_oversoul_director()
        self.test_truth_validator()
        self.test_cosmic_synchronizer()
        self.test_strategy_evolver()
        self.test_decision_explainer()
        self.test_anomaly_reflector()
        self.test_qc_bridge()
        self.test_config_manager()
        self.test_dependency_checker()
        self.test_audit_logger()
        self.test_full_integration()
        
        self.results["summary"]["total"] = len(self.results["tests"])
        self.results["summary"]["passed"] = sum(1 for test in self.results["tests"].values() if test["status"] == "PASSED")
        self.results["summary"]["failed"] = sum(1 for test in self.results["tests"].values() if test["status"] == "FAILED")
        
        return self.results
    
    def test_oversoul_director(self):
        """Test OversoulDirector integration"""
        test_name = "OversoulDirector"
        
        try:
            oversoul = OversoulDirector()
            
            assert oversoul is not None, "OversoulDirector initialization failed"
            
            assert oversoul.priority_matrix is not None, "Priority matrix initialization failed"
            
            active_modules = oversoul._determine_active_modules({"volatility": 30})
            assert active_modules is not None, "Module activation failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_truth_validator(self):
        """Test TruthValidator integration"""
        test_name = "TruthValidator"
        
        try:
            truth_validator = TruthValidator()
            
            assert truth_validator is not None, "TruthValidator initialization failed"
            
            assert truth_validator.add_signal("phoenix", "BUY", 0.8), "Signal addition failed"
            assert truth_validator.add_signal("aurora", "BUY", 0.7), "Signal addition failed"
            assert truth_validator.add_signal("qmp", "BUY", 0.9), "Signal addition failed"
            
            resolution = truth_validator.resolve_signal()
            assert resolution is not None, "Signal resolution failed"
            assert resolution["signal"] == "BUY", "Signal resolution returned incorrect direction"
            
            assert truth_validator.validate({"direction": "BUY", "confidence": 0.8}), "Validation failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_cosmic_synchronizer(self):
        """Test CosmicSynchronizer integration"""
        test_name = "CosmicSynchronizer"
        
        try:
            cosmic_synchronizer = CosmicSynchronizer()
            
            assert cosmic_synchronizer is not None, "CosmicSynchronizer initialization failed"
            
            alignment = cosmic_synchronizer.is_aligned()
            assert alignment is not None, "Alignment check failed"
            
            ritual_data = cosmic_synchronizer.get_ritual_data()
            assert ritual_data is not None, "Ritual data retrieval failed"
            assert "moon_phase" in ritual_data, "Ritual data missing moon phase"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_strategy_evolver(self):
        """Test StrategyEvolver integration"""
        test_name = "StrategyEvolver"
        
        try:
            strategy_evolver = StrategyEvolver()
            
            assert strategy_evolver is not None, "StrategyEvolver initialization failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_decision_explainer(self):
        """Test DecisionExplainer integration"""
        test_name = "DecisionExplainer"
        
        try:
            decision_explainer = DecisionExplainer()
            
            assert decision_explainer is not None, "DecisionExplainer initialization failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_anomaly_reflector(self):
        """Test AnomalyReflector integration"""
        test_name = "AnomalyReflector"
        
        try:
            anomaly_reflector = AnomalyReflector()
            
            assert anomaly_reflector is not None, "AnomalyReflector initialization failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_qc_bridge(self):
        """Test QCBridge integration"""
        test_name = "QCBridge"
        
        try:
            qc_bridge = QCBridge()
            
            assert qc_bridge is not None, "QCBridge initialization failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_config_manager(self):
        """Test ConfigManager integration"""
        test_name = "ConfigManager"
        
        try:
            config_manager = ConfigManager()
            
            assert config_manager is not None, "ConfigManager initialization failed"
            
            config = config_manager.get_config()
            assert config is not None, "Configuration retrieval failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_dependency_checker(self):
        """Test DependencyChecker integration"""
        test_name = "DependencyChecker"
        
        try:
            dependency_checker = DependencyChecker()
            
            assert dependency_checker is not None, "DependencyChecker initialization failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_audit_logger(self):
        """Test AuditLogger integration"""
        test_name = "AuditLogger"
        
        try:
            audit_logger = AuditLogger()
            
            assert audit_logger is not None, "AuditLogger initialization failed"
            
            assert audit_logger.initialize(), "AuditLogger initialization failed"
            
            audit_logger.log_system("Integration test")
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def test_full_integration(self):
        """Test full system integration"""
        test_name = "FullIntegration"
        
        try:
            oversoul = OversoulDirector()
            truth_validator = TruthValidator()
            cosmic_synchronizer = CosmicSynchronizer()
            strategy_evolver = StrategyEvolver()
            decision_explainer = DecisionExplainer()
            anomaly_reflector = AnomalyReflector()
            
            oversoul.set_truth_validator(truth_validator)
            oversoul.set_cosmic_synchronizer(cosmic_synchronizer)
            oversoul.set_strategy_evolver(strategy_evolver)
            oversoul.set_decision_explainer(decision_explainer)
            oversoul.set_anomaly_reflector(anomaly_reflector)
            
            market_state = {
                "volatility": 30,
                "unusual_conditions": True,
                "time_sensitive": True,
                "event_risk": 0.7
            }
            
            truth_validator.add_signal("phoenix", "BUY", 0.8)
            truth_validator.add_signal("aurora", "BUY", 0.7)
            truth_validator.add_signal("qmp", "BUY", 0.9)
            
            signal = oversoul.route_signal(market_state)
            
            assert signal is not None, "Signal routing failed"
            
            self.record_test_result(test_name, "PASSED")
        except Exception as e:
            self.record_test_result(test_name, "FAILED", str(e))
    
    def record_test_result(self, test_name, status, message=None):
        """
        Record test result
        
        Parameters:
        - test_name: Test name
        - status: Test status (PASSED, FAILED)
        - message: Error message (optional)
        """
        self.results["tests"][test_name] = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def print_results(self):
        """Print test results"""
        print(f"Integration Tests: {self.results['timestamp']}")
        print(f"Total: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        print()
        
        for test_name, test_result in self.results["tests"].items():
            status_str = f"[{test_result['status']}]"
            
            if test_result["status"] == "PASSED":
                print(f"{status_str.ljust(10)} {test_name}")
            else:
                print(f"{status_str.ljust(10)} {test_name}: {test_result['message']}")
    
    def save_results(self, file_path=None):
        """
        Save test results to file
        
        Parameters:
        - file_path: Path to results file (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "integration_test_results.json"
            )
        
        try:
            with open(file_path, "w") as f:
                json.dump(self.results, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving test results: {e}")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QMP Overrider Complete Integration Verification")
    parser.add_argument("--save", action="store_true", help="Save test results to file")
    parser.add_argument("--output", type=str, help="Path to results file")
    
    args = parser.parse_args()
    
    verifier = IntegrationVerifier()
    
    verifier.run_all_tests()
    
    verifier.print_results()
    
    if args.save or args.output:
        verifier.save_results(args.output)

if __name__ == "__main__":
    main()
