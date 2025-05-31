"""
Debug script to test failing modules individually
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from advanced_modules.temporal_stability.quantum_clock_synchronizer import QuantumClockSynchronizer
from advanced_modules.ai_only_trades.ai_only_pattern_detector import AIOnlyPatternDetector

def test_quantum_clock():
    print("Testing QuantumClockSynchronizer...")
    try:
        module = QuantumClockSynchronizer()
        if module.initialize():
            print("  ✓ Initialization successful")
            
            market_data = {
                "prices": [100 + i * 0.1 for i in range(100)],
                "timestamps": list(range(100))
            }
            
            analysis = module.analyze(market_data)
            if "error" in analysis:
                print(f"  ✗ Analysis error: {analysis['error']}")
                return False
            else:
                print("  ✓ Analysis successful")
                
            signal = module.get_signal(market_data)
            if "error" in signal:
                print(f"  ✗ Signal error: {signal['error']}")
                return False
            else:
                print("  ✓ Signal generation successful")
                return True
        else:
            print("  ✗ Initialization failed")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def test_ai_pattern_detector():
    print("Testing AIOnlyPatternDetector...")
    try:
        module = AIOnlyPatternDetector()
        if module.initialize():
            print("  ✓ Initialization successful")
            
            market_data = {
                "prices": [100 + i * 0.1 for i in range(100)],
                "volumes": [1000 + i * 10 for i in range(100)]
            }
            
            analysis = module.analyze(market_data)
            if "error" in analysis:
                print(f"  ✗ Analysis error: {analysis['error']}")
                return False
            else:
                print("  ✓ Analysis successful")
                
            signal = module.get_signal(market_data)
            if "error" in signal:
                print(f"  ✗ Signal error: {signal['error']}")
                return False
            else:
                print("  ✓ Signal generation successful")
                return True
        else:
            print("  ✗ Initialization failed")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

if __name__ == "__main__":
    print("Debugging failing modules...")
    
    clock_result = test_quantum_clock()
    pattern_result = test_ai_pattern_detector()
    
    print(f"\nResults:")
    print(f"QuantumClockSynchronizer: {'PASS' if clock_result else 'FAIL'}")
    print(f"AIOnlyPatternDetector: {'PASS' if pattern_result else 'FAIL'}")
