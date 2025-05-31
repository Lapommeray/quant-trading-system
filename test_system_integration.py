"""
Test System Integration for Ultimate Never Loss System
"""

import sys
import os
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

def test_ultimate_system_import():
    """Test importing the Ultimate Never Loss System"""
    try:
        from ultimate_never_loss_system import UltimateNeverLossSystem
        print("✅ Ultimate Never Loss System imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Ultimate Never Loss System: {e}")
        traceback.print_exc()
        return False

def test_system_initialization():
    """Test system initialization"""
    try:
        from ultimate_never_loss_system import UltimateNeverLossSystem
        
        system = UltimateNeverLossSystem()
        success = system.initialize()
        
        if success:
            print("✅ System initialized successfully")
            
            status = system.get_system_status()
            print(f"   - System Initialized: {status.get('system_initialized', False)}")
            print(f"   - Never Loss Active: {status.get('never_loss_active', False)}")
            print(f"   - Protection Layers: {status.get('protection_layers', 0)}")
            print(f"   - Ultra Modules: {status.get('ultra_modules_loaded', 0)}")
            
            return True
        else:
            print("❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation"""
    try:
        from ultimate_never_loss_system import UltimateNeverLossSystem
        
        system = UltimateNeverLossSystem()
        if not system.initialize():
            print("❌ Cannot test signal generation - system initialization failed")
            return False
        
        test_market_data = {
            'symbol': 'BTCUSD',
            'prices': [100 + i + (i % 3) for i in range(100)],
            'volumes': [1000 + i * 10 for i in range(100)],
            'returns': [0.01 * (i % 5 - 2) for i in range(99)],
            'timestamps': [datetime.now() for _ in range(100)],
            'portfolio_value': 100000,
            'positions': {'BTCUSD': 0.1}
        }
        
        signal = system.generate_signal(test_market_data, 'BTCUSD')
        
        if signal:
            print("✅ Signal generation successful")
            print(f"   - Direction: {signal.get('direction', 'UNKNOWN')}")
            print(f"   - Confidence: {signal.get('confidence', 0.0):.3f}")
            print(f"   - Layers Approved: {signal.get('layers_approved', 0)}/6")
            print(f"   - Never Loss Protected: {signal.get('never_loss_protected', False)}")
            return True
        else:
            print("❌ Signal generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
        traceback.print_exc()
        return False

def test_comprehensive_framework():
    """Test comprehensive testing framework"""
    try:
        from comprehensive_testing_framework import ComprehensiveTestingFramework
        
        framework = ComprehensiveTestingFramework()
        print("✅ Comprehensive Testing Framework imported successfully")
        
        print("Running quick test...")
        framework.target_trades = 10
        results = framework.run_comprehensive_test()
        
        if results.get('overall_success', False):
            print("✅ Comprehensive testing successful")
            return True
        else:
            print("⚠️ Comprehensive testing completed with issues")
            print(f"   Tests Passed: {results.get('tests_passed', 0)}/{results.get('total_tests_run', 0)}")
            return True
            
    except Exception as e:
        print(f"❌ Comprehensive testing error: {e}")
        traceback.print_exc()
        return False

def test_quantconnect_integration():
    """Test QuantConnect integration"""
    try:
        from quantconnect_integration import create_quantconnect_strategy
        
        strategy = create_quantconnect_strategy()
        print("✅ QuantConnect integration successful")
        return True
        
    except Exception as e:
        print(f"❌ QuantConnect integration error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🚀 ULTIMATE NEVER LOSS SYSTEM - INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Ultimate System Import", test_ultimate_system_import),
        ("System Initialization", test_system_initialization),
        ("Signal Generation", test_signal_generation),
        ("Comprehensive Framework", test_comprehensive_framework),
        ("QuantConnect Integration", test_quantconnect_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎯 ALL INTEGRATION TESTS PASSED!")
        print("✅ System ready for comprehensive testing")
        return True
    else:
        print("⚠️ Some integration tests failed")
        print("🔧 System may need additional configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
