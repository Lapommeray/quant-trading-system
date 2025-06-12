#!/usr/bin/env python3
"""
Basic test to verify module structure without dependencies
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_structure():
    """Test basic module structure"""
    print("Testing basic module structure...")
    
    modules_to_check = [
        "core/qol_engine.py",
        "signals/veve_triggers.py", 
        "signals/legba_crossroads.py",
        "quant/entropy_shield.py",
        "quant/liquidity_mirror.py",
        "advanced_modules/dna_breath.py",
        "advanced_modules/dna_overlord.py",
        "advanced_modules/spectral_signal_fusion.py",
        "advanced_modules/quantum_tremor_scanner.py",
        "advanced_modules/time_fractal_fft.py"
    ]
    
    success_count = 0
    for module_path in modules_to_check:
        if os.path.exists(module_path):
            print(f"✓ Found {module_path}")
            success_count += 1
        else:
            print(f"✗ Missing {module_path}")
    
    print(f"\nStructure test results: {success_count}/{len(modules_to_check)} modules found")
    
    if success_count == len(modules_to_check):
        print("🎉 All module files exist!")
        return 0
    else:
        print("❌ Some module files are missing")
        return 1

if __name__ == "__main__":
    sys.exit(test_basic_structure())
