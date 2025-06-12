#!/usr/bin/env python3
"""
Test module imports to verify all restored modules work correctly
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_module_import(module_path, class_name):
    """Test importing a specific module and class"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ Successfully imported {class_name} from {module_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {class_name} from {module_path}: {e}")
        traceback.print_exc()
        return False

def main():
    """Test all restored modules"""
    print("Testing module imports for restored Sacred-Quant modules...")
    print("=" * 60)
    
    modules_to_test = [
        ("core.qol_engine", "QOLEngine"),
        ("signals.veve_triggers", "VeveTriggers"),
        ("signals.legba_crossroads", "LegbaCrossroads"),
        ("quant.entropy_shield", "EntropyShield"),
        ("quant.liquidity_mirror", "LiquidityMirror"),
        ("advanced_modules.dna_breath", "DNABreath"),
        ("advanced_modules.dna_overlord", "DNAOverlord"),
        ("advanced_modules.spectral_signal_fusion", "SpectralSignalFusion"),
        ("advanced_modules.quantum_tremor_scanner", "QuantumTremorScanner"),
        ("advanced_modules.time_fractal_fft", "TimeFractalFFT"),
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_path, class_name in modules_to_test:
        if test_module_import(module_path, class_name):
            success_count += 1
    
    print("=" * 60)
    print(f"Import test results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 All modules imported successfully!")
        return 0
    else:
        print("❌ Some modules failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
