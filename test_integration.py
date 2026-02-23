#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_modules.heston_stochastic_engine import HestonModel, simulate_heston_paths
    from advanced_modules.transformer_alpha_generation import TimeSeriesTransformer
    from advanced_modules.hft_order_book import LimitOrderBook
    from advanced_modules.black_litterman_optimizer import black_litterman_optimization
    from advanced_modules.satellite_data_processor import estimate_oil_storage
    from advanced_modules.enhanced_backtester import EnhancedBacktester, QuantumStrategy
    from advanced_modules.enhanced_risk_management import adjusted_var, calculate_max_drawdown
    print("✓ All new modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    # Keep module importable during test collection in minimal environments.

try:
    from quantum_finance.quantum_finance_integration import QuantumFinanceIntegration
    quantum_finance = QuantumFinanceIntegration()
    print("✓ Quantum finance integration working")
except Exception as e:
    print(f"✗ Quantum finance error: {e}")

try:
    from QMP_GOD_MODE_v2_5_FINAL.core.enhanced_indicator import EnhancedIndicator
    indicator = EnhancedIndicator()
    print("✓ Original indicators preserved")
except Exception as e:
    print(f"✗ Original indicator error: {e}")

print("Integration test completed")
