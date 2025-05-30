"""
Test script for the enhanced indicator with advanced components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the EnhancedIndicator
try:
    from Deco_30.core.enhanced_indicator import EnhancedIndicator
    ENHANCED_INDICATOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing EnhancedIndicator: {e}")
    ENHANCED_INDICATOR_AVAILABLE = False

def create_test_data(periods=200):
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    np.random.seed(42)
    prices = [100.0]
    for _ in range(periods-1):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))
    
    df = pd.DataFrame({
        'open': prices,
        'close': prices,
        'volume': np.random.normal(1000000, 200000, periods)
    }, index=dates)
    
    for i in range(len(df)):
        df.loc[df.index[i], 'high'] = df.loc[df.index[i], 'open'] * (1 + np.random.uniform(0, 0.02))
        df.loc[df.index[i], 'low'] = df.loc[df.index[i], 'open'] * (1 - np.random.uniform(0, 0.02))
    
    return df

def test_base_indicator():
    """Test the base indicator functionality"""
    print("Testing base indicator functionality...")
    
    indicator = EnhancedIndicator()
    df = create_test_data()
    
    indicator.advanced_indicators_enabled = False
    
    signal = indicator.get_signal("SPY", df)
    
    assert "signal" in signal, "Signal should contain 'signal' key"
    assert "confidence" in signal, "Signal should contain 'confidence' key"
    assert "fed_bias" in signal, "Signal should contain 'fed_bias' key"
    assert "dna_pattern" in signal, "Signal should contain 'dna_pattern' key"
    assert "liquidity_direction" in signal, "Signal should contain 'liquidity_direction' key"
    
    print("✓ Base indicator functionality preserved")
    return True

def test_advanced_indicators():
    """Test the advanced indicators functionality"""
    print("Testing advanced indicators functionality...")
    
    indicator = EnhancedIndicator()
    df = create_test_data()
    
    if not indicator.advanced_indicators_enabled:
        print("⚠ Advanced indicators not available, skipping test")
        return True
    
    signal = indicator.get_signal("SPY", df)
    
    assert "volatility" in signal, "Signal should contain 'volatility' key"
    assert "ml_rsi_prediction" in signal, "Signal should contain 'ml_rsi_prediction' key"
    assert "order_flow_imbalance" in signal, "Signal should contain 'order_flow_imbalance' key"
    assert "market_regime" in signal, "Signal should contain 'market_regime' key"
    
    print("✓ Advanced indicators functionality working")
    return True

def test_performance_metrics():
    """Test the performance metrics calculation"""
    print("Testing performance metrics calculation...")
    
    indicator = EnhancedIndicator()
    
    metrics = indicator.get_combined_performance_metrics()
    
    assert "total_win_rate_boost" in metrics, "Metrics should contain 'total_win_rate_boost'"
    assert "total_drawdown_reduction" in metrics, "Metrics should contain 'total_drawdown_reduction'"
    assert "accuracy_improvement_percentage" in metrics, "Metrics should contain 'accuracy_improvement_percentage'"
    assert "accuracy_multiplier" in metrics, "Metrics should contain 'accuracy_multiplier'"
    
    if indicator.advanced_indicators_enabled:
        assert metrics["accuracy_multiplier"] > 1.0, "Advanced indicators should improve accuracy"
        print(f"✓ Advanced indicators improve accuracy by {metrics['accuracy_improvement_percentage']:.2f}%")
    else:
        print("⚠ Advanced indicators not available, skipping accuracy improvement check")
    
    print("✓ Performance metrics calculation working")
    return True

def test_error_handling():
    """Test error handling for missing data"""
    print("Testing error handling...")
    
    indicator = EnhancedIndicator()
    
    signal = indicator.get_signal("SPY", None)
    assert "signal" in signal, "Signal should be generated even with None data"
    
    empty_df = pd.DataFrame()
    signal = indicator.get_signal("SPY", empty_df)
    assert "signal" in signal, "Signal should be generated even with empty DataFrame"
    
    invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})
    signal = indicator.get_signal("SPY", invalid_df)
    assert "signal" in signal, "Signal should be generated even with invalid column names"
    
    print("✓ Error handling working correctly")
    return True

def main():
    """Run all tests"""
    print("Testing Enhanced Indicator with Advanced Components...")
    
    if not ENHANCED_INDICATOR_AVAILABLE:
        print("\n⚠ EnhancedIndicator could not be imported. Skipping tests.")
        print("Please ensure all dependencies are installed and the import path is correct.")
        return 1
    
    try:
        base_ok = test_base_indicator()
        advanced_ok = test_advanced_indicators()
        metrics_ok = test_performance_metrics()
        error_ok = test_error_handling()
        
        if base_ok and advanced_ok and metrics_ok and error_ok:
            print("\n🎉 All tests passed! Enhanced indicator is ready to use.")
            
            indicator = EnhancedIndicator()
            metrics = indicator.get_combined_performance_metrics()
            
            if indicator.advanced_indicators_enabled:
                print(f"\nAccuracy Improvement: {metrics['accuracy_improvement_percentage']:.2f}%")
                print(f"Accuracy Multiplier: {metrics['accuracy_multiplier']:.2f}x")
                
                if metrics['accuracy_multiplier'] >= 2.0:
                    print("\n✅ 200% accuracy achieved!")
                else:
                    print(f"\n⚠ Current accuracy improvement: {metrics['accuracy_multiplier'] * 100:.2f}%")
                    print("   Target: 200%")
            
            return 0
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
            return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
