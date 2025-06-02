"""
Test script for QuantConnect integration
"""

import sys
import os
sys.path.append('.')

def test_quantconnect_integration():
    """Test QuantConnect integration and strategies"""
    try:
        print("🔧 Testing QuantConnect integration...")
        
        from quantconnect_integration import UltimateNeverLossStrategy
        strategy = UltimateNeverLossStrategy()
        
        print("✅ Strategy imported successfully")
        
        strategy.Initialize()
        print("✅ Strategy initialized successfully")
        
        assert hasattr(strategy, 'entropy_scanner'), "EntropyScanner not initialized"
        assert hasattr(strategy, 'algo_fingerprinter'), "AlgoFingerprinter not initialized"
        assert hasattr(strategy, 'lstm_predictor'), "LSTMLiquidityPredictor not initialized"
        assert hasattr(strategy, 'ai_consensus'), "AI Consensus not initialized"
        
        print("✅ All strategy components initialized")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        test_market_data = {
            'symbol': 'BTCUSD',
            'prices': [100 + i + np.random.randn() for i in range(150)],
            'volumes': [1000 + np.random.randint(0, 500) for _ in range(150)],
            'returns': [np.random.randn() * 0.01 for _ in range(149)],
            'timestamps': [datetime.now() - timedelta(minutes=i) for i in range(150, 0, -1)],
            'portfolio_value': 100000,
            'positions': {'BTCUSD': 0}
        }
        
        entropy_result = strategy.entropy_scanner.analyze(test_market_data, 'BTCUSD')
        assert 'signal' in entropy_result, "EntropyScanner missing signal"
        assert 'confidence' in entropy_result, "EntropyScanner missing confidence"
        print(f"✅ EntropyScanner: {entropy_result['signal']} (confidence: {entropy_result['confidence']:.3f})")
        
        algo_result = strategy.algo_fingerprinter.analyze(test_market_data, 'BTCUSD')
        assert 'signal' in algo_result, "AlgoFingerprinter missing signal"
        assert 'confidence' in algo_result, "AlgoFingerprinter missing confidence"
        print(f"✅ AlgoFingerprinter: {algo_result['signal']} (confidence: {algo_result['confidence']:.3f})")
        
        lstm_result = strategy.lstm_predictor.analyze(test_market_data, 'BTCUSD')
        assert 'signal' in lstm_result, "LSTMLiquidityPredictor missing signal"
        assert 'confidence' in lstm_result, "LSTMLiquidityPredictor missing confidence"
        print(f"✅ LSTMLiquidityPredictor: {lstm_result['signal']} (confidence: {lstm_result['confidence']:.3f})")
        
        print("🚀 QuantConnect integration test passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ QuantConnect integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quantconnect_integration()
    sys.exit(0 if success else 1)
