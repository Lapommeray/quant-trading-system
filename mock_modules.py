"""
Mock modules for testing when actual modules are not available
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

class MockEnhancedIndicator:
    """Mock Enhanced Indicator for testing"""
    
    def __init__(self):
        self.initialized = True
    
    def get_trading_signal(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate mock trading signal"""
        prices = market_data.get('prices', [100])
        if len(prices) < 2:
            return {'signal': 'NEUTRAL', 'confidence': 0.5}
        
        recent_change = (prices[-1] - prices[-2]) / prices[-2]
        
        if recent_change > 0.01:
            return {'signal': 'BUY', 'confidence': 0.8}
        elif recent_change < -0.01:
            return {'signal': 'SELL', 'confidence': 0.8}
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0.6}

class MockOversoulDirector:
    """Mock Oversoul Director for testing"""
    
    def __init__(self):
        self.initialized = False
        self.modules = {}
    
    def initialize(self, mode: str = "full") -> bool:
        """Initialize mock oversoul director"""
        self.modules = {
            'phoenix': MockModule('phoenix'),
            'aurora': MockModule('aurora'),
            'truth': MockModule('truth'),
            'hadron_collider': MockModule('hadron_collider'),
            'quantum_entanglement': MockModule('quantum_entanglement'),
            'temporal_probability': MockModule('temporal_probability')
        }
        self.initialized = True
        return True
    
    def get_modules(self) -> Dict[str, Any]:
        """Get mock modules"""
        return self.modules

class MockModule:
    """Mock module for testing"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock signal"""
        return {
            'direction': 'BUY',
            'confidence': 0.85,
            'module': self.name
        }
    
    def predict(self, data: Any) -> float:
        """Mock prediction"""
        return 0.75

class MockAIConsensusEngine:
    """Mock AI Consensus Engine for testing"""
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.modules = {}
    
    def register_ai_module(self, name: str, module: Any, weight: float):
        """Register mock AI module"""
        self.modules[name] = {'module': module, 'weight': weight}
    
    def achieve_consensus(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate mock consensus"""
        if len(self.modules) == 0:
            return {
                'consensus_achieved': False,
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'consensus_ratio': 0.0
            }
        
        return {
            'consensus_achieved': True,
            'signal': 'BUY',
            'confidence': 0.85,
            'consensus_ratio': 0.85
        }

class MockPerformanceMetrics:
    """Mock Performance Metrics for testing"""
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.trades = []
    
    def calculate_current_accuracy(self) -> Dict[str, Any]:
        """Calculate mock accuracy metrics"""
        return {
            'never_loss_rate': 1.0,
            'accuracy_multiplier': 2.0,
            'super_high_confidence_rate': 0.95
        }
    
    def record_trade_prediction(self, symbol: str, direction: str, confidence: float, 
                              consensus: Dict[str, Any], opportunity: Dict[str, Any], 
                              reality: Dict[str, Any]):
        """Record mock trade prediction"""
        self.trades.append({
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def update_trade_outcome(self, symbol: str, timestamp: datetime, outcome: float):
        """Update mock trade outcome"""
        pass

def create_mock_system_components():
    """Create mock system components for testing"""
    return {
        'enhanced_indicator': MockEnhancedIndicator(),
        'oversoul_director': MockOversoulDirector(),
        'ai_consensus_engine': MockAIConsensusEngine(),
        'performance_metrics': MockPerformanceMetrics()
    }
