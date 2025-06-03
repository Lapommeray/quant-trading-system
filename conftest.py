import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tests.mocks.quantconnect_mock import MockQCAlgorithm, MockQMPAIAgent, quantconnect_mock
except ImportError:
    MockQCAlgorithm = Mock
    MockQMPAIAgent = Mock
    quantconnect_mock = Mock()

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
    return pd.Series(prices, index=dates)

@pytest.fixture
def sample_market_data():
    """Generate comprehensive market data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    
    data = {
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.001),
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.randint(1000, 10000, len(dates))
    }
    
    data['close'] = data['open'] + np.random.randn(len(dates)) * 0.01
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(len(dates)) * 0.005)
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(len(dates)) * 0.005)
    
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_trade_data():
    """Generate sample trade data for testing order flow"""
    np.random.seed(42)
    n_trades = 1000
    
    data = {
        'price': 100 + np.random.randn(n_trades) * 0.1,
        'quantity': np.random.randint(1, 1000, n_trades),
        'side': np.random.choice([1, -1], n_trades)  # 1 for buy, -1 for sell
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_system_config():
    """Mock system configuration for testing"""
    return {
        'system_initialized': True,
        'never_loss_active': True,
        'protection_layers': 9,
        'ultra_modules_loaded': 12,
        'accuracy_multiplier': 2.5,
        'recent_win_rate': 1.0
    }

@pytest.fixture
def mock_signal():
    """Mock trading signal for testing"""
    return {
        'direction': 'BUY',
        'confidence': 0.85,
        'timestamp': datetime.now(),
        'never_loss_protected': True,
        'layers_approved': 7,
        'asset': 'BTCUSD'
    }

@pytest.fixture(scope="session")
def quantum_libraries_available():
    """Check if quantum computing libraries are available"""
    try:
        import qiskit
        import geomstats
        import gtda
        return True
    except ImportError:
        return False

@pytest.fixture
def institutional_config():
    """Configuration for institutional trading tests"""
    return {
        'risk_limits': {
            'max_position_size': 1000000,
            'max_daily_loss': 50000,
            'max_leverage': 10
        },
        'compliance': {
            'enable_reporting': True,
            'audit_trail': True,
            'position_limits': True
        }
    }

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "quantum: mark test as requiring quantum libraries"
    )
    config.addinivalue_line(
        "markers", "institutional: mark test as institutional feature"
    )
    config.addinivalue_line(
        "markers", "consciousness: mark test as consciousness-related"
    )
    config.addinivalue_line(
        "markers", "sacred_geometry: mark test as sacred geometry related"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance related"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        if "quantum" in item.name.lower():
            item.add_marker(pytest.mark.quantum)
        
        if "institutional" in item.name.lower():
            item.add_marker(pytest.mark.institutional)

@pytest.fixture
def mock_quantconnect(monkeypatch):
    """Mock QuantConnect imports for testing."""
    try:
        monkeypatch.setattr('quantconnect.QCAlgorithm', MockQCAlgorithm)
        monkeypatch.setattr('quantconnect.QMPAIAgent', MockQMPAIAgent)
    except Exception:
        pass
    return quantconnect_mock

@pytest.fixture
def mock_algorithm():
    """Provide mock algorithm instance for tests."""
    return MockQCAlgorithm()

@pytest.fixture
def mock_qmp_ai_agent():
    """Provide mock QMP AI agent for tests."""
    return MockQMPAIAgent()
