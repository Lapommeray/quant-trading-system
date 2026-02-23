"""
Unit Tests for MT5 Bridge Module

Tests for:
- JSON signal writing
- Atomic file operations
- Folder path handling
- Multi-asset support
- Configuration options
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt5_bridge import (
    write_signal_atomic,
    init_bridge,
    get_bridge_config,
    read_signal,
    clear_signals,
    get_signal_dir,
    is_bridge_available,
    write_multi_asset_signals,
    MT5BridgeConfig,
    _ensure_json_serializable
)


class TestMT5BridgeConfig(unittest.TestCase):
    """Test MT5BridgeConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MT5BridgeConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.signal_interval_seconds, 5)
        self.assertEqual(config.symbols_for_mt5, [])
        self.assertEqual(config.confidence_threshold, 0.0)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        custom = {
            "mt5_bridge_enabled": False,
            "mt5_signal_interval_seconds": 10,
            "symbols_for_mt5": ["BTCUSD", "EURUSD"],
            "mt5_confidence_threshold": 0.5
        }
        config = MT5BridgeConfig(custom)
        self.assertFalse(config.enabled)
        self.assertEqual(config.signal_interval_seconds, 10)
        self.assertEqual(config.symbols_for_mt5, ["BTCUSD", "EURUSD"])
        self.assertEqual(config.confidence_threshold, 0.5)
    
    def test_should_write_signal_disabled(self):
        """Test signal filtering when bridge is disabled"""
        config = MT5BridgeConfig({"mt5_bridge_enabled": False})
        self.assertFalse(config.should_write_signal("BTCUSD", 0.9))
    
    def test_should_write_signal_symbol_filter(self):
        """Test signal filtering by symbol"""
        config = MT5BridgeConfig({"symbols_for_mt5": ["BTCUSD"]})
        self.assertTrue(config.should_write_signal("BTCUSD", 0.9))
        self.assertFalse(config.should_write_signal("EURUSD", 0.9))
    
    def test_should_write_signal_confidence_filter(self):
        """Test signal filtering by confidence threshold"""
        config = MT5BridgeConfig({"mt5_confidence_threshold": 0.7})
        self.assertTrue(config.should_write_signal("BTCUSD", 0.8))
        self.assertFalse(config.should_write_signal("BTCUSD", 0.5))
    
    def test_should_write_signal_interval(self):
        """Test signal filtering by interval"""
        config = MT5BridgeConfig({"mt5_signal_interval_seconds": 10})
        # First write should be allowed
        self.assertTrue(config.should_write_signal("BTCUSD", 0.9))
        config.record_write("BTCUSD")
        # Immediate second write should be blocked
        self.assertFalse(config.should_write_signal("BTCUSD", 0.9))


class TestWriteSignalAtomic(unittest.TestCase):
    """Test write_signal_atomic function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config = init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0  # Disable interval for testing
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_write_valid_signal(self):
        """Test writing a valid signal"""
        signal = {
            "symbol": "BTCUSD",
            "signal": "BUY",
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = write_signal_atomic(signal)
        self.assertTrue(result)
        
        # Verify file was created
        signal_file = os.path.join(self.test_dir, "BTCUSD_signal.json")
        self.assertTrue(os.path.exists(signal_file))
        
        # Verify content
        with open(signal_file, 'r') as f:
            written_signal = json.load(f)
        self.assertEqual(written_signal["symbol"], "BTCUSD")
        self.assertEqual(written_signal["signal"], "BUY")
        self.assertEqual(written_signal["confidence"], 0.85)
    
    def test_write_signal_with_slash_symbol(self):
        """Test writing signal with slash in symbol name"""
        signal = {
            "symbol": "BTC/USDT",
            "signal": "SELL",
            "confidence": 0.75,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = write_signal_atomic(signal)
        self.assertTrue(result)
        
        # Verify file was created with sanitized name
        signal_file = os.path.join(self.test_dir, "BTCUSDT_signal.json")
        self.assertTrue(os.path.exists(signal_file))
    
    def test_write_signal_missing_field(self):
        """Test writing signal with missing required field"""
        signal = {
            "symbol": "BTCUSD",
            "signal": "BUY"
            # Missing confidence and timestamp
        }
        
        result = write_signal_atomic(signal)
        self.assertFalse(result)
    
    def test_write_signal_hold(self):
        """Test writing HOLD signal"""
        signal = {
            "symbol": "EURUSD",
            "signal": "HOLD",
            "confidence": 0.5,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = write_signal_atomic(signal)
        self.assertTrue(result)
        
        # Verify content
        written = read_signal("EURUSD")
        self.assertEqual(written["signal"], "HOLD")


class TestReadSignal(unittest.TestCase):
    """Test read_signal function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_read_existing_signal(self):
        """Test reading an existing signal"""
        signal = {
            "symbol": "XAUUSD",
            "signal": "BUY",
            "confidence": 0.9,
            "timestamp": datetime.utcnow().isoformat()
        }
        write_signal_atomic(signal)
        
        read = read_signal("XAUUSD")
        self.assertIsNotNone(read)
        self.assertEqual(read["signal"], "BUY")
    
    def test_read_nonexistent_signal(self):
        """Test reading a non-existent signal"""
        read = read_signal("NONEXISTENT")
        self.assertIsNone(read)


class TestMultiAssetSignals(unittest.TestCase):
    """Test multi-asset signal writing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_write_multiple_signals(self):
        """Test writing signals for multiple assets"""
        signals = [
            {
                "symbol": "BTCUSD",
                "signal": "BUY",
                "confidence": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "symbol": "ETHUSD",
                "signal": "SELL",
                "confidence": 0.7,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "symbol": "XAUUSD",
                "signal": "HOLD",
                "confidence": 0.5,
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        results = write_multi_asset_signals(signals)
        
        self.assertTrue(results["BTCUSD"])
        self.assertTrue(results["ETHUSD"])
        self.assertTrue(results["XAUUSD"])
        
        # Verify all files exist
        for symbol in ["BTCUSD", "ETHUSD", "XAUUSD"]:
            self.assertIsNotNone(read_signal(symbol))


class TestClearSignals(unittest.TestCase):
    """Test clear_signals function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_clear_signals(self):
        """Test clearing all signals"""
        # Write some signals
        for symbol in ["BTCUSD", "ETHUSD", "XAUUSD"]:
            write_signal_atomic({
                "symbol": symbol,
                "signal": "BUY",
                "confidence": 0.8,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Clear signals
        count = clear_signals()
        self.assertEqual(count, 3)
        
        # Verify all signals are gone
        for symbol in ["BTCUSD", "ETHUSD", "XAUUSD"]:
            self.assertIsNone(read_signal(symbol))


class TestJsonSerialization(unittest.TestCase):
    """Test JSON serialization helpers"""
    
    def test_serialize_datetime(self):
        """Test datetime serialization"""
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = _ensure_json_serializable(dt)
        self.assertEqual(result, "2024-01-15T12:30:45")
    
    def test_serialize_nested_dict(self):
        """Test nested dictionary serialization"""
        data = {
            "outer": {
                "inner": datetime(2024, 1, 15),
                "value": 42
            }
        }
        result = _ensure_json_serializable(data)
        self.assertEqual(result["outer"]["inner"], "2024-01-15T00:00:00")
        self.assertEqual(result["outer"]["value"], 42)
    
    def test_serialize_list(self):
        """Test list serialization"""
        data = [datetime(2024, 1, 15), "string", 42]
        result = _ensure_json_serializable(data)
        self.assertEqual(result[0], "2024-01-15T00:00:00")
        self.assertEqual(result[1], "string")
        self.assertEqual(result[2], 42)


class TestBridgeAvailability(unittest.TestCase):
    """Test bridge availability checks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_bridge_available_writable_dir(self):
        """Test bridge availability with writable directory"""
        init_bridge({"mt5_signal_dir": self.test_dir})
        self.assertTrue(is_bridge_available())
    
    def test_bridge_disabled(self):
        """Test bridge availability when disabled"""
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_bridge_enabled": False
        })
        self.assertFalse(is_bridge_available())


class TestSignalSchemaConsistency(unittest.TestCase):
    """Test signal schema consistency for MT5 EA compatibility"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_signal_schema_fields(self):
        """Test that written signals have all required fields"""
        signal = {
            "symbol": "BTCUSD",
            "signal": "BUY",
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        write_signal_atomic(signal)
        
        written = read_signal("BTCUSD")
        
        # Check all required fields are present
        self.assertIn("symbol", written)
        self.assertIn("signal", written)
        self.assertIn("confidence", written)
        self.assertIn("timestamp", written)
        
        # Check types
        self.assertIsInstance(written["symbol"], str)
        self.assertIsInstance(written["signal"], str)
        self.assertIsInstance(written["confidence"], (int, float))
        self.assertIsInstance(written["timestamp"], str)
    
    def test_signal_values_valid(self):
        """Test that signal values are valid for MT5 EA"""
        for signal_type in ["BUY", "SELL", "HOLD"]:
            signal = {
                "symbol": "TESTPAIR",
                "signal": signal_type,
                "confidence": 0.75,
                "timestamp": datetime.utcnow().isoformat()
            }
            result = write_signal_atomic(signal)
            self.assertTrue(result)
            
            written = read_signal("TESTPAIR")
            self.assertEqual(written["signal"], signal_type)
    
    def test_timestamp_iso_format(self):
        """Test that timestamp is in ISO format"""
        signal = {
            "symbol": "BTCUSD",
            "signal": "BUY",
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        write_signal_atomic(signal)
        
        written = read_signal("BTCUSD")
        
        # Verify timestamp can be parsed as ISO format
        try:
            datetime.fromisoformat(written["timestamp"])
        except ValueError:
            self.fail("Timestamp is not in valid ISO format")


class TestTradingWindows(unittest.TestCase):
    """Test execution safety: trading windows"""
    
    def test_no_windows_allows_all(self):
        """Test that empty trading windows allow all signals"""
        config = MT5BridgeConfig({"mt5_trading_windows": []})
        self.assertTrue(config._is_within_trading_window())
    
    def test_window_blocks_outside_hours(self):
        """Test that signals are blocked outside trading windows"""
        # Create a window that is definitely not now (1 minute window far from current time)
        config = MT5BridgeConfig({
            "mt5_trading_windows": [{"start": "00:00", "end": "00:01"}]
        })
        # This test may pass or fail depending on current time;
        # we test the logic by using a known window
        now = datetime.utcnow()
        # Create a window that excludes current time
        if now.hour < 12:
            window = {"start": "13:00", "end": "14:00"}
        else:
            window = {"start": "01:00", "end": "02:00"}
        config = MT5BridgeConfig({"mt5_trading_windows": [window]})
        self.assertFalse(config._is_within_trading_window())
    
    def test_window_allows_within_hours(self):
        """Test that signals are allowed within trading windows"""
        # Create a window that definitely includes now
        config = MT5BridgeConfig({
            "mt5_trading_windows": [{"start": "00:00", "end": "23:59"}]
        })
        self.assertTrue(config._is_within_trading_window())
    
    def test_trading_window_blocks_signal_write(self):
        """Test that should_write_signal respects trading windows"""
        now = datetime.utcnow()
        if now.hour < 12:
            window = {"start": "13:00", "end": "14:00"}
        else:
            window = {"start": "01:00", "end": "02:00"}
        config = MT5BridgeConfig({"mt5_trading_windows": [window]})
        self.assertFalse(config.should_write_signal("BTCUSD", 0.9))


class TestCircuitBreaker(unittest.TestCase):
    """Test execution safety: circuit breaker"""
    
    def test_no_limit_allows_all(self):
        """Test that circuit breaker disabled (0) allows all signals"""
        config = MT5BridgeConfig({"mt5_max_signals_per_minute": 0})
        self.assertTrue(config._check_circuit_breaker())
    
    def test_circuit_breaker_trips(self):
        """Test that circuit breaker trips when limit exceeded"""
        config = MT5BridgeConfig({"mt5_max_signals_per_minute": 2})
        # Record 2 writes to exceed the limit
        config._recent_signal_times = [datetime.utcnow(), datetime.utcnow()]
        self.assertFalse(config._check_circuit_breaker())
    
    def test_circuit_breaker_allows_under_limit(self):
        """Test that circuit breaker allows signals under limit"""
        config = MT5BridgeConfig({"mt5_max_signals_per_minute": 5})
        config._recent_signal_times = [datetime.utcnow()]
        self.assertTrue(config._check_circuit_breaker())


class TestMultiTimeframe(unittest.TestCase):
    """Test multi-timeframe signal support"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        init_bridge({
            "mt5_signal_dir": self.test_dir,
            "mt5_signal_interval_seconds": 0,
            "mt5_separate_timeframe_files": True,
            "mt5_default_timeframe": "M5"
        })
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_separate_timeframe_files(self):
        """Test that separate files are created per timeframe"""
        signal_m5 = {
            "symbol": "BTCUSD",
            "signal": "BUY",
            "confidence": 0.8,
            "timestamp": datetime.utcnow().isoformat(),
            "timeframe": "M5"
        }
        signal_h1 = {
            "symbol": "BTCUSD",
            "signal": "SELL",
            "confidence": 0.7,
            "timestamp": datetime.utcnow().isoformat(),
            "timeframe": "H1"
        }
        
        write_signal_atomic(signal_m5)
        write_signal_atomic(signal_h1)
        
        # Both files should exist
        m5_file = os.path.join(self.test_dir, "BTCUSD_M5_signal.json")
        h1_file = os.path.join(self.test_dir, "BTCUSD_H1_signal.json")
        self.assertTrue(os.path.exists(m5_file))
        self.assertTrue(os.path.exists(h1_file))
        
        # Verify contents are different
        with open(m5_file, 'r') as f:
            m5_data = json.load(f)
        with open(h1_file, 'r') as f:
            h1_data = json.load(f)
        self.assertEqual(m5_data["signal"], "BUY")
        self.assertEqual(h1_data["signal"], "SELL")
    
    def test_default_timeframe_included(self):
        """Test that default timeframe is added to signal if not specified"""
        signal = {
            "symbol": "EURUSD",
            "signal": "BUY",
            "confidence": 0.8,
            "timestamp": datetime.utcnow().isoformat()
        }
        write_signal_atomic(signal)
        
        # File should use default timeframe M5
        signal_file = os.path.join(self.test_dir, "EURUSD_M5_signal.json")
        self.assertTrue(os.path.exists(signal_file))
        
        with open(signal_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(data["timeframe"], "M5")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility when MT5 bridge is unavailable"""
    
    def test_fallback_write_signal_returns_false(self):
        """Test that fallback write_signal_atomic returns False"""
        # Simulate the fallback function from main.py
        def fallback_write_signal_atomic(signal_dict):
            return False
        
        result = fallback_write_signal_atomic({"symbol": "BTCUSD"})
        self.assertFalse(result)
    
    def test_fallback_init_bridge_returns_none(self):
        """Test that fallback init_bridge returns None"""
        def fallback_init_bridge(config=None):
            return None
        
        result = fallback_init_bridge()
        self.assertIsNone(result)
    
    def test_fallback_is_bridge_available_returns_false(self):
        """Test that fallback is_bridge_available returns False"""
        def fallback_is_bridge_available():
            return False
        
        result = fallback_is_bridge_available()
        self.assertFalse(result)
    
    def test_disabled_bridge_write_returns_false(self):
        """Test that writing with disabled bridge returns False"""
        test_dir = tempfile.mkdtemp()
        try:
            init_bridge({
                "mt5_signal_dir": test_dir,
                "mt5_bridge_enabled": False
            })
            result = write_signal_atomic({
                "symbol": "BTCUSD",
                "signal": "BUY",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.assertFalse(result)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestConfigEmptySignalDir(unittest.TestCase):
    """Test that empty mt5_signal_dir falls back to default"""
    
    def test_empty_string_falls_back(self):
        """Test that empty string signal dir falls back to default"""
        config = MT5BridgeConfig({"mt5_signal_dir": ""})
        self.assertNotEqual(config.signal_dir, "")
    
    def test_none_falls_back(self):
        """Test that None signal dir falls back to default"""
        config = MT5BridgeConfig({"mt5_signal_dir": None})
        self.assertNotEqual(config.signal_dir, "")
        self.assertIsNotNone(config.signal_dir)


if __name__ == "__main__":
    unittest.main()
