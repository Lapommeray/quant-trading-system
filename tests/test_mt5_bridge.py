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


if __name__ == "__main__":
    unittest.main()
