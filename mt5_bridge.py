"""
MT5 Bridge Module - Atomic Signal Writer for RayBridge EA

This module provides atomic file writing functionality to output trading signals
to the MT5 Common/Files/raybridge directory for consumption by the RayBridge EA.

The bridge ensures:
- Atomic writes to prevent partial reads by MT5
- JSON schema consistency for reliable EA parsing
- Configurable output frequency and filtering
- Backward compatibility when MT5 is unavailable
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger("MT5Bridge")

# Default MT5 Common Files path (Windows)
# On Windows: C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\raybridge
# This can be overridden via config
DEFAULT_MT5_SIGNAL_DIR = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal", "Common", "Files", "raybridge"
)

# Fallback for non-Windows systems (for testing/development)
FALLBACK_SIGNAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mt5_signals"
)

# Thread lock for atomic operations
_write_lock = threading.Lock()


class MT5BridgeConfig:
    """Configuration for MT5 Bridge"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Signal output directory
        self.signal_dir = config.get("mt5_signal_dir", self._get_default_signal_dir())
        
        # Output frequency in seconds (minimum interval between writes)
        self.signal_interval_seconds = config.get("mt5_signal_interval_seconds", 5)
        
        # Symbols to output signals for (empty list = all symbols)
        self.symbols_for_mt5 = config.get("symbols_for_mt5", [])
        
        # Minimum confidence threshold for signal output
        self.confidence_threshold = config.get("mt5_confidence_threshold", 0.0)
        
        # Enable/disable bridge
        self.enabled = config.get("mt5_bridge_enabled", True)
        
        # Last write timestamps per symbol
        self._last_write_times: Dict[str, datetime] = {}
        
    def _get_default_signal_dir(self) -> str:
        """Get the default signal directory based on platform"""
        if sys.platform == "win32" and os.path.exists(os.environ.get("APPDATA", "")):
            return DEFAULT_MT5_SIGNAL_DIR
        return FALLBACK_SIGNAL_DIR
    
    def should_write_signal(self, symbol: str, confidence: float) -> bool:
        """Check if a signal should be written based on config"""
        if not self.enabled:
            return False
            
        # Check symbol filter
        if self.symbols_for_mt5 and symbol not in self.symbols_for_mt5:
            return False
            
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
            
        # Check interval
        now = datetime.utcnow()
        last_write = self._last_write_times.get(symbol)
        if last_write:
            elapsed = (now - last_write).total_seconds()
            if elapsed < self.signal_interval_seconds:
                return False
                
        return True
    
    def record_write(self, symbol: str):
        """Record that a signal was written for a symbol"""
        self._last_write_times[symbol] = datetime.utcnow()


# Global config instance
_bridge_config: Optional[MT5BridgeConfig] = None


def init_bridge(config: Optional[Dict[str, Any]] = None) -> MT5BridgeConfig:
    """Initialize the MT5 bridge with configuration
    
    Args:
        config: Configuration dictionary with optional keys:
            - mt5_signal_dir: Directory for signal files
            - mt5_signal_interval_seconds: Minimum interval between writes
            - symbols_for_mt5: List of symbols to output (empty = all)
            - mt5_confidence_threshold: Minimum confidence for output
            - mt5_bridge_enabled: Enable/disable bridge
            
    Returns:
        MT5BridgeConfig instance
    """
    global _bridge_config
    _bridge_config = MT5BridgeConfig(config)
    
    # Ensure signal directory exists
    if _bridge_config.enabled:
        try:
            os.makedirs(_bridge_config.signal_dir, exist_ok=True)
            logger.info(f"MT5 Bridge initialized. Signal directory: {_bridge_config.signal_dir}")
        except Exception as e:
            logger.warning(f"Could not create MT5 signal directory: {e}")
            
    return _bridge_config


def get_bridge_config() -> MT5BridgeConfig:
    """Get the current bridge configuration, initializing if needed"""
    global _bridge_config
    if _bridge_config is None:
        _bridge_config = MT5BridgeConfig()
    return _bridge_config


def write_signal_atomic(signal_dict: Dict[str, Any]) -> bool:
    """Write a trading signal atomically to the MT5 bridge directory
    
    This function performs an atomic write by:
    1. Writing to a temporary file
    2. Moving the temp file to the final location
    
    This ensures MT5 EA never reads a partial file.
    
    Args:
        signal_dict: Dictionary containing signal data with required keys:
            - symbol: Trading symbol (e.g., "BTCUSD", "EURUSD")
            - signal: Signal direction ("BUY", "SELL", "HOLD")
            - confidence: Confidence level (0.0 to 1.0)
            - timestamp: ISO format timestamp
            
    Returns:
        bool: True if write succeeded, False otherwise
    """
    config = get_bridge_config()
    
    if not config.enabled:
        logger.debug("MT5 Bridge disabled, skipping signal write")
        return False
    
    # Validate required fields
    required_fields = ["symbol", "signal", "confidence", "timestamp"]
    for field in required_fields:
        if field not in signal_dict:
            logger.error(f"MT5 signal missing required field: {field}")
            return False
    
    symbol = signal_dict["symbol"]
    confidence = float(signal_dict.get("confidence", 0.0))
    
    # Check if we should write this signal
    if not config.should_write_signal(symbol, confidence):
        logger.debug(f"Skipping signal write for {symbol} (filtered by config)")
        return False
    
    # Normalize symbol for filename (remove special chars)
    safe_symbol = symbol.replace("/", "").replace("\\", "").replace(":", "")
    signal_file = os.path.join(config.signal_dir, f"{safe_symbol}_signal.json")
    
    # Ensure all values are JSON-serializable
    try:
        serializable_signal = _ensure_json_serializable(signal_dict)
    except Exception as e:
        logger.error(f"Failed to serialize signal for {symbol}: {e}")
        return False
    
    with _write_lock:
        try:
            # Ensure directory exists
            os.makedirs(config.signal_dir, exist_ok=True)
            
            # Write to temporary file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=f"{safe_symbol}_",
                dir=config.signal_dir
            )
            
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                    
                # Atomic move to final location
                shutil.move(temp_path, signal_file)
                
                # Record successful write
                config.record_write(symbol)
                
                logger.info(f"MT5 signal output: {serializable_signal}")
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
                
        except Exception as e:
            logger.error(f"Failed to write MT5 signal for {symbol}: {e}")
            return False


def _ensure_json_serializable(data: Any) -> Any:
    """Ensure all values in a dict are JSON-serializable
    
    Converts:
    - datetime objects to ISO format strings
    - numpy types to Python native types
    - Other non-serializable types to strings
    """
    if isinstance(data, dict):
        return {k: _ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_ensure_json_serializable(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    elif hasattr(data, 'item'):  # numpy scalar
        return data.item()
    elif hasattr(data, 'tolist'):  # numpy array
        return data.tolist()
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        # Convert unknown types to string
        return str(data)


def write_multi_asset_signals(signals: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Write signals for multiple assets
    
    Args:
        signals: List of signal dictionaries
        
    Returns:
        Dict mapping symbol to write success status
    """
    results = {}
    for signal in signals:
        symbol = signal.get("symbol", "UNKNOWN")
        results[symbol] = write_signal_atomic(signal)
    return results


def read_signal(symbol: str) -> Optional[Dict[str, Any]]:
    """Read the current signal for a symbol (for testing/verification)
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Signal dictionary or None if not found
    """
    config = get_bridge_config()
    safe_symbol = symbol.replace("/", "").replace("\\", "").replace(":", "")
    signal_file = os.path.join(config.signal_dir, f"{safe_symbol}_signal.json")
    
    try:
        with open(signal_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to read signal for {symbol}: {e}")
        return None


def clear_signals() -> int:
    """Clear all signal files (for testing/cleanup)
    
    Returns:
        Number of files removed
    """
    config = get_bridge_config()
    count = 0
    
    try:
        if os.path.exists(config.signal_dir):
            for filename in os.listdir(config.signal_dir):
                if filename.endswith("_signal.json"):
                    os.remove(os.path.join(config.signal_dir, filename))
                    count += 1
    except Exception as e:
        logger.error(f"Failed to clear signals: {e}")
        
    return count


def get_signal_dir() -> str:
    """Get the current signal directory path"""
    return get_bridge_config().signal_dir


def is_bridge_available() -> bool:
    """Check if the MT5 bridge is available and writable
    
    Returns:
        True if bridge directory exists and is writable
    """
    config = get_bridge_config()
    
    if not config.enabled:
        return False
        
    try:
        os.makedirs(config.signal_dir, exist_ok=True)
        test_file = os.path.join(config.signal_dir, ".bridge_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False


def write_signal_output(signal_dict: Dict[str, Any]) -> bool:
    """Write a trading signal to the canonical signal_output.json file
    
    This is the AUTHORITATIVE function for MT5 signal output.
    It writes to a single file (signal_output.json) that MT5 EA reads.
    
    The file is overwritten atomically on each call - no appending.
    
    Args:
        signal_dict: Dictionary containing signal data with required keys:
            - symbol: Trading symbol (e.g., "XAUUSD")
            - signal: Signal direction ("BUY", "SELL", "HOLD", or null)
            - confidence: Confidence level (0.0 to 1.0)
            - timestamp: ISO-8601 format timestamp
            
    Returns:
        bool: True if write succeeded, False otherwise
    """
    config = get_bridge_config()
    
    if not config.enabled:
        logger.debug("MT5 Bridge disabled, skipping signal write")
        return False
    
    # Validate required fields
    required_fields = ["symbol", "signal", "confidence", "timestamp"]
    for field in required_fields:
        if field not in signal_dict:
            logger.error(f"MT5 signal missing required field: {field}")
            return False
    
    # Canonical output file
    signal_file = os.path.join(config.signal_dir, "signal_output.json")
    
    # Ensure all values are JSON-serializable
    try:
        serializable_signal = _ensure_json_serializable(signal_dict)
    except Exception as e:
        logger.error(f"Failed to serialize signal: {e}")
        return False
    
    with _write_lock:
        try:
            # Ensure directory exists
            os.makedirs(config.signal_dir, exist_ok=True)
            
            # Write to temporary file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix="signal_output_",
                dir=config.signal_dir
            )
            
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(serializable_signal, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                    
                # Atomic move to final location (overwrites existing)
                shutil.move(temp_path, signal_file)
                
                logger.info(f"MT5 signal_output.json written: {serializable_signal}")
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
                
        except Exception as e:
            logger.error(f"Failed to write signal_output.json: {e}")
            return False


def read_signal_output() -> Optional[Dict[str, Any]]:
    """Read the current signal from signal_output.json
    
    Returns:
        Signal dictionary or None if not found
    """
    config = get_bridge_config()
    signal_file = os.path.join(config.signal_dir, "signal_output.json")
    
    try:
        with open(signal_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to read signal_output.json: {e}")
        return None
