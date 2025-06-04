"""
Configuration module for quant trading system
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from .config_manager import ConfigManager
except ImportError:
    ConfigManager = None

try:
    from .settings import Settings
except ImportError:
    Settings = None

try:
    from .environment import Environment
except ImportError:
    Environment = None

class DefaultConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config = None
    
    def _find_config_file(self) -> str:
        possible_paths = [
            "config.yaml",
            "Sa_son_code/quant_trading_system/config/config.yaml",
            "../../../config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return "config.yaml"
    
    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            try:
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            except FileNotFoundError:
                self._config = self._get_default_config()
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "system": {
                "name": "Quant Trading System",
                "version": "2.5.0",
                "environment": os.getenv("QTS_ENV", "development"),
                "debug": os.getenv("QTS_DEBUG", "true").lower() == "true"
            },
            "quantum": {
                "enabled": os.getenv("QTS_QUANTUM_ENABLED", "true").lower() == "true",
                "consciousness_mode": "advanced",
                "dimensional_analysis": 11
            },
            "trading": {
                "mode": os.getenv("QTS_TRADING_MODE", "paper"),
                "max_position_size": 0.1,
                "risk_tolerance": 0.02
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        config = self.load_config()
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

if ConfigManager is None:
    ConfigManager = DefaultConfigManager

__all__ = []

if ConfigManager is not None:
    __all__.append('ConfigManager')
if Settings is not None:
    __all__.append('Settings')
if Environment is not None:
    __all__.append('Environment')
