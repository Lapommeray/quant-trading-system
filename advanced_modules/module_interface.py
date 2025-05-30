"""
Module Interface for Advanced Trading Modules

This interface defines the standard contract that all advanced modules must implement
to integrate with the QMP Overrider system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

class AdvancedModuleInterface(ABC):
    """
    Abstract base class for all advanced modules in the QMP Overrider system.
    
    All modules must implement these methods to ensure proper integration
    with the OversoulDirector and other system components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the module with optional configuration."""
        self.config = config or {}
        self.initialized = False
        self.last_analysis = None
        self.last_signal = None
        self.activation_level = 0.0
        self.confidence = 0.0
        self.module_name = self.__class__.__name__
        self.module_type = "advanced"
        self.module_version = "1.0.0"
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the module. Must be called before any other method."""
        pass
        
    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return analysis results."""
        pass
        
    @abstractmethod
    def get_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading signal based on market data."""
        pass
        
    @abstractmethod
    def validate_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trading signal against market data."""
        pass
