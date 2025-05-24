"""
Market Mind Reader Package

This package contains the Market Mind Reader upgrade for the QMP Overrider system.
It includes the Fed Whisperer, Candlestick DNA Sequencer, Liquidity X-Ray, Retail DNA Extractor,
Quantum Noise Trader, Congressional Trades Analyzer, Weather Alpha Generator, and Short Interest Anomaly Detector modules.
"""

import logging
logger = logging.getLogger(__name__)

__all__ = []

try:
    from .fed_whisperer import FedWhisperer
    __all__.append('FedWhisperer')
except ImportError as e:
    logger.warning(f"Could not import FedWhisperer: {e}")

try:
    from .candlestick_dna_sequencer import CandlestickDNASequencer
    __all__.append('CandlestickDNASequencer')
except ImportError as e:
    logger.warning(f"Could not import CandlestickDNASequencer: {e}")

try:
    from .liquidity_xray import LiquidityXRay
    __all__.append('LiquidityXRay')
except ImportError as e:
    logger.warning(f"Could not import LiquidityXRay: {e}")

try:
    from .enhanced_indicator import EnhancedIndicator
    __all__.append('EnhancedIndicator')
except ImportError as e:
    logger.warning(f"Could not import EnhancedIndicator: {e}")

try:
    from .retail_dna_extractor import RetailDNAExtractor
    __all__.append('RetailDNAExtractor')
except ImportError as e:
    logger.warning(f"Could not import RetailDNAExtractor: {e}")

try:
    from .quantum_noise_trader import QuantumNoiseTrader
    __all__.append('QuantumNoiseTrader')
except ImportError as e:
    logger.warning(f"Could not import QuantumNoiseTrader: {e}")

try:
    from .congressional_trades_analyzer import CongressionalTradesAnalyzer
    __all__.append('CongressionalTradesAnalyzer')
except ImportError as e:
    logger.warning(f"Could not import CongressionalTradesAnalyzer: {e}")

try:
    from .weather_alpha_generator import WeatherAlphaGenerator
    __all__.append('WeatherAlphaGenerator')
except ImportError as e:
    logger.warning(f"Could not import WeatherAlphaGenerator: {e}")

try:
    from .short_interest_anomaly_detector import ShortInterestAnomalyDetector
    __all__.append('ShortInterestAnomalyDetector')
except ImportError as e:
    logger.warning(f"Could not import ShortInterestAnomalyDetector: {e}")

try:
    from .walk_forward_backtest import WalkForwardBacktester
    __all__.append('WalkForwardBacktester')
except ImportError as e:
    logger.warning(f"Could not import WalkForwardBacktester: {e}")

try:
    from .risk_manager import RiskManager
    __all__.append('RiskManager')
except ImportError as e:
    logger.warning(f"Could not import RiskManager: {e}")

try:
    from .dynamic_slippage import DynamicLiquiditySlippage
    __all__.append('DynamicLiquiditySlippage')
except ImportError as e:
    logger.warning(f"Could not import DynamicLiquiditySlippage: {e}")

try:
    from .event_blackout import EventBlackoutManager
    __all__.append('EventBlackoutManager')
except ImportError as e:
    logger.warning(f"Could not import EventBlackoutManager: {e}")

try:
    from .performance_optimizer import PerformanceOptimizer
    __all__.append('PerformanceOptimizer')
except ImportError as e:
    logger.warning(f"Could not import PerformanceOptimizer: {e}")
