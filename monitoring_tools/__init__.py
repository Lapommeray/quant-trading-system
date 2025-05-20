"""
Monitoring Tools for QMP Overrider

This package contains specialized monitoring tools that provide real-time
market intelligence and event detection capabilities for the QMP Overrider system.

Tools:
1. BTC Off-chain Monitor - Tracks large Bitcoin transfers not visible on public blockchain
2. Fed Jet Monitor - Analyzes Federal Reserve officials' private jet movements
3. Spoofing Detector - Identifies high-frequency trading manipulation patterns
4. Stress Detector - Analyzes executive biometric signals during earnings calls
5. Port Activity Analyzer - Monitors global shipping and supply chain disruptions
"""

from .btc_offchain_monitor import BTCOffchainMonitor
from .fed_jet_monitor import FedJetMonitor
from .spoofing_detector import SpoofingDetector
from .stress_detector import StressDetector
from .port_activity_analyzer import PortActivityAnalyzer

__all__ = [
    'BTCOffchainMonitor',
    'FedJetMonitor',
    'SpoofingDetector',
    'StressDetector',
    'PortActivityAnalyzer'
]
