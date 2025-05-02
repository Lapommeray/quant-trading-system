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
6. Event Probability Module - Aggregates multiple event indicators for unified probability scores
7. Human Lag Exploit - Detects and exploits human reaction lag in market movements
8. Invisible Data Miner - Extracts hidden patterns from legitimate market data sources
9. Meta-Adaptive AI - Self-evolving neural architecture that adapts to market conditions
10. Quantum Sentiment Decoder - Decodes quantum-level sentiment patterns in market data
11. Self-Destruct Protocol - Automatically disables failing strategies to protect capital
12. Compliance Firewall - Ensures all trading activities comply with regulatory requirements
"""

from .btc_offchain_monitor import BTCOffchainMonitor
from .fed_jet_monitor import FedJetMonitor
from .spoofing_detector import SpoofingDetector
from .stress_detector import StressDetector
from .port_activity_analyzer import PortActivityAnalyzer
from .event_probability_module import EventProbabilityModule
from .human_lag_exploit import HumanLagExploit
from .invisible_data_miner import InvisibleDataMiner
from .meta_adaptive_ai import MetaAdaptiveAI
from .quantum_sentiment_decoder import QuantumSentimentDecoder
from .self_destruct_protocol import SelfDestructProtocol
from .compliance_firewall import ComplianceFirewall

__all__ = [
    'BTCOffchainMonitor',
    'FedJetMonitor',
    'SpoofingDetector',
    'StressDetector',
    'PortActivityAnalyzer',
    'EventProbabilityModule',
    'HumanLagExploit',
    'InvisibleDataMiner',
    'MetaAdaptiveAI',
    'QuantumSentimentDecoder',
    'SelfDestructProtocol',
    'ComplianceFirewall'
]
