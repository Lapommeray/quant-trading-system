"""
Advanced Modules for QMP Overrider

This package contains advanced modules that enhance the QMP Overrider system with
specialized capabilities for market perception and execution.

Modules:
1. Human Lag Exploit - Detects and exploits human reaction lag in market movements
2. Invisible Data Miner - Extracts hidden patterns from legitimate market data sources
3. Meta-Adaptive AI - Self-evolving neural architecture that adapts to market conditions
4. Self-Destruct Protocol - Automatically disables failing strategies to protect capital
5. Quantum Sentiment Decoder - Decodes quantum-level sentiment patterns in market data
"""

from .human_lag_exploit import HumanLagExploit
from .invisible_data_miner import InvisibleDataMiner
from .meta_adaptive_ai import MetaAdaptiveAI
from .self_destruct_protocol import SelfDestructProtocol
from .quantum_sentiment_decoder import QuantumSentimentDecoder

__all__ = [
    'HumanLagExploit',
    'InvisibleDataMiner',
    'MetaAdaptiveAI',
    'SelfDestructProtocol',
    'QuantumSentimentDecoder'
]
