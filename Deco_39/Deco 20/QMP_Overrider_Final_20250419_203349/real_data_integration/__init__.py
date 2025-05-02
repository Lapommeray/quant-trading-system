"""
Real Data Integration Module for QMP Overrider

This module integrates legitimate, legally accessible data sources into the QMP Overrider system,
providing real-time market intelligence from approved sources.

Approved data sources:
1. PredictLeads - Biometric and corporate leadership tracking
2. Thinknum - Workplace sentiment, hiring trends, digital footprints
3. Thasos - Real-time satellite & parking lot analytics
4. Glassnode - On-chain crypto data (Bitcoin, Ethereum)
5. FlowAlgo - Institutional dark pool alerts
6. Orbital Insight - Satellite imagery and macro-level movement tracking
7. Alpaca - Real-time equity trading data + earnings transcripts
8. Visa/Amex/SpendTrend+ - Purchase metadata (if licensed)
"""

from .real_data_connector import RealDataConnector, ComplianceCheck

__all__ = [
    'RealDataConnector',
    'ComplianceCheck'
]
