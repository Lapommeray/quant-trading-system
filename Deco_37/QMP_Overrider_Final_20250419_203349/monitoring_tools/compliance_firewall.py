"""
Compliance Firewall for QMP Overrider

This module provides legal screening for trading signals and ensures all data sources
are publicly available and legally accessible. It acts as a firewall between the
trading system and potential regulatory issues.

The Compliance Firewall:
1. Validates all data sources are public and legally accessible
2. Screens trading signals for potential regulatory issues
3. Maintains an audit trail of all compliance checks
4. Blocks execution of trades that fail compliance checks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

class ComplianceFirewall:
    """
    Compliance Firewall that ensures all trading activities comply with
    regulatory requirements and only use legally accessible data sources.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Compliance Firewall
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.approved_data_sources = [
            "market_data",           # Standard price/volume data
            "sec_filings",           # Public SEC filings
            "earnings_calls",        # Public earnings call transcripts
            "news_sentiment",        # Public news sentiment analysis
            "social_media",          # Public social media sentiment
            "macro_economic",        # Public economic indicators
            "satellite_imagery",     # Commercial satellite imagery
            "blockchain_data",       # Public blockchain transactions
            "options_flow",          # Public options flow data
            "corporate_events",      # Public corporate event calendars
            "fed_announcements",     # Public Federal Reserve announcements
            "retail_trends"          # Aggregated retail spending trends
        ]
        
        self.blocked_data_sources = [
            "private_communications",  # Private emails, texts, calls
            "insider_information",     # Non-public material information
            "hacked_data",             # Illegally obtained data
            "confidential_documents",  # Confidential internal documents
            "private_transactions",    # Non-public transaction data
            "restricted_databases"     # Access-restricted databases
        ]
        
        self.compliance_log = []
        self.audit_trail = []
        
    def check_trade(self, symbol, direction, data_sources, current_time=None):
        """
        Check if a trade complies with all regulatory requirements
        
        Parameters:
        - symbol: Trading symbol
        - direction: "BUY" or "SELL"
        - data_sources: List of data sources used for the signal
        - current_time: Optional timestamp to use instead of algorithm time
        
        Returns:
        - Dictionary with approval status and reason
        """
        if current_time is None and self.algorithm:
            current_time = self.algorithm.Time
        elif current_time is None:
            current_time = datetime.now()
            
        for source in data_sources:
            if source in self.blocked_data_sources:
                result = {
                    "approved": False,
                    "reason": f"Blocked data source used: {source}",
                    "timestamp": current_time
                }
                self._log_check(symbol, direction, result)
                return result
                
        for source in data_sources:
            if source not in self.approved_data_sources:
                result = {
                    "approved": False,
                    "reason": f"Unapproved data source: {source}",
                    "timestamp": current_time
                }
                self._log_check(symbol, direction, result)
                return result
        
        if self.algorithm is not None and hasattr(self.algorithm, 'Securities') and symbol in self.algorithm.Securities:
            security = self.algorithm.Securities[symbol]
            if not security.Exchange.ExchangeOpen:
                result = {
                    "approved": False,
                    "reason": "Market closed for this security",
                    "timestamp": current_time
                }
                self._log_check(symbol, direction, result)
                return result
        
        result = {
            "approved": True,
            "reason": "All compliance checks passed",
            "timestamp": current_time
        }
        self._log_check(symbol, direction, result)
        return result
    
    def pre_trade_check(self, symbol, direction, data_sources, current_time=None):
        """
        Alias for check_trade to maintain compatibility with existing code
        """
        return self.check_trade(symbol, direction, data_sources, current_time)
    
    def validate_data_source(self, source_name, source_type, source_description):
        """
        Validate if a data source is compliant with regulations
        
        Parameters:
        - source_name: Name of the data source
        - source_type: Type of data source
        - source_description: Description of the data source
        
        Returns:
        - True if compliant, False otherwise
        """
        if source_name in self.blocked_data_sources:
            return False
            
        if source_name in self.approved_data_sources:
            return True
            
        approved_types = [
            "public_data",
            "aggregated_data",
            "anonymized_data",
            "commercial_data"
        ]
        
        if source_type in approved_types:
            self.approved_data_sources.append(source_name)
            return True
            
        return False
    
    def _log_check(self, symbol, direction, result):
        """Log compliance check to audit trail"""
        log_entry = {
            "timestamp": result["timestamp"],
            "symbol": str(symbol),
            "direction": direction,
            "approved": result["approved"],
            "reason": result["reason"]
        }
        
        self.compliance_log.append(log_entry)
        self.audit_trail.append(log_entry)
        
        if self.algorithm:
            status = "APPROVED" if result["approved"] else "REJECTED"
            self.algorithm.Debug(f"Compliance {status}: {symbol} {direction} - {result['reason']}")
    
    def get_compliance_report(self, days=7):
        """
        Generate compliance report for the specified number of days
        
        Parameters:
        - days: Number of days to include in report
        
        Returns:
        - DataFrame with compliance statistics
        """
        if not self.compliance_log:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.compliance_log)
        
        if 'timestamp' in df.columns:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
        stats = {
            "total_checks": len(df),
            "approved": df['approved'].sum(),
            "rejected": len(df) - df['approved'].sum(),
            "approval_rate": df['approved'].mean() * 100 if len(df) > 0 else 0
        }
        
        if 'reason' in df.columns and 'approved' in df.columns:
            rejection_reasons = df[~df['approved']]['reason'].value_counts().to_dict()
            stats["rejection_reasons"] = rejection_reasons
            
        return stats
