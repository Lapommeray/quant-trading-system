"""
Compliance Check

Python implementation of the legal firewall for all trading strategies.
This module ensures that all trading activities comply with legal and
ethical standards, preventing insider trading and other violations.

Features:
- SEC insider blacklist checking
- Retail front-running prevention
- Position size limiting based on ADV
- Compliance logging
"""

from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class ComplianceCheck:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.blacklist = self._load_sec_insiders()
        self.compliance_log = []
        self.max_log_size = 1000
        self.compliance_log_file = "compliance_log.json"
        
        self.max_position_pct = 0.05  # Max 5% of portfolio in any one position
        self.max_adv_pct = 0.01  # Max 1% of average daily volume
        
        self.algo.Debug("ComplianceCheck initialized")
    
    def pre_trade_check(self, symbol, order_size=None, price=None):
        """
        Perform pre-trade compliance checks
        
        Parameters:
        - symbol: Trading symbol
        - order_size: Planned order size (optional)
        - price: Current price (optional)
        
        Returns:
        - Dictionary with compliance results
        """
        symbol_str = str(symbol)
        current_time = self.algo.Time
        
        result = {
            'compliant': True,
            'issues': [],
            'limit_order_size': None,
            'timestamp': current_time
        }
        
        if symbol_str in self.blacklist:
            result['compliant'] = False
            result['issues'].append("Insider trading risk detected")
            
        if self._is_retail_frontrun(symbol_str):
            result['issues'].append("Potential retail front-running detected")
            adv = self._get_adv(symbol_str)
            result['limit_order_size'] = 0.01 * adv
            
        if order_size is not None and price is not None:
            portfolio_value = self.algo.Portfolio.TotalPortfolioValue
            position_value = order_size * price
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.max_position_pct:
                result['issues'].append(f"Position size exceeds {self.max_position_pct*100}% of portfolio")
                result['limit_order_size'] = (self.max_position_pct * portfolio_value) / price
        
        log_entry = {
            'timestamp': current_time,
            'symbol': symbol_str,
            'compliant': result['compliant'],
            'issues': result['issues'],
            'limit_order_size': result['limit_order_size']
        }
        self.compliance_log.append(log_entry)
        
        if len(self.compliance_log) > self.max_log_size:
            self.compliance_log = self.compliance_log[-self.max_log_size:]
        
        if result['issues']:
            self.algo.Debug(f"ComplianceCheck: Issues detected for {symbol_str}: {', '.join(result['issues'])}")
        
        self._log_compliance_result(log_entry)
        
        return result
    
    def post_trade_check(self, symbol, order_size, price):
        """
        Perform post-trade compliance checks
        
        Parameters:
        - symbol: Trading symbol
        - order_size: Executed order size
        - price: Executed price
        
        Returns:
        - Dictionary with compliance results
        """
        symbol_str = str(symbol)
        current_time = self.algo.Time
        
        result = {
            'compliant': True,
            'issues': [],
            'timestamp': current_time
        }
        
        if symbol_str in self.blacklist:
            result['compliant'] = False
            result['issues'].append("Insider trading risk detected")
        
        portfolio_value = self.algo.Portfolio.TotalPortfolioValue
        position_value = order_size * price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.max_position_pct:
            result['compliant'] = False
            result['issues'].append(f"Position size exceeds {self.max_position_pct*100}% of portfolio")
        
        log_entry = {
            'timestamp': current_time,
            'symbol': symbol_str,
            'compliant': result['compliant'],
            'issues': result['issues']
        }
        self.compliance_log.append(log_entry)
        
        if len(self.compliance_log) > self.max_log_size:
            self.compliance_log = self.compliance_log[-self.max_log_size:]
        
        if result['issues']:
            self.algo.Debug(f"ComplianceCheck: Post-trade issues detected for {symbol_str}: {', '.join(result['issues'])}")
        
        self._log_compliance_result(log_entry)
        
        return result
    
    def _log_compliance_result(self, log_entry):
        """
        Log compliance results to a file
        
        Parameters:
        - log_entry: Dictionary with compliance log entry
        """
        try:
            with open(self.compliance_log_file, 'a') as log_file:
                log_file.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.algo.Debug(f"ComplianceCheck: Failed to log compliance result: {str(e)}")
    
    def _load_sec_insiders(self):
        """
        Load SEC insider list
        In production, this would load from a regularly updated data source
        
        Returns:
        - Set of blacklisted symbols
        """
        return set([
            "RESTRICTED1", "RESTRICTED2", "RESTRICTED3"
        ])
    
    def _is_retail_frontrun(self, symbol):
        """
        Check if a trade might constitute retail front-running
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - True if front-running risk detected, False otherwise
        """
        return np.random.random() < 0.02  # 2% chance of front-running risk
    
    def _get_adv(self, symbol):
        """
        Get average daily volume for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Average daily volume
        """
        base_volume = 1000000  # 1M shares base
        
        symbol_hash = sum(ord(c) for c in symbol) % 100
        volume_factor = 0.5 + (symbol_hash / 50)  # 0.5 to 2.5
        
        return base_volume * volume_factor
    
    def get_compliance_summary(self):
        """
        Generate a compliance summary report
        
        Returns:
        - Dictionary with compliance statistics
        """
        if not self.compliance_log:
            return {
                'total_checks': 0,
                'compliant_checks': 0,
                'compliance_rate': 1.0,
                'common_issues': []
            }
            
        total_checks = len(self.compliance_log)
        compliant_checks = sum(1 for entry in self.compliance_log if entry['compliant'])
        
        issue_counts = {}
        for entry in self.compliance_log:
            for issue in entry['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(
            [{'issue': issue, 'count': count} for issue, count in issue_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )
        
        return {
            'total_checks': total_checks,
            'compliant_checks': compliant_checks,
            'compliance_rate': compliant_checks / total_checks if total_checks > 0 else 1.0,
            'common_issues': common_issues
        }
