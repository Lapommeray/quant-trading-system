"""
Enhanced Performance Metrics - Track 200% accuracy and super high confidence
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

class EnhancedPerformanceMetrics:
    """
    Enhanced performance tracking for 200% accuracy validation
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.logger = logging.getLogger("EnhancedPerformanceMetrics")
        self.trade_history = []
        self.accuracy_history = []
        self.confidence_history = []
        
    def record_trade_prediction(self, symbol, signal, confidence, ai_consensus_result, temporal_result, reality_result):
        """Record a trade prediction for later accuracy calculation"""
        record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'predicted_signal': signal,
            'confidence': confidence,
            'ai_consensus_achieved': ai_consensus_result.get('consensus_achieved', False),
            'consensus_ratio': ai_consensus_result.get('consensus_ratio', 0),
            'temporal_opportunity': temporal_result.get('opportunity', False),
            'reality_compliant': reality_result.get('reality_compliant', False),
            'actual_outcome': None,  # To be filled later
            'accuracy_multiplier': ai_consensus_result.get('accuracy_multiplier', 1.0)
        }
        
        self.trade_history.append(record)
        
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def update_trade_outcome(self, symbol, timestamp, actual_return):
        """Update the actual outcome of a trade prediction"""
        for record in reversed(self.trade_history):
            if (record['symbol'] == symbol and 
                abs((record['timestamp'] - timestamp).total_seconds()) < 300):  # Within 5 minutes
                
                if record['predicted_signal'] == "BUY" and actual_return > 0:
                    record['actual_outcome'] = 'correct'
                elif record['predicted_signal'] == "SELL" and actual_return < 0:
                    record['actual_outcome'] = 'correct'
                elif record['predicted_signal'] == "NEUTRAL":
                    record['actual_outcome'] = 'neutral'
                else:
                    record['actual_outcome'] = 'incorrect'
                
                record['actual_return'] = actual_return
                break
    
    def calculate_current_accuracy(self):
        """Calculate current accuracy metrics"""
        completed_trades = [t for t in self.trade_history if t['actual_outcome'] is not None]
        
        if not completed_trades:
            return {'accuracy': 0, 'accuracy_multiplier': 1.0, 'super_high_confidence_rate': 0}
        
        correct_trades = [t for t in completed_trades if t['actual_outcome'] == 'correct']
        basic_accuracy = len(correct_trades) / len(completed_trades)
        
        consensus_trades = [t for t in completed_trades if t['ai_consensus_achieved']]
        if consensus_trades:
            consensus_correct = [t for t in consensus_trades if t['actual_outcome'] == 'correct']
            consensus_accuracy = len(consensus_correct) / len(consensus_trades)
            accuracy_multiplier = consensus_accuracy / basic_accuracy if basic_accuracy > 0 else 1.0
        else:
            accuracy_multiplier = 1.0
        
        high_conf_trades = [t for t in completed_trades if t['confidence'] > 0.9]
        super_high_confidence_rate = len(high_conf_trades) / len(completed_trades)
        
        achieved_200_percent = accuracy_multiplier >= 2.0
        
        return {
            'basic_accuracy': basic_accuracy,
            'accuracy_multiplier': accuracy_multiplier,
            'achieved_200_percent': achieved_200_percent,
            'super_high_confidence_rate': super_high_confidence_rate,
            'total_trades': len(completed_trades),
            'consensus_trades': len(consensus_trades),
            'never_loss_rate': self._calculate_never_loss_rate(completed_trades)
        }
    
    def _calculate_never_loss_rate(self, completed_trades):
        """Calculate the rate of trades that resulted in no loss"""
        if not completed_trades:
            return 0
        
        no_loss_trades = [t for t in completed_trades 
                         if t.get('actual_return', 0) >= 0 or t['actual_outcome'] == 'neutral']
        
        return len(no_loss_trades) / len(completed_trades)
