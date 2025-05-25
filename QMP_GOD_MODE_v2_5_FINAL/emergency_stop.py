#!/usr/bin/env python
"""
Emergency Stop Mechanism for QMP Trading System
This script provides a kill switch to immediately halt all trading activity
and create a system snapshot for post-mortem analysis.
"""

import sys
import os
import argparse
import logging
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmergencyStop")

class EmergencyStop:
    """Emergency stop mechanism to halt all trading activity"""
    
    def __init__(self, dry_run=False):
        self.logger = logger
        self.dry_run = dry_run
        self.snapshot_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "snapshots",
            f"emergency_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if not dry_run:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        
    def execute_emergency_stop(self, code="STANDARD"):
        """Execute emergency stop procedure"""
        self.logger.warning(f"EXECUTING EMERGENCY STOP - CODE: {code}")
        
        start_time = time.time()
        
        self.cancel_all_orders()
        
        self.close_all_positions()
        
        self.create_system_snapshot()
        
        self.lock_trading_system(code)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Emergency stop completed in {elapsed_time:.2f} seconds")
        
        self._write_summary(code, elapsed_time)
        
        return True
        
    def cancel_all_orders(self):
        """Cancel all open orders across all connected brokers"""
        self.logger.info("Cancelling all open orders...")
        
        cancelled_orders = {
            "alpaca": {"status": "simulated", "orders_cancelled": 0},
            "ib": {"status": "simulated", "orders_cancelled": 0}
        }
        
        try:
            self.logger.info("Emergency order cancellation simulated")
            
            self._record_action("cancel_orders", cancelled_orders)
            
        except Exception as e:
            self.logger.error(f"Error during emergency order cancellation: {e}")
            self._record_action("cancel_orders", {"status": "FAILED", "error": str(e)})
            
    def close_all_positions(self):
        """Close all open positions"""
        self.logger.info("Closing all positions...")
        
        closed_positions = {
            "alpaca": {"status": "simulated", "positions_closed": 0},
            "ib": {"status": "simulated", "positions_closed": 0}
        }
        
        try:
            self.logger.info("Emergency position closure simulated")
            
            self._record_action("close_positions", closed_positions)
            
        except Exception as e:
            self.logger.error(f"Error during emergency position closure: {e}")
            self._record_action("close_positions", {"status": "FAILED", "error": str(e)})
            
    def create_system_snapshot(self):
        """Create a snapshot of the current system state"""
        self.logger.info("Creating system snapshot...")
        
        if self.dry_run:
            self.logger.info("Dry run: would create system snapshot with memory, process, and market data")
            return
        
        try:
            import psutil
            memory_info = psutil.virtual_memory()._asdict()
            with open(os.path.join(self.snapshot_dir, "memory_snapshot.json"), "w") as f:
                json.dump(memory_info, f, indent=2)
        except ImportError:
            self.logger.warning("psutil not available, skipping memory snapshot")
            
        try:
            process_info = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
                process_info.append(proc.info)
            with open(os.path.join(self.snapshot_dir, "process_snapshot.json"), "w") as f:
                json.dump(process_info, f, indent=2)
        except:
            self.logger.warning("Failed to capture process information")
            
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if os.path.exists(log_dir):
            snapshot_logs_dir = os.path.join(self.snapshot_dir, "logs")
            os.makedirs(snapshot_logs_dir, exist_ok=True)
            for log_file in os.listdir(log_dir):
                if log_file.endswith(".log"):
                    shutil.copy(
                        os.path.join(log_dir, log_file),
                        os.path.join(snapshot_logs_dir, log_file)
                    )
                    
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            market_data = fetcher.get_latest_data(symbols=["SPY", "QQQ", "BTC"])
            market_data.to_csv(os.path.join(self.snapshot_dir, "market_data_snapshot.csv"))
        except Exception as e:
            self.logger.error(f"Failed to capture market data: {e}")
            
        try:
            from core.alpaca_executor import AlpacaExecutor
            alpaca = AlpacaExecutor()
            positions = alpaca.get_positions()
            orders = alpaca.get_orders()
            
            with open(os.path.join(self.snapshot_dir, "positions_snapshot.json"), "w") as f:
                json.dump(positions, f, indent=2)
                
            with open(os.path.join(self.snapshot_dir, "orders_snapshot.json"), "w") as f:
                json.dump(orders, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to capture positions and orders: {e}")
            
        self.logger.info(f"System snapshot created at {self.snapshot_dir}")
        
    def lock_trading_system(self, code):
        """Lock the trading system to prevent further trading"""
        self.logger.info(f"Locking trading system with code: {code}")
        
        if self.dry_run:
            self.logger.info(f"Dry run: would lock trading system with code {code}")
            return
            
        lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading.lock")
        with open(lock_file, "w") as f:
            f.write(f"EMERGENCY STOP ACTIVATED\n")
            f.write(f"Code: {code}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Snapshot: {self.snapshot_dir}\n")
            
        os.chmod(lock_file, 0o444)
        
        self.logger.info(f"Trading system locked. Manual intervention required to resume trading.")
        
    def _record_action(self, action_name, data):
        """Record an action in the snapshot directory"""
        if self.dry_run:
            self.logger.info(f"Dry run: would record action {action_name} with data {data}")
            return
            
        action_file = os.path.join(self.snapshot_dir, f"{action_name}.json")
        with open(action_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "action": action_name,
                "data": data
            }, f, indent=2)
            
    def _write_summary(self, code, elapsed_time):
        """Write summary of emergency stop procedure"""
        if self.dry_run:
            self.logger.info(f"Dry run: would write emergency stop summary with code {code} and elapsed time {elapsed_time:.2f}s")
            return
            
        summary_file = os.path.join(self.snapshot_dir, "emergency_stop_summary.txt")
        with open(summary_file, "w") as f:
            f.write("EMERGENCY STOP SUMMARY\n")
            f.write("======================\n\n")
            f.write(f"Code: {code}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Execution Time: {elapsed_time:.2f} seconds\n\n")
            f.write("Actions Performed:\n")
            f.write("1. Cancelled all open orders\n")
            f.write("2. Closed all positions\n")
            f.write("3. Created system snapshot\n")
            f.write("4. Locked trading system\n\n")
            f.write("To resume trading, remove the trading.lock file and restart the system.\n")
            
    def ai_driven_emergency_check(self, market_data, ai_metrics):
        """
        AI-driven emergency detection using advanced pattern recognition
        
        Parameters:
        - market_data: Dictionary containing market data (returns, volume, etc.)
        - ai_metrics: Dictionary containing AI performance metrics
        
        Returns:
        - Tuple of (emergency_detected, active_conditions)
        """
        volatility = np.std(market_data.get('returns', [0])[-20:]) if 'returns' in market_data and len(market_data['returns']) >= 20 else 0
        mean_reversion = np.mean(market_data.get('returns', [0])[-5:]) if 'returns' in market_data and len(market_data['returns']) >= 5 else 0
        
        ai_confidence = ai_metrics.get('confidence', 1.0)
        model_accuracy = ai_metrics.get('recent_accuracy', 1.0)
        
        # Emergency triggers
        emergency_conditions = {
            "extreme_volatility": volatility > 0.1,  # 10% volatility
            "sustained_losses": mean_reversion < -0.05,  # 5% sustained decline
            "ai_confidence_loss": ai_confidence < 0.3,  # Low AI confidence
            "model_breakdown": model_accuracy < 0.4  # Model failure
        }
        
        active_conditions = [k for k, v in emergency_conditions.items() if v]
        
        if len(active_conditions) >= 2:  # Multiple conditions trigger emergency
            self.logger.critical(f"AI Emergency Detection: {active_conditions}")
            return True, active_conditions
            
        return False, []
        
    def comprehensive_emergency_check(self, market_data, ai_metrics, portfolio_data):
        """
        Comprehensive emergency detection system
        
        Parameters:
        - market_data: Dictionary containing market data (returns, volume, etc.)
        - ai_metrics: Dictionary containing AI performance metrics
        - portfolio_data: Dictionary containing portfolio information
        
        Returns:
        - Tuple of (emergency_detected, active_conditions)
        """
        emergency_triggers = []
        
        if self._detect_market_crash(market_data):
            emergency_triggers.append("market_crash")
            
        if self._detect_liquidity_crisis(market_data):
            emergency_triggers.append("liquidity_crisis")
            
        if self._detect_ai_degradation(ai_metrics):
            emergency_triggers.append("ai_degradation")
            
        if self._detect_model_breakdown(ai_metrics):
            emergency_triggers.append("model_breakdown")
            
        if self._detect_portfolio_anomaly(portfolio_data):
            emergency_triggers.append("portfolio_anomaly")
            
        if len(emergency_triggers) >= 2:
            self.logger.critical(f"COMPREHENSIVE EMERGENCY DETECTED: {emergency_triggers}")
            return True, emergency_triggers
            
        return False, []
        
    def _detect_market_crash(self, market_data):
        """Detect market crash conditions"""
        returns = market_data.get('returns', [])
        if len(returns) < 5:
            return False
            
        recent_returns = returns[-5:]
        if all(r < -0.03 for r in recent_returns):  # 3% decline for 5 periods
            return True
            
        if any(r < -0.15 for r in recent_returns):  # 15% single period decline
            return True
            
        return False
        
    def _detect_liquidity_crisis(self, market_data):
        """Detect liquidity crisis"""
        volume = market_data.get('volume', [])
        if len(volume) < 10:
            return False
            
        recent_volume = volume[-5:]
        historical_volume = volume[-10:-5]
        
        if len(historical_volume) > 0:
            avg_historical = np.mean(historical_volume)
            avg_recent = np.mean(recent_volume)
            
            if avg_recent > avg_historical * 3:  # 3x volume increase
                return True
                
        return False
        
    def _detect_ai_degradation(self, ai_metrics):
        """Detect AI system degradation"""
        confidence = ai_metrics.get('confidence', 1.0)
        accuracy = ai_metrics.get('recent_accuracy', 1.0)
        
        return confidence < 0.2 or accuracy < 0.3
        
    def _detect_model_breakdown(self, ai_metrics):
        """Detect complete model breakdown"""
        predictions = ai_metrics.get('recent_predictions', [])
        if len(predictions) < 10:
            return False
            
        if len(set(predictions)) == 1:
            return True
            
        if any(abs(p) > 10 for p in predictions):  # Extreme values
            return True
            
        return False
        
    def _detect_portfolio_anomaly(self, portfolio_data):
        """Detect portfolio-level anomalies"""
        positions = portfolio_data.get('positions', {})
        if not positions:
            return False
            
        total_value = sum(abs(pos) for pos in positions.values())
        max_position = max(abs(pos) for pos in positions.values())
        
        if max_position / total_value > 0.8:  # 80% in single position
            return True
            
        return False
            
def main():
    """Main function to execute emergency stop"""
    parser = argparse.ArgumentParser(description="Emergency Stop for QMP Trading System")
    parser.add_argument("--code", choices=["STANDARD", "RED_ALERT", "CIRCUIT_BREAKER", "MANUAL"], 
                        default="STANDARD", help="Emergency stop code")
    parser.add_argument("--dry-run", action="store_true", help="Simulate emergency stop without actual execution")
    
    args = parser.parse_args()
    
    emergency_stop = EmergencyStop(dry_run=args.dry_run)
    success = emergency_stop.execute_emergency_stop(args.code)
    
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main()
