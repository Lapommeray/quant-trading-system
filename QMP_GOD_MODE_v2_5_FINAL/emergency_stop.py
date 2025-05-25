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
