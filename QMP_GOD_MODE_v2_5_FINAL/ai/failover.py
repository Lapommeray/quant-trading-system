"""
Dark Pool Failover Protocol Module

This module implements the Dark Pool Failover Protocol for the QMP Overrider system.
It provides real-time health monitoring and intelligent failover for dark pool liquidity access.
"""

import random
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class DarkPoolRouter:
    """
    Dark Pool Router for the QMP Overrider system.
    
    This class provides real-time health monitoring and intelligent failover
    for dark pool liquidity access, ensuring optimal execution quality.
    """
    
    def __init__(self, log_dir=None):
        """
        Initialize the Dark Pool Router.
        
        Parameters:
            log_dir: Directory to store router logs (or None for default)
        """
        self.logger = logging.getLogger("DarkPoolRouter")
        
        if log_dir is None:
            self.log_dir = Path("logs/dark_pool")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.pools = {
            "pool_alpha": {"health": 1.0, "last_check": None, "fill_rate": 0.95, "latency": 15},
            "pool_sigma": {"health": 1.0, "last_check": None, "fill_rate": 0.90, "latency": 12},
            "pool_omega": {"health": 1.0, "last_check": None, "fill_rate": 0.85, "latency": 10}
        }
        
        self.current_pool = "pool_alpha"
        self.failover_history = []
        
        self._load_history()
        
        self.logger.info(f"Dark Pool Router initialized with {len(self.pools)} pools")
    
    def _load_history(self):
        """Load failover history from file"""
        history_file = self.log_dir / "failover_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.failover_history = json.load(f)
                self.logger.info(f"Loaded {len(self.failover_history)} historical failovers")
            except Exception as e:
                self.logger.error(f"Error loading failover history: {e}")
    
    def _save_history(self):
        """Save failover history to file"""
        history_file = self.log_dir / "failover_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.failover_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving failover history: {e}")
    
    def check_pool_health(self):
        """
        Check the health of all dark pools.
        
        This method tests the latency and fill rate of each pool and
        updates their health scores accordingly.
        """
        self.logger.info("Checking health of all dark pools")
        
        for pool in self.pools:
            try:
                latency = self.test_latency(pool)
                fill_rate = self.test_fill_rate(pool)
                
                self.pools[pool]["latency"] = latency
                self.pools[pool]["fill_rate"] = fill_rate
                
                latency_score = 1.0 / (latency / 10) if latency > 0 else 0
                self.pools[pool]["health"] = (0.6 * fill_rate) + (0.4 * min(1.0, latency_score))
                self.pools[pool]["last_check"] = datetime.now()
                
                self.logger.debug(f"Pool {pool} health: {self.pools[pool]['health']:.2f} "
                                 f"(fill_rate={fill_rate:.2f}, latency={latency}ms)")
                
            except Exception as e:
                self.pools[pool]["health"] = 0.0
                self.pools[pool]["last_check"] = datetime.now()
                self.logger.error(f"Dark pool {pool} health check failed: {str(e)}")
    
    def get_optimal_pool(self, order_size):
        """
        Get the optimal dark pool for a given order size.
        
        Parameters:
            order_size: Size of the order to route
            
        Returns:
            Name of the optimal dark pool
        """
        self.check_pool_health()
        
        viable = {}
        for k, v in self.pools.items():
            if v["last_check"] is None:
                continue
                
            if (v["health"] > 0.7 and 
                (datetime.now() - v["last_check"]).total_seconds() < 30):
                viable[k] = v
        
        if not viable:
            self.logger.warning("No healthy dark pools available")
            return max(self.pools.keys(), key=lambda x: self.pools[x]["health"])
        
        if order_size > 1000:
            optimal_pool = max(viable.keys(), key=lambda x: viable[x]["health"])
            self.logger.info(f"Selected {optimal_pool} for large order (size={order_size})")
            return optimal_pool
        else:
            weights = [viable[pool]["health"] for pool in viable]
            optimal_pool = random.choices(list(viable.keys()), weights=weights)[0]
            self.logger.info(f"Selected {optimal_pool} for order (size={order_size})")
            return optimal_pool
    
    def execute_failover(self, reason="health_check"):
        """
        Execute a failover to the optimal dark pool.
        
        Parameters:
            reason: Reason for the failover
            
        Returns:
            New pool name
        """
        old_pool = self.current_pool
        
        self.current_pool = self.get_optimal_pool(100)
        
        failover_record = {
            "timestamp": datetime.now().timestamp(),
            "old_pool": old_pool,
            "new_pool": self.current_pool,
            "reason": reason,
            "pool_health": {k: v["health"] for k, v in self.pools.items()}
        }
        
        self.failover_history.append(failover_record)
        self._save_history()
        
        self.logger.warning(f"Failover executed: {old_pool} → {self.current_pool} (reason: {reason})")
        
        return self.current_pool
    
    def test_latency(self, pool):
        """
        Test the latency of a dark pool.
        
        Parameters:
            pool: Name of the pool to test
            
        Returns:
            Latency in milliseconds
        """
        
        current = self.pools[pool].get("latency", 15)
        
        variation = current * 0.3
        new_latency = max(5, current + random.uniform(-variation, variation))
        
        if random.random() < 0.05:
            new_latency *= random.uniform(2, 4)
            self.logger.warning(f"Latency spike detected in {pool}: {new_latency:.1f}ms")
        
        return new_latency
    
    def test_fill_rate(self, pool):
        """
        Test the fill rate of a dark pool.
        
        Parameters:
            pool: Name of the pool to test
            
        Returns:
            Fill rate as a fraction (0.0 to 1.0)
        """
        
        current = self.pools[pool].get("fill_rate", 0.9)
        
        variation = 0.1
        new_fill_rate = max(0.5, min(0.99, current + random.uniform(-variation, variation)))
        
        if random.random() < 0.05:
            new_fill_rate *= random.uniform(0.6, 0.8)
            self.logger.warning(f"Fill rate drop detected in {pool}: {new_fill_rate:.2f}")
        
        return new_fill_rate
    
    def route_order(self, order, force_pool=None):
        """
        Route an order to the optimal dark pool.
        
        Parameters:
            order: Order object with size and other details
            force_pool: Force routing to a specific pool (or None for automatic selection)
            
        Returns:
            Routing result with execution details
        """
        if force_pool and force_pool in self.pools:
            target_pool = force_pool
        else:
            target_pool = self.get_optimal_pool(order.get("size", 100))
        
        self.current_pool = target_pool
        
        fill_rate = self.pools[target_pool]["fill_rate"]
        latency = self.pools[target_pool]["latency"]
        
        filled_size = order.get("size", 100) * fill_rate
        
        if fill_rate < 0.7:
            self.logger.warning(f"Low fill rate ({fill_rate:.2f}) detected, scheduling failover")
            self.execute_failover(reason="low_fill_rate")
        
        return {
            "pool": target_pool,
            "order_id": f"DP-{int(time.time())}-{random.randint(1000, 9999)}",
            "filled_size": filled_size,
            "fill_rate": fill_rate,
            "latency": latency,
            "timestamp": datetime.now().timestamp()
        }
