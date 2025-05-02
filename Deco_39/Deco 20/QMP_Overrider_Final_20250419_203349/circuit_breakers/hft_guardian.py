"""
HFT Circuit Breaker Guardian Module

This module implements the HFT Circuit Breaker Guardian for the QMP Overrider system.
It provides real-time protection against market anomalies and volatility spikes.
"""

from dataclasses import dataclass
import time
import logging
import json
from pathlib import Path

@dataclass
class BreakerConfig:
    """
    Configuration for the HFT Circuit Breaker.
    
    Attributes:
        volatility_threshold: Threshold for volatility-based circuit breaking (0.0 to 1.0)
        latency_spike_ms: Threshold for latency spike detection in milliseconds
        order_imbalance_ratio: Threshold for order book imbalance detection
        cooling_off_period: Cooling period after circuit breaker activation in seconds
    """
    volatility_threshold: float = 0.08  # 8%
    latency_spike_ms: int = 50
    order_imbalance_ratio: float = 3.0
    cooling_off_period: int = 60  # seconds

class HFTCircuitBreaker:
    """
    High-Frequency Trading Circuit Breaker for the QMP Overrider system.
    
    This class implements a sophisticated circuit breaker that monitors market conditions
    and automatically halts trading when anomalies are detected.
    """
    
    def __init__(self, config=None):
        """
        Initialize the HFT Circuit Breaker with configuration.
        
        Parameters:
            config: BreakerConfig object with circuit breaker parameters
        """
        self.config = config or BreakerConfig()
        self._tripped = False
        self._last_trip = 0
        self.logger = logging.getLogger("HFTCircuitBreaker")
        
        self.log_dir = Path("logs/circuit_breakers")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trip_history = []
        self._load_history()
        
        self.logger.info(f"HFT Circuit Breaker initialized with config: {self.config}")
    
    def _load_history(self):
        """Load trip history from file"""
        history_file = self.log_dir / "trip_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.trip_history = json.load(f)
                self.logger.info(f"Loaded {len(self.trip_history)} historical circuit breaker trips")
            except Exception as e:
                self.logger.error(f"Error loading trip history: {e}")
    
    def _save_history(self):
        """Save trip history to file"""
        history_file = self.log_dir / "trip_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.trip_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trip history: {e}")
    
    def check_conditions(self, market_data):
        """
        Check if circuit breaker conditions are met.
        
        Parameters:
            market_data: Market data object with volatility, latency, and bid_ask_imbalance
            
        Returns:
            Boolean indicating if circuit breaker should be triggered
        """
        if time.time() - self._last_trip < self.config.cooling_off_period:
            return False
        
        trip_signals = [
            market_data.volatility > self.config.volatility_threshold,
            market_data.latency > self.config.latency_spike_ms,
            market_data.bid_ask_imbalance > self.config.order_imbalance_ratio
        ]
        
        if any(trip_signals) and not all(trip_signals):
            self.logger.warning(f"Near-trip condition: volatility={market_data.volatility}, "
                               f"latency={market_data.latency}ms, "
                               f"imbalance={market_data.bid_ask_imbalance}")
        
        return any(trip_signals)
    
    def trigger(self, strategy):
        """
        Trigger the circuit breaker.
        
        Parameters:
            strategy: Trading strategy object to apply circuit breaker actions to
        """
        if not self._tripped:
            self.logger.critical(f"[!] HFT CIRCUIT BREAKER ACTIVATED")
            
            strategy.cancel_all_orders()
            strategy.enable_risk_limits()
            
            trip_record = {
                "timestamp": time.time(),
                "config": {
                    "volatility_threshold": self.config.volatility_threshold,
                    "latency_spike_ms": self.config.latency_spike_ms,
                    "order_imbalance_ratio": self.config.order_imbalance_ratio
                }
            }
            self.trip_history.append(trip_record)
            self._save_history()
            
            self._tripped = True
            self._last_trip = time.time()
            
            self._notify_post_mortem(trip_record)
    
    def _notify_post_mortem(self, trip_record):
        """
        Notify the post-mortem system of a circuit breaker trip.
        
        Parameters:
            trip_record: Record of the circuit breaker trip
        """
        try:
            from post_mortem.analyzer import PostMortemEngine
            analyzer = PostMortemEngine()
            report = analyzer.generate_report(trip_record["timestamp"])
            
            report_file = self.log_dir / f"post_mortem_{int(trip_record['timestamp'])}.txt"
            with open(report_file, "w") as f:
                f.write(report)
            
            self.logger.info(f"Post-mortem report generated: {report_file}")
        except ImportError:
            self.logger.warning("Post-mortem analyzer not available")
        except Exception as e:
            self.logger.error(f"Error generating post-mortem report: {e}")
    
    def reset(self):
        """Reset the circuit breaker state."""
        if self._tripped:
            self.logger.info("Circuit breaker reset")
            self._tripped = False
    
    @property
    def is_tripped(self):
        """Get the current circuit breaker state."""
        return self._tripped
    
    @property
    def cooling_time_remaining(self):
        """Get the remaining cooling time in seconds."""
        if not self._tripped:
            return 0
        
        elapsed = time.time() - self._last_trip
        remaining = max(0, self.config.cooling_off_period - elapsed)
        return remaining
