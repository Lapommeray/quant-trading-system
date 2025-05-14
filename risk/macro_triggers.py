"""
Macro Triggers Module

This module implements macro-economic triggers for the Quantum Trading System,
including inverse yield curve auto-exit logic and other macro risk indicators.

Dependencies:
- numpy
- pandas
- matplotlib (optional, for plotting)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
import json
import time
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('macro_triggers.log')
    ]
)

logger = logging.getLogger("MacroTriggers")

class YieldCurveMonitor:
    """
    Monitor for yield curve inversions and other yield curve anomalies.
    Provides auto-exit logic for bonds when yield curve inversions persist.
    """
    
    def __init__(
        self,
        inversion_threshold_days: int = 5,
        check_interval_hours: int = 6,
        api_key: Optional[str] = None
    ):
        """
        Initialize the yield curve monitor.
        
        Parameters:
        - inversion_threshold_days: Number of days inversion must persist to trigger exit
        - check_interval_hours: Interval between checks
        - api_key: API key for data source
        """
        self.inversion_threshold_days = inversion_threshold_days
        self.check_interval_hours = check_interval_hours
        self.api_key = api_key
        
        self.yield_data = {}
        self.spreads = {}
        self.inversion_history = {}
        
        self.monitoring = False
        self.monitor_thread = None
        self.last_update = None
        
        self.active_exit_signals = {}
        self.exit_signal_history = []
        
        self.key_spreads = [
            ("2y", "10y"),  # 2-year vs 10-year (most common)
            ("3m", "10y"),  # 3-month vs 10-year (Fed's preferred)
            ("2y", "30y"),  # 2-year vs 30-year (long-term)
            ("5y", "30y"),  # 5-year vs 30-year (medium-term)
            ("1y", "5y")    # 1-year vs 5-year (short-term)
        ]
        
        self._initialize_yield_data()
        
        logger.info("YieldCurveMonitor initialized")
        
    def _initialize_yield_data(self) -> None:
        """Initialize yield data structure"""
        tenors = ["3m", "6m", "1y", "2y", "5y", "10y", "30y"]
        
        for tenor in tenors:
            self.yield_data[tenor] = {
                "current": None,
                "previous": None,
                "history": [],
                "last_updated": None
            }
            
        for short_tenor, long_tenor in self.key_spreads:
            spread_key = f"{short_tenor}_{long_tenor}"
            self.spreads[spread_key] = {
                "current": None,
                "history": [],
                "inversion": False,
                "inversion_days": 0,
                "last_updated": None
            }
            
            self.inversion_history[spread_key] = []
            
    def start_monitoring(self) -> bool:
        """
        Start monitoring yield curve inversions.
        
        Returns:
        - Success status
        """
        if self.monitoring:
            logger.warning("Monitoring already active")
            return False
            
        try:
            self.monitoring = True
            
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("Started yield curve monitoring")
            return True
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            self.monitoring = False
            return False
            
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring yield curve inversions.
        
        Returns:
        - Success status
        """
        if not self.monitoring:
            logger.warning("Monitoring not active")
            return False
            
        try:
            self.monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
            logger.info("Stopped yield curve monitoring")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            return False
            
    def _monitor_loop(self) -> None:
        """Background thread for continuous monitoring"""
        while self.monitoring:
            try:
                self.update_yield_data()
                
                self._calculate_spreads()
                
                self._check_inversions()
                
                self._generate_exit_signals()
                
                time.sleep(self.check_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(self.check_interval_hours * 3600)
                
    def update_yield_data(self) -> bool:
        """
        Update yield curve data from the data source.
        
        Returns:
        - Success status
        """
        try:
            tenors = ["3m", "6m", "1y", "2y", "5y", "10y", "30y"]
            
            for tenor in tenors:
                if tenor == "3m":
                    base_yield = 4.5
                elif tenor == "6m":
                    base_yield = 4.6
                elif tenor == "1y":
                    base_yield = 4.7
                elif tenor == "2y":
                    base_yield = 4.8
                elif tenor == "5y":
                    base_yield = 4.7  # Slight inversion at 5y
                elif tenor == "10y":
                    base_yield = 4.6  # Inversion at 10y
                else:  # 30y
                    base_yield = 4.7
                    
                np.random.seed(42 + ord(tenor[0]))
                current_yield = base_yield + np.random.normal(0, 0.1)
                previous_yield = base_yield + np.random.normal(0, 0.1)
                
                self.yield_data[tenor]["current"] = current_yield
                self.yield_data[tenor]["previous"] = previous_yield
                self.yield_data[tenor]["last_updated"] = datetime.now()
                
                current_date = datetime.now()
                dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
                yields = [base_yield + np.random.normal(0, 0.1) for _ in range(30)]
                
                history = [{"date": date, "value": yield_val} for date, yield_val in zip(dates, yields)]
                self.yield_data[tenor]["history"] = history
                
            self.last_update = datetime.now()
            logger.info("Yield data updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating yield data: {str(e)}")
            return False
            
    def _calculate_spreads(self) -> None:
        """Calculate spreads between different tenors"""
        try:
            for short_tenor, long_tenor in self.key_spreads:
                spread_key = f"{short_tenor}_{long_tenor}"
                
                if (self.yield_data[short_tenor]["current"] is not None and 
                    self.yield_data[long_tenor]["current"] is not None):
                    
                    spread = self.yield_data[long_tenor]["current"] - self.yield_data[short_tenor]["current"]
                    
                    self.spreads[spread_key]["current"] = spread
                    
                    timestamp = datetime.now()
                    self.spreads[spread_key]["history"].append({
                        "timestamp": timestamp,
                        "spread": spread
                    })
                    
                    if len(self.spreads[spread_key]["history"]) > 100:
                        self.spreads[spread_key]["history"] = self.spreads[spread_key]["history"][-100:]
                        
                    self.spreads[spread_key]["last_updated"] = timestamp
                    
                    logger.debug(f"Spread {spread_key}: {spread:.4f}")
        except Exception as e:
            logger.error(f"Error calculating spreads: {str(e)}")
            
    def _check_inversions(self) -> None:
        """Check for yield curve inversions"""
        try:
            for short_tenor, long_tenor in self.key_spreads:
                spread_key = f"{short_tenor}_{long_tenor}"
                
                if self.spreads[spread_key]["current"] is not None:
                    is_inverted = self.spreads[spread_key]["current"] < 0
                    
                    if is_inverted:
                        if not self.spreads[spread_key]["inversion"]:
                            self.spreads[spread_key]["inversion"] = True
                            self.spreads[spread_key]["inversion_start"] = datetime.now()
                            
                            self.inversion_history[spread_key].append({
                                "start": datetime.now(),
                                "end": None,
                                "duration_days": 0,
                                "min_spread": self.spreads[spread_key]["current"],
                                "resolved": False
                            })
                            
                            logger.info(f"Yield curve inversion detected: {spread_key}")
                        else:
                            inversion_start = self.spreads[spread_key]["inversion_start"]
                            duration = (datetime.now() - inversion_start).total_seconds() / 86400  # days
                            
                            self.spreads[spread_key]["inversion_days"] = duration
                            
                            if self.inversion_history[spread_key]:
                                current_inversion = self.inversion_history[spread_key][-1]
                                current_inversion["duration_days"] = duration
                                current_inversion["min_spread"] = min(
                                    current_inversion["min_spread"],
                                    self.spreads[spread_key]["current"]
                                )
                    else:
                        if self.spreads[spread_key]["inversion"]:
                            self.spreads[spread_key]["inversion"] = False
                            
                            if self.inversion_history[spread_key]:
                                current_inversion = self.inversion_history[spread_key][-1]
                                current_inversion["end"] = datetime.now()
                                current_inversion["resolved"] = True
                                
                                logger.info(f"Yield curve inversion resolved: {spread_key}, duration: {current_inversion['duration_days']:.2f} days")
                                
                            self.spreads[spread_key]["inversion_days"] = 0
        except Exception as e:
            logger.error(f"Error checking inversions: {str(e)}")
            
    def _generate_exit_signals(self) -> None:
        """Generate exit signals based on inversion duration"""
        try:
            for short_tenor, long_tenor in self.key_spreads:
                spread_key = f"{short_tenor}_{long_tenor}"
                
                if (self.spreads[spread_key]["inversion"] and 
                    self.spreads[spread_key]["inversion_days"] >= self.inversion_threshold_days):
                    
                    if spread_key not in self.active_exit_signals:
                        signal = {
                            "spread_key": spread_key,
                            "short_tenor": short_tenor,
                            "long_tenor": long_tenor,
                            "spread": self.spreads[spread_key]["current"],
                            "inversion_days": self.spreads[spread_key]["inversion_days"],
                            "timestamp": datetime.now(),
                            "status": "active"
                        }
                        
                        self.active_exit_signals[spread_key] = signal
                        self.exit_signal_history.append(signal)
                        
                        logger.warning(f"Bond exit signal generated: {spread_key}, inversion: {self.spreads[spread_key]['inversion_days']:.2f} days")
                elif spread_key in self.active_exit_signals:
                    self.active_exit_signals[spread_key]["status"] = "resolved"
                    self.active_exit_signals[spread_key]["resolved_at"] = datetime.now()
                    
                    del self.active_exit_signals[spread_key]
                    
                    logger.info(f"Bond exit signal resolved: {spread_key}")
        except Exception as e:
            logger.error(f"Error generating exit signals: {str(e)}")
            
    def get_active_exit_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active exit signals.
        
        Returns:
        - Dictionary of active exit signals
        """
        return self.active_exit_signals
        
    def get_exit_signal_history(self) -> List[Dict[str, Any]]:
        """
        Get exit signal history.
        
        Returns:
        - List of exit signals
        """
        return self.exit_signal_history
        
    def get_inversion_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current inversion status for all spreads.
        
        Returns:
        - Dictionary with inversion status
        """
        status = {}
        
        for short_tenor, long_tenor in self.key_spreads:
            spread_key = f"{short_tenor}_{long_tenor}"
            
            status[spread_key] = {
                "spread": self.spreads[spread_key]["current"],
                "inverted": self.spreads[spread_key]["inversion"],
                "duration_days": self.spreads[spread_key]["inversion_days"],
                "last_updated": self.spreads[spread_key]["last_updated"]
            }
            
        return status
        
    def get_yield_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current yield data for all tenors.
        
        Returns:
        - Dictionary with yield data
        """
        data = {}
        
        for tenor, tenor_data in self.yield_data.items():
            data[tenor] = {
                "current": tenor_data["current"],
                "previous": tenor_data["previous"],
                "last_updated": tenor_data["last_updated"]
            }
            
        return data

class MacroRiskMonitor:
    """
    Monitor for various macro-economic risk factors.
    Integrates multiple risk indicators including yield curve inversions.
    """
    
    def __init__(
        self,
        check_interval_hours: int = 6,
        risk_threshold: float = 0.7
    ):
        """
        Initialize the macro risk monitor.
        
        Parameters:
        - check_interval_hours: Interval between checks
        - risk_threshold: Threshold for high risk
        """
        self.check_interval_hours = check_interval_hours
        self.risk_threshold = risk_threshold
        
        self.risk_indicators = {
            "yield_curve": {
                "monitor": YieldCurveMonitor(),
                "weight": 0.4,
                "current_risk": 0.0,
                "last_updated": None
            },
            "vix": {
                "current": None,
                "threshold": 30.0,
                "weight": 0.3,
                "current_risk": 0.0,
                "last_updated": None
            },
            "credit_spreads": {
                "current": None,
                "threshold": 5.0,
                "weight": 0.2,
                "current_risk": 0.0,
                "last_updated": None
            },
            "liquidity": {
                "current": None,
                "threshold": 700.0,  # $700B in reverse repo
                "weight": 0.1,
                "current_risk": 0.0,
                "last_updated": None
            }
        }
        
        self.overall_risk = 0.0
        self.risk_history = []
        
        self.monitoring = False
        self.monitor_thread = None
        self.last_update = None
        
        self.active_risk_signals = {}
        self.risk_signal_history = []
        
        logger.info("MacroRiskMonitor initialized")
        
    def start_monitoring(self) -> bool:
        """
        Start monitoring macro risks.
        
        Returns:
        - Success status
        """
        if self.monitoring:
            logger.warning("Monitoring already active")
            return False
            
        try:
            self.monitoring = True
            
            self.risk_indicators["yield_curve"]["monitor"].start_monitoring()
            
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("Started macro risk monitoring")
            return True
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            self.monitoring = False
            return False
            
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring macro risks.
        
        Returns:
        - Success status
        """
        if not self.monitoring:
            logger.warning("Monitoring not active")
            return False
            
        try:
            self.monitoring = False
            
            self.risk_indicators["yield_curve"]["monitor"].stop_monitoring()
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
            logger.info("Stopped macro risk monitoring")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            return False
            
    def _monitor_loop(self) -> None:
        """Background thread for continuous monitoring"""
        while self.monitoring:
            try:
                self._update_risk_indicators()
                
                self._calculate_overall_risk()
                
                self._generate_risk_signals()
                
                time.sleep(self.check_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(self.check_interval_hours * 3600)
                
    def _update_risk_indicators(self) -> None:
        """Update all risk indicators"""
        try:
            yield_curve_monitor = self.risk_indicators["yield_curve"]["monitor"]
            inversion_status = yield_curve_monitor.get_inversion_status()
            
            inversion_count = sum(1 for status in inversion_status.values() if status["inverted"])
            max_duration = max((status["duration_days"] for status in inversion_status.values()), default=0)
            
            yield_curve_risk = min(1.0, (inversion_count / len(inversion_status)) * 0.5 + (max_duration / 30) * 0.5)
            
            self.risk_indicators["yield_curve"]["current_risk"] = yield_curve_risk
            self.risk_indicators["yield_curve"]["last_updated"] = datetime.now()
            
            vix_value = 20.0 + np.random.normal(0, 5)
            vix_risk = min(1.0, max(0.0, vix_value / self.risk_indicators["vix"]["threshold"]))
            
            self.risk_indicators["vix"]["current"] = vix_value
            self.risk_indicators["vix"]["current_risk"] = vix_risk
            self.risk_indicators["vix"]["last_updated"] = datetime.now()
            
            credit_spread = 3.0 + np.random.normal(0, 1)
            credit_risk = min(1.0, max(0.0, credit_spread / self.risk_indicators["credit_spreads"]["threshold"]))
            
            self.risk_indicators["credit_spreads"]["current"] = credit_spread
            self.risk_indicators["credit_spreads"]["current_risk"] = credit_risk
            self.risk_indicators["credit_spreads"]["last_updated"] = datetime.now()
            
            liquidity_value = 500.0 + np.random.normal(0, 100)
            liquidity_risk = min(1.0, max(0.0, liquidity_value / self.risk_indicators["liquidity"]["threshold"]))
            
            self.risk_indicators["liquidity"]["current"] = liquidity_value
            self.risk_indicators["liquidity"]["current_risk"] = liquidity_risk
            self.risk_indicators["liquidity"]["last_updated"] = datetime.now()
            
            logger.info("Risk indicators updated")
        except Exception as e:
            logger.error(f"Error updating risk indicators: {str(e)}")
            
    def _calculate_overall_risk(self) -> None:
        """Calculate overall risk based on weighted indicators"""
        try:
            weighted_risks = [
                indicator["current_risk"] * indicator["weight"]
                for indicator in self.risk_indicators.values()
            ]
            
            self.overall_risk = sum(weighted_risks)
            
            self.risk_history.append({
                "timestamp": datetime.now(),
                "overall_risk": self.overall_risk,
                "indicators": {
                    name: {
                        "current_risk": indicator["current_risk"],
                        "weight": indicator["weight"]
                    }
                    for name, indicator in self.risk_indicators.items()
                }
            })
            
            if len(self.risk_history) > 100:
                self.risk_history = self.risk_history[-100:]
                
            self.last_update = datetime.now()
            
            logger.info(f"Overall risk calculated: {self.overall_risk:.4f}")
        except Exception as e:
            logger.error(f"Error calculating overall risk: {str(e)}")
            
    def _generate_risk_signals(self) -> None:
        """Generate risk signals based on risk levels"""
        try:
            if self.overall_risk >= self.risk_threshold:
                if "high_risk" not in self.active_risk_signals:
                    signal = {
                        "type": "high_risk",
                        "overall_risk": self.overall_risk,
                        "indicators": {
                            name: indicator["current_risk"]
                            for name, indicator in self.risk_indicators.items()
                        },
                        "timestamp": datetime.now(),
                        "status": "active"
                    }
                    
                    self.active_risk_signals["high_risk"] = signal
                    self.risk_signal_history.append(signal)
                    
                    logger.warning(f"High risk signal generated: {self.overall_risk:.4f}")
            elif "high_risk" in self.active_risk_signals:
                self.active_risk_signals["high_risk"]["status"] = "resolved"
                self.active_risk_signals["high_risk"]["resolved_at"] = datetime.now()
                
                del self.active_risk_signals["high_risk"]
                
                logger.info("High risk signal resolved")
                
            for name, indicator in self.risk_indicators.items():
                if name == "yield_curve":
                    continue
                    
                if indicator["current_risk"] >= 0.8:  # 80% of max risk
                    signal_key = f"{name}_high_risk"
                    
                    if signal_key not in self.active_risk_signals:
                        signal = {
                            "type": signal_key,
                            "indicator": name,
                            "risk": indicator["current_risk"],
                            "value": indicator["current"],
                            "timestamp": datetime.now(),
                            "status": "active"
                        }
                        
                        self.active_risk_signals[signal_key] = signal
                        self.risk_signal_history.append(signal)
                        
                        logger.warning(f"{name} high risk signal generated: {indicator['current_risk']:.4f}")
                elif f"{name}_high_risk" in self.active_risk_signals:
                    self.active_risk_signals[f"{name}_high_risk"]["status"] = "resolved"
                    self.active_risk_signals[f"{name}_high_risk"]["resolved_at"] = datetime.now()
                    
                    del self.active_risk_signals[f"{name}_high_risk"]
                    
                    logger.info(f"{name} high risk signal resolved")
        except Exception as e:
            logger.error(f"Error generating risk signals: {str(e)}")
            
    def get_active_risk_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active risk signals.
        
        Returns:
        - Dictionary of active risk signals
        """
        yield_curve_signals = self.risk_indicators["yield_curve"]["monitor"].get_active_exit_signals()
        
        for key, signal in yield_curve_signals.items():
            self.active_risk_signals[f"yield_curve_{key}"] = {
                "type": "yield_curve_inversion",
                "spread_key": signal["spread_key"],
                "spread": signal["spread"],
                "inversion_days": signal["inversion_days"],
                "timestamp": signal["timestamp"],
                "status": "active"
            }
            
        return self.active_risk_signals
        
    def get_risk_signal_history(self) -> List[Dict[str, Any]]:
        """
        Get risk signal history.
        
        Returns:
        - List of risk signals
        """
        return self.risk_signal_history
        
    def get_risk_status(self) -> Dict[str, Any]:
        """
        Get current risk status.
        
        Returns:
        - Dictionary with risk status
        """
        return {
            "overall_risk": self.overall_risk,
            "indicators": {
                name: {
                    "current_risk": indicator["current_risk"],
                    "current_value": indicator.get("current"),
                    "last_updated": indicator["last_updated"]
                }
                for name, indicator in self.risk_indicators.items()
            },
            "last_updated": self.last_update
        }
