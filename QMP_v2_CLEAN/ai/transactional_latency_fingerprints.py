"""
Transactional Latency Fingerprints Module

This module implements the Transactional Latency Fingerprints (TLF) system for the QMP Overrider strategy.
It detects and analyzes millisecond-level delay patterns in market transactions to identify hidden liquidity.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats

class TransactionalLatencyFingerprints:
    """
    Transactional Latency Fingerprints for the QMP Overrider system.
    
    This class detects and analyzes millisecond-level delay patterns in market transactions,
    providing a comprehensive system for identifying hidden liquidity and smart money movements.
    """
    
    def __init__(self, log_dir=None, anomaly_threshold=2.5):
        """
        Initialize the Transactional Latency Fingerprints system.
        
        Parameters:
        - log_dir: Directory to store latency logs (or None for default)
        - anomaly_threshold: Z-score threshold for latency anomaly detection
        """
        self.logger = logging.getLogger("TransactionalLatencyFingerprints")
        
        if log_dir is None:
            self.log_dir = Path("logs/latency_fingerprints")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.anomaly_threshold = anomaly_threshold
        
        self.symbol_latency_data = {}
        self.latency_history = {}
        self.latency_anomalies = []
        self.venue_fingerprints = {}
        
        self.venue_types = {
            "lit_exchange": ["NYSE", "NASDAQ", "AMEX", "ARCA"],
            "dark_pool": ["SIGMA", "UBS", "JPMX", "MS", "CITI", "BAML"],
            "market_maker": ["CITADEL", "VIRTU", "JANE", "SUSQ", "TWO", "FLOW"],
            "retail": ["HOOD", "IBKR", "ETRADE", "SCHWAB", "FIDELITY"]
        }
        
        self._load_data()
        
        self.logger.info(f"Transactional Latency Fingerprints initialized with threshold: {anomaly_threshold}")
    
    def _load_data(self):
        """Load existing latency data"""
        latency_file = self.log_dir / "latency_data.json"
        history_file = self.log_dir / "latency_history.json"
        anomalies_file = self.log_dir / "latency_anomalies.json"
        fingerprints_file = self.log_dir / "venue_fingerprints.json"
        
        if latency_file.exists():
            try:
                with open(latency_file, "r") as f:
                    self.symbol_latency_data = json.load(f)
                
                self.logger.info(f"Loaded latency data for {len(self.symbol_latency_data)} symbols")
            except Exception as e:
                self.logger.error(f"Error loading latency data: {e}")
        
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.latency_history = json.load(f)
                
                self.logger.info(f"Loaded latency history for {len(self.latency_history)} symbols")
            except Exception as e:
                self.logger.error(f"Error loading latency history: {e}")
        
        if anomalies_file.exists():
            try:
                with open(anomalies_file, "r") as f:
                    self.latency_anomalies = json.load(f)
                
                self.logger.info(f"Loaded {len(self.latency_anomalies)} latency anomalies")
            except Exception as e:
                self.logger.error(f"Error loading latency anomalies: {e}")
        
        if fingerprints_file.exists():
            try:
                with open(fingerprints_file, "r") as f:
                    self.venue_fingerprints = json.load(f)
                
                self.logger.info(f"Loaded fingerprints for {len(self.venue_fingerprints)} venues")
            except Exception as e:
                self.logger.error(f"Error loading venue fingerprints: {e}")
    
    def _save_data(self):
        """Save latency data to file"""
        latency_file = self.log_dir / "latency_data.json"
        history_file = self.log_dir / "latency_history.json"
        anomalies_file = self.log_dir / "latency_anomalies.json"
        fingerprints_file = self.log_dir / "venue_fingerprints.json"
        
        try:
            with open(latency_file, "w") as f:
                json.dump(self.symbol_latency_data, f, indent=2)
            
            with open(history_file, "w") as f:
                json.dump(self.latency_history, f, indent=2)
            
            with open(anomalies_file, "w") as f:
                json.dump(self.latency_anomalies, f, indent=2)
            
            with open(fingerprints_file, "w") as f:
                json.dump(self.venue_fingerprints, f, indent=2)
            
            self.logger.info("Saved latency data")
        except Exception as e:
            self.logger.error(f"Error saving latency data: {e}")
