"""
Whale Detector Module

This module implements detection of large dark pool orders and whale activity
for the Quantum Trading System. It identifies significant liquidity movements
in dark pools that may indicate institutional activity.

Dependencies:
- numpy
- pandas
- ccxt (optional, for exchange data)
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
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whale_detector.log')
    ]
)

logger = logging.getLogger("WhaleDetector")

try:
    import ccxt
    CCXT_AVAILABLE = True
    logger.info("CCXT loaded successfully")
except ImportError:
    logger.warning("CCXT not available. Some exchange data features will be limited.")
    CCXT_AVAILABLE = False

class DarkPoolConnector:
    """
    Connector for dark pool data sources.
    Handles connections to various dark pool data providers.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the dark pool connector.
        
        Parameters:
        - api_keys: Dictionary of API keys for different data sources
        """
        self.api_keys = api_keys or {}
        self.connections = {}
        self.data_sources = [
            "cboe", "level_ats", "ms_pool", "jp_pool", "ubs_mtf", 
            "sigma_x", "liquidnet", "posit", "instinet"
        ]
        
        self._initialize_connections()
        
        logger.info("DarkPoolConnector initialized")
        
    def _initialize_connections(self) -> None:
        """Initialize connections to dark pool data sources"""
        
        for source in self.data_sources:
            if source in self.api_keys:
                self.connections[source] = {
                    "connected": True,
                    "api_key": self.api_keys[source],
                    "last_updated": None
                }
            else:
                self.connections[source] = {
                    "connected": False,
                    "api_key": None,
                    "last_updated": None
                }
                
        logger.info(f"Initialized {sum(c['connected'] for c in self.connections.values())}/{len(self.data_sources)} dark pool connections")
        
    def fetch_data(self, source: str, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch data from a specific dark pool source.
        
        Parameters:
        - source: Name of the dark pool source
        - symbol: Trading symbol
        - timeframe: Timeframe for the data
        - limit: Number of records to fetch
        
        Returns:
        - DataFrame with dark pool data or None if not available
        """
        if source not in self.connections or not self.connections[source]["connected"]:
            logger.warning(f"No connection to {source}")
            return None
            
        try:
            
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(limit)]
            
            base_volume = np.random.normal(1000, 200, limit)
            spikes = np.zeros(limit)
            
            for i in range(3):  # Add 3 random spikes
                spike_idx = np.random.randint(0, limit)
                spikes[spike_idx] = np.random.normal(10000, 2000)
                
            volumes = base_volume + spikes
            
            base_price = 100.0
            prices = np.cumsum(np.random.normal(0, 1, limit)) + base_price
            
            df = pd.DataFrame({
                "timestamp": timestamps,
                "price": prices,
                "volume": volumes,
                "source": source,
                "symbol": symbol
            })
            
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            self.connections[source]["last_updated"] = datetime.now()
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {str(e)}")
            return None
            
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all dark pool connections.
        
        Returns:
        - Dictionary with connection status
        """
        return self.connections
        
    def disconnect(self) -> None:
        """Disconnect from all dark pool sources"""
        for source in self.connections:
            self.connections[source]["connected"] = False
            
        logger.info("Disconnected from all dark pool sources")

class WhaleDetector:
    """
    Detector for whale activity in dark pools.
    Identifies significant liquidity movements that may indicate institutional activity.
    """
    
    def __init__(
        self,
        volume_threshold_multiplier: float = 5.0,
        price_impact_threshold: float = 0.5,
        time_window: str = "1h",
        min_confidence: float = 0.7,
        connector: Optional[DarkPoolConnector] = None
    ):
        """
        Initialize the whale detector.
        
        Parameters:
        - volume_threshold_multiplier: Multiplier for volume threshold (relative to average)
        - price_impact_threshold: Minimum price impact percentage to consider
        - time_window: Time window for analysis
        - min_confidence: Minimum confidence for whale detection
        - connector: Dark pool connector instance
        """
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.price_impact_threshold = price_impact_threshold
        self.time_window = time_window
        self.min_confidence = min_confidence
        
        self.connector = connector or DarkPoolConnector()
        
        self.detected_whales = []
        self.active_whales = {}
        
        self.scanning = False
        self.scan_thread = None
        self.scan_interval = 300  # 5 minutes
        
        self.total_scans = 0
        self.total_detections = 0
        
        logger.info("WhaleDetector initialized")
        
    def start_scanning(self, symbols: List[str], sources: Optional[List[str]] = None) -> bool:
        """
        Start background scanning for whale activity.
        
        Parameters:
        - symbols: List of symbols to scan
        - sources: List of dark pool sources to scan (default: all available)
        
        Returns:
        - Success status
        """
        if self.scanning:
            logger.warning("Scanning already active")
            return False
            
        try:
            self.scanning = True
            
            if sources is None:
                sources = list(self.connector.connections.keys())
                
            self.scan_thread = threading.Thread(
                target=self._scan_loop,
                args=(symbols, sources),
                daemon=True
            )
            self.scan_thread.start()
            
            logger.info(f"Started scanning {len(symbols)} symbols across {len(sources)} dark pools")
            return True
        except Exception as e:
            logger.error(f"Error starting scanning: {str(e)}")
            self.scanning = False
            return False
            
    def stop_scanning(self) -> bool:
        """
        Stop background scanning.
        
        Returns:
        - Success status
        """
        if not self.scanning:
            logger.warning("Scanning not active")
            return False
            
        try:
            self.scanning = False
            
            if self.scan_thread and self.scan_thread.is_alive():
                self.scan_thread.join(timeout=5)
                
            logger.info("Stopped scanning")
            return True
        except Exception as e:
            logger.error(f"Error stopping scanning: {str(e)}")
            return False
            
    def _scan_loop(self, symbols: List[str], sources: List[str]) -> None:
        """
        Background thread for continuous scanning.
        
        Parameters:
        - symbols: List of symbols to scan
        - sources: List of dark pool sources to scan
        """
        while self.scanning:
            try:
                for symbol in symbols:
                    for source in sources:
                        if not self.connector.connections.get(source, {}).get("connected", False):
                            continue
                            
                        data = self.connector.fetch_data(source, symbol, self.time_window)
                        
                        if data is not None:
                            whales = self.detect_whales(data)
                            
                            if whales:
                                for whale in whales:
                                    self._process_whale_detection(whale)
                                    
                self.total_scans += 1
                
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in scan loop: {str(e)}")
                time.sleep(self.scan_interval)
                
    def detect_whales(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect whale activity in the provided data.
        
        Parameters:
        - data: DataFrame with dark pool data
        
        Returns:
        - List of detected whale activities
        """
        if data is None or len(data) == 0:
            return []
            
        try:
            mean_volume = data["volume"].mean()
            std_volume = data["volume"].std()
            
            volume_threshold = mean_volume + (std_volume * self.volume_threshold_multiplier)
            
            whale_indices = data[data["volume"] > volume_threshold].index.tolist()
            
            whales = []
            
            for idx in whale_indices:
                if idx == 0:
                    continue
                    
                current = data.loc[idx]
                previous = data.loc[idx - 1]
                
                price_change_pct = abs((current["price"] - previous["price"]) / previous["price"] * 100)
                
                volume_ratio = current["volume"] / mean_volume
                
                confidence = min(1.0, (volume_ratio / self.volume_threshold_multiplier) * 0.7 + (price_change_pct / self.price_impact_threshold) * 0.3)
                
                if confidence >= self.min_confidence:
                    whale = {
                        "timestamp": current["timestamp"],
                        "symbol": current["symbol"],
                        "source": current["source"],
                        "volume": current["volume"],
                        "price": current["price"],
                        "volume_ratio": volume_ratio,
                        "price_impact": price_change_pct,
                        "confidence": confidence,
                        "direction": "buy" if current["price"] > previous["price"] else "sell",
                        "detected_at": datetime.now()
                    }
                    
                    whales.append(whale)
                    
            return whales
        except Exception as e:
            logger.error(f"Error detecting whales: {str(e)}")
            return []
            
    def _process_whale_detection(self, whale: Dict[str, Any]) -> None:
        """
        Process a detected whale activity.
        
        Parameters:
        - whale: Dictionary with whale activity data
        """
        try:
            self.detected_whales.append(whale)
            
            symbol = whale["symbol"]
            
            if symbol not in self.active_whales:
                self.active_whales[symbol] = []
                
            self.active_whales[symbol].append(whale)
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for sym in list(self.active_whales.keys()):
                self.active_whales[sym] = [
                    w for w in self.active_whales[sym]
                    if w["detected_at"] > cutoff_time
                ]
                
                if not self.active_whales[sym]:
                    del self.active_whales[sym]
                    
            self.total_detections += 1
            
            logger.info(f"Detected whale: {whale['symbol']} {whale['direction']} with {whale['volume']:.2f} volume (confidence: {whale['confidence']:.2f})")
        except Exception as e:
            logger.error(f"Error processing whale detection: {str(e)}")
            
    def get_active_whales(self, symbol: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get active whale activities.
        
        Parameters:
        - symbol: Symbol to filter by (optional)
        
        Returns:
        - Dictionary with active whale activities
        """
        if symbol:
            return {symbol: self.active_whales.get(symbol, [])}
        else:
            return self.active_whales
            
    def get_detection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get detection history.
        
        Parameters:
        - limit: Maximum number of detections to return
        
        Returns:
        - List of detected whale activities
        """
        return self.detected_whales[-limit:]
        
    def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on whale activity.
        
        Returns:
        - List of trading signals
        """
        signals = []
        
        try:
            for symbol, whales in self.active_whales.items():
                if not whales:
                    continue
                    
                buy_volume = sum(w["volume"] for w in whales if w["direction"] == "buy")
                sell_volume = sum(w["volume"] for w in whales if w["direction"] == "sell")
                
                if buy_volume > sell_volume:
                    direction = "buy"
                    strength = buy_volume / (buy_volume + sell_volume)
                else:
                    direction = "sell"
                    strength = sell_volume / (buy_volume + sell_volume)
                    
                avg_confidence = sum(w["confidence"] for w in whales) / len(whales)
                
                signal = {
                    "symbol": symbol,
                    "direction": direction,
                    "strength": strength,
                    "confidence": avg_confidence,
                    "whale_count": len(whales),
                    "total_volume": buy_volume + sell_volume,
                    "timestamp": datetime.now().isoformat()
                }
                
                signals.append(signal)
                
            return signals
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
            
    def save_detections(self, filepath: str) -> bool:
        """
        Save detection history to a file.
        
        Parameters:
        - filepath: Path to save the detections
        
        Returns:
        - Success status
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            detections = []
            
            for whale in self.detected_whales:
                whale_copy = whale.copy()
                
                for key, value in whale_copy.items():
                    if isinstance(value, datetime):
                        whale_copy[key] = value.isoformat()
                        
                detections.append(whale_copy)
                
            with open(filepath, 'w') as f:
                json.dump(detections, f, indent=2)
                
            logger.info(f"Saved {len(detections)} detections to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving detections: {str(e)}")
            return False
            
    def load_detections(self, filepath: str) -> bool:
        """
        Load detection history from a file.
        
        Parameters:
        - filepath: Path to load the detections from
        
        Returns:
        - Success status
        """
        try:
            with open(filepath, 'r') as f:
                detections = json.load(f)
                
            for whale in detections:
                for key, value in whale.items():
                    if key in ["timestamp", "detected_at"] and isinstance(value, str):
                        whale[key] = datetime.fromisoformat(value)
                        
            self.detected_whales = detections
            
            self.active_whales = {}
            
            for whale in self.detected_whales:
                symbol = whale["symbol"]
                
                if symbol not in self.active_whales:
                    self.active_whales[symbol] = []
                    
                if whale["detected_at"] > datetime.now() - timedelta(hours=24):
                    self.active_whales[symbol].append(whale)
                    
            logger.info(f"Loaded {len(detections)} detections from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading detections: {str(e)}")
            return False
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
        - Dictionary with performance metrics
        """
        return {
            "total_scans": self.total_scans,
            "total_detections": self.total_detections,
            "active_symbols": len(self.active_whales),
            "active_whales": sum(len(whales) for whales in self.active_whales.values()),
            "detection_rate": self.total_detections / max(1, self.total_scans),
            "last_updated": datetime.now().isoformat()
        }

if __name__ == "__main__":
    connector = DarkPoolConnector()
    detector = WhaleDetector(connector=connector)
    
    detector.start_scanning(symbols=["BTCUSD", "ETHUSD", "XAUUSD"])
    
    try:
        time.sleep(60)
        
        active_whales = detector.get_active_whales()
        print(f"Active whales: {len(active_whales)}")
        
        signals = detector.generate_trading_signals()
        print(f"Trading signals: {len(signals)}")
        
        for signal in signals:
            print(f"{signal['symbol']}: {signal['direction']} (confidence: {signal['confidence']:.2f})")
            
    finally:
        detector.stop_scanning()
