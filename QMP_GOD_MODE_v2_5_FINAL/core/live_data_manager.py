import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class LiveDataManager:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.data_cache = {}
        self.last_update = {}
        
    def get_live_alignment_data(self, timestamp, symbol, history_data):
        """Generate live alignment data instead of using static CSV"""
        timeframes = ['1m', '5m', '10m', '15m', '20m', '25m']
        
        if not all(tf in history_data for tf in timeframes):
            return False
            
        directions = []
        
        for tf in timeframes:
            df = history_data[tf]
            if df.empty:
                return False
                
            recent_candles = df[df.index <= timestamp].tail(1)
            if recent_candles.empty:
                return False
                
            candle = recent_candles.iloc[0]
            direction = candle['Close'] > candle['Open']
            directions.append(direction)
            
        alignment = all(d == directions[0] for d in directions)
        
        self._log_alignment_data(timestamp, symbol, directions, alignment)
        
        return alignment
        
    def _log_alignment_data(self, timestamp, symbol, directions, alignment):
        """Log alignment data to file for analysis"""
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'symbol': str(symbol),
            'directions': directions,
            'aligned': alignment,
            'timeframes': ['1m', '5m', '10m', '15m', '20m', '25m']
        }
        
        log_path = os.path.join(self.algorithm.DataFolder, "data", "live_alignment_log.json")
        
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
                
            log_data.append(log_entry)
            
            if len(log_data) > 1000:
                log_data = log_data[-1000:]
                
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.algorithm.Debug(f"Error logging alignment data: {e}")
