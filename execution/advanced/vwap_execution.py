"""
VWAP Execution Algorithm

Volume-Weighted Average Price execution strategy for institutional trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

class VWAPExecution:
    """
    Volume-Weighted Average Price execution algorithm.
    """
    
    def __init__(self, historical_volumes: pd.Series):
        self.volume_profile = self._calculate_profile(historical_volumes)
        self.logger = logging.getLogger('VWAPExecution')
        
    def _calculate_profile(self, volumes: pd.Series) -> pd.Series:
        """Calculate typical volume profile by time of day"""
        return volumes.groupby(volumes.index.time).mean()
    
    def get_execution_schedule(self, target_quantity: float, start_time: str, end_time: str, 
                             time_slice_minutes: int = 5) -> Dict[str, float]:
        """
        Generate VWAP execution schedule.
        
        Parameters:
        - target_quantity: Total quantity to execute
        - start_time: Start time (HH:MM format)
        - end_time: End time (HH:MM format)  
        - time_slice_minutes: Minutes per execution slice
        
        Returns:
        - Dictionary mapping time to quantities
        """
        try:
            start = pd.to_datetime(start_time, format='%H:%M').time()
            end = pd.to_datetime(end_time, format='%H:%M').time()
            
            time_range = pd.date_range(
                start=pd.datetime.combine(pd.datetime.today(), start),
                end=pd.datetime.combine(pd.datetime.today(), end),
                freq=f'{time_slice_minutes}min'
            )
            
            time_slices = [t.time() for t in time_range]
            
            slice_volumes = []
            for t in time_slices:
                if t in self.volume_profile.index:
                    slice_volumes.append(self.volume_profile.loc[t])
                else:
                    nearest_times = self.volume_profile.index
                    closest_time = min(nearest_times, key=lambda x: abs(
                        pd.datetime.combine(pd.datetime.today(), x) - 
                        pd.datetime.combine(pd.datetime.today(), t)
                    ).total_seconds())
                    slice_volumes.append(self.volume_profile.loc[closest_time])
            
            total_volume = sum(slice_volumes)
            if total_volume == 0:
                self.logger.warning("No volume data available for time range")
                return {}
            
            schedule = {}
            for i, t in enumerate(time_slices):
                proportion = slice_volumes[i] / total_volume
                schedule[t.strftime('%H:%M')] = target_quantity * proportion
                
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error creating VWAP schedule: {str(e)}")
            return {}
            
    def calculate_vwap_benchmark(self, prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate VWAP benchmark for performance measurement"""
        if len(prices) != len(volumes) or len(prices) == 0:
            return np.nan
            
        return np.sum(prices * volumes) / np.sum(volumes)
