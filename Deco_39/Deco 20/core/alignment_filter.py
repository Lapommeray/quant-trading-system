
import pandas as pd
import numpy as np

def is_fully_aligned(timestamp, alignment_df=None, history_data=None):
    """
    Checks if all candles in the 25-minute block are aligned in the same direction.
    
    Input:
        timestamp: datetime of the 25-minute block
        alignment_df: pandas DataFrame with ['Time', 'Label'] (from CSV)
        history_data: Dictionary of DataFrames for different timeframes (real-time data)
    
    Output:
        True if aligned, else False
    """
    if alignment_df is not None and not alignment_df.empty:
        row = alignment_df[alignment_df["Time"] == timestamp]
        if not row.empty:
            return row.iloc[0]["Label"] == "ALIGNED"
    
    if history_data is not None:
        timeframes = ['1m', '5m', '10m', '15m', '20m', '25m']
        
        if not all(tf in history_data for tf in timeframes):
            return False
            
        directions = []
        
        for tf in timeframes:
            df = history_data[tf]
            candles = df[df.index <= timestamp].tail(1)
            if candles.empty:
                return False
                
            candle = candles.iloc[0]
            direction = candle['Close'] > candle['Open']
            directions.append(direction)
            
        return all(d == directions[0] for d in directions)
            
    return False
