"""
COVID Data Simulator

This module provides functions to simulate market data during the COVID crash period.
It generates realistic price movements with volatility patterns similar to the actual
COVID crash for testing the trading system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any

def simulate_crash(
    asset: str,
    start_date: str = "2020-02-15",
    end_date: str = "2020-04-15",
    initial_price: float = None,
    volatility_factor: float = 1.5,
    crash_intensity: float = 0.8,
    recovery_strength: float = 0.6,
    seed: int = None
) -> pd.DataFrame:
    """
    Simulate market data for the COVID crash period.
    
    Parameters:
    - asset: Asset symbol (e.g., "SPX", "BTC", "XRP")
    - start_date: Start date for simulation (default: "2020-02-15")
    - end_date: End date for simulation (default: "2020-04-15")
    - initial_price: Initial price (default: asset-specific)
    - volatility_factor: Volatility multiplier (default: 1.5)
    - crash_intensity: How severe the crash is (0-1, default: 0.8)
    - recovery_strength: How strong the recovery is (0-1, default: 0.6)
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with simulated OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    if initial_price is None:
        if asset == "SPX":
            initial_price = 3380.0  # S&P 500 before crash
        elif asset == "BTC":
            initial_price = 10000.0  # Bitcoin before crash
        elif asset == "XRP":
            initial_price = 0.30  # XRP before crash
        else:
            initial_price = 100.0  # Default
    
    base_volatility = {
        "SPX": 0.015,
        "BTC": 0.035,
        "XRP": 0.045
    }.get(asset, 0.025)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    crash_start = start + timedelta(days=int(len(dates) * 0.15))
    crash_end = start + timedelta(days=int(len(dates) * 0.45))
    recovery_start = crash_end
    recovery_end = end
    
    prices = []
    current_price = initial_price
    
    for date in dates:
        if date < crash_start:
            daily_volatility = base_volatility * volatility_factor
            drift = 0.0005  # Slight upward bias
        elif date <= crash_end:
            progress = (date - crash_start).days / max(1, (crash_end - crash_start).days)
            daily_volatility = base_volatility * volatility_factor * (1 + progress)
            drift = -0.02 * crash_intensity * (1 - progress * 0.5)  # Strong downward bias
        else:
            progress = (date - recovery_start).days / max(1, (recovery_end - recovery_start).days)
            daily_volatility = base_volatility * volatility_factor * (1 + (1 - progress) * 0.5)
            drift = 0.01 * recovery_strength * progress  # Upward bias
        
        daily_return = np.random.normal(drift, daily_volatility)
        
        current_price *= (1 + daily_return)
        prices.append(current_price)
    
    data = []
    for i, date in enumerate(dates):
        close_price = prices[i]
        
        intraday_vol = base_volatility * volatility_factor * (0.5 + np.random.random())
        
        high_price = close_price * (1 + np.random.random() * intraday_vol)
        low_price = close_price * (1 - np.random.random() * intraday_vol)
        open_price = low_price + np.random.random() * (high_price - low_price)
        
        if date < crash_start:
            volume_factor = 1.0 + np.random.random() * 0.5
        elif date <= crash_end:
            progress = (date - crash_start).days / max(1, (crash_end - crash_start).days)
            volume_factor = 1.5 + progress * 3.0 * np.random.random()
        else:
            progress = (date - recovery_start).days / max(1, (recovery_end - recovery_start).days)
            volume_factor = 2.5 - progress * np.random.random()
        
        base_volume = {
            "SPX": 4e9,
            "BTC": 2e10,
            "XRP": 5e9
        }.get(asset, 1e9)
        
        volume = base_volume * volume_factor * (0.8 + 0.4 * np.random.random())
        
        data.append({
            "timestamp": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })
    
    return pd.DataFrame(data)

def generate_news_events(
    start_date: str = "2020-02-15",
    end_date: str = "2020-04-15",
    num_events: int = 20,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Generate simulated news events during the COVID crash period.
    
    Parameters:
    - start_date: Start date for simulation (default: "2020-02-15")
    - end_date: End date for simulation (default: "2020-04-15")
    - num_events: Number of news events to generate (default: 20)
    - seed: Random seed for reproducibility
    
    Returns:
    - List of news event dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    event_types = [
        "WHO Announcement",
        "Government Lockdown",
        "Stimulus Package",
        "Case Count Update",
        "Central Bank Action",
        "Travel Restriction",
        "Vaccine Development",
        "Corporate Earnings",
        "Unemployment Data",
        "Market Circuit Breaker"
    ]
    
    impact_levels = ["Low", "Medium", "High", "Critical"]
    
    events = []
    for _ in range(num_events):
        event_date = start + timedelta(days=np.random.randint(0, (end - start).days))
        event_type = np.random.choice(event_types)
        impact = np.random.choice(impact_levels, p=[0.2, 0.3, 0.3, 0.2])
        
        if event_type in ["Stimulus Package", "Central Bank Action", "Vaccine Development"]:
            sentiment = "Positive"
        elif event_type in ["Government Lockdown", "Case Count Update", "Travel Restriction", "Market Circuit Breaker"]:
            sentiment = "Negative"
        else:
            sentiment = np.random.choice(["Positive", "Negative", "Neutral"], p=[0.3, 0.5, 0.2])
        
        events.append({
            "date": event_date.strftime("%Y-%m-%d"),
            "type": event_type,
            "impact": impact,
            "sentiment": sentiment,
            "description": f"{event_type} - {impact} Impact - {sentiment} Sentiment"
        })
    
    events.sort(key=lambda x: x["date"])
    
    return events

def save_simulated_data(
    assets: List[str] = ["SPX", "BTC", "XRP"],
    start_date: str = "2020-02-15",
    end_date: str = "2020-04-15",
    output_dir: str = "covid_test/data"
) -> Dict[str, str]:
    """
    Generate and save simulated data for multiple assets.
    
    Parameters:
    - assets: List of asset symbols (default: ["SPX", "BTC", "XRP"])
    - start_date: Start date for simulation (default: "2020-02-15")
    - end_date: End date for simulation (default: "2020-04-15")
    - output_dir: Directory to save data (default: "covid_test/data")
    
    Returns:
    - Dictionary mapping assets to file paths
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/news_events", exist_ok=True)
    
    file_paths = {}
    for asset in assets:
        df = simulate_crash(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            seed=hash(asset) % 10000
        )
        
        file_path = f"{output_dir}/{asset.lower()}_covid_crash.csv"
        df.to_csv(file_path, index=False)
        file_paths[asset] = file_path
    
    news_events = generate_news_events(
        start_date=start_date,
        end_date=end_date,
        num_events=30,
        seed=42
    )
    
    import json
    with open(f"{output_dir}/news_events/covid_news_events.json", "w") as f:
        json.dump(news_events, f, indent=2)
    
    return file_paths
