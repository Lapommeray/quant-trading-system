"""
Void Trader Chart Overlays Module

This module implements advanced candlestick chart overlays for the Quantum Trading System,
providing visualization of quantum signals, void zones, and predictive patterns.

Dependencies:
- plotly
- numpy
- pandas
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import json
import base64
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('candle_overlays.log')
    ]
)

logger = logging.getLogger("CandleOverlays")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly loaded successfully")
except ImportError:
    logger.warning("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
    logger.info("Matplotlib loaded successfully")
except ImportError:
    logger.warning("Matplotlib not available. Some features will be limited.")
    MPL_AVAILABLE = False

class VoidTraderOverlays:
    """
    Void Trader Chart Overlays for the Quantum Trading System.
    Provides advanced visualization of quantum signals and predictive patterns.
    """
    
    def __init__(
        self,
        theme: str = "dark",
        show_ghost_candles: bool = True,
        show_quantum_zones: bool = True,
        show_void_levels: bool = True,
        show_fibonacci_grid: bool = True
    ):
        """
        Initialize the Void Trader Overlays.
        
        Parameters:
        - theme: Color theme ('dark' or 'light')
        - show_ghost_candles: Whether to show ghost (predicted) candles
        - show_quantum_zones: Whether to show quantum probability zones
        - show_void_levels: Whether to show void price levels
        - show_fibonacci_grid: Whether to show Fibonacci grid
        """
        self.theme = theme
        self.show_ghost_candles = show_ghost_candles
        self.show_quantum_zones = show_quantum_zones
        self.show_void_levels = show_void_levels
        self.show_fibonacci_grid = show_fibonacci_grid
        
        if theme == "dark":
            self.bg_color = "#0f1117"
            self.text_color = "#e0e0e0"
            self.grid_color = "#2a2e39"
            self.up_color = "#26a69a"
            self.down_color = "#ef5350"
            self.ghost_up_color = "rgba(38, 166, 154, 0.5)"
            self.ghost_down_color = "rgba(239, 83, 80, 0.5)"
            self.quantum_zone_color = "rgba(103, 58, 183, 0.3)"
            self.void_level_color = "rgba(255, 152, 0, 0.8)"
            self.fibonacci_colors = [
                "rgba(255, 215, 0, 0.3)",
                "rgba(255, 165, 0, 0.3)",
                "rgba(255, 69, 0, 0.3)",
                "rgba(138, 43, 226, 0.3)",
                "rgba(0, 191, 255, 0.3)"
            ]
        else:
            self.bg_color = "#ffffff"
            self.text_color = "#333333"
            self.grid_color = "#e0e0e0"
            self.up_color = "#4caf50"
            self.down_color = "#f44336"
            self.ghost_up_color = "rgba(76, 175, 80, 0.5)"
            self.ghost_down_color = "rgba(244, 67, 54, 0.5)"
            self.quantum_zone_color = "rgba(103, 58, 183, 0.2)"
            self.void_level_color = "rgba(255, 152, 0, 0.7)"
            self.fibonacci_colors = [
                "rgba(255, 215, 0, 0.2)",
                "rgba(255, 165, 0, 0.2)",
                "rgba(255, 69, 0, 0.2)",
                "rgba(138, 43, 226, 0.2)",
                "rgba(0, 191, 255, 0.2)"
            ]
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly is required for this module")
            
        logger.info("VoidTraderOverlays initialized")
        
    def create_candlestick_chart(
        self,
        ohlc_data: pd.DataFrame,
        ghost_candles: Optional[pd.DataFrame] = None,
        quantum_zones: Optional[List[Dict[str, Any]]] = None,
        void_levels: Optional[List[Dict[str, Any]]] = None,
        signals: Optional[List[Dict[str, Any]]] = None,
        title: str = "Void Trader Chart",
        height: int = 800,
        width: int = 1200
    ) -> Optional[go.Figure]:
        """
        Create a candlestick chart with Void Trader overlays.
        
        Parameters:
        - ohlc_data: DataFrame with OHLC data (must have 'timestamp', 'open', 'high', 'low', 'close', 'volume' columns)
        - ghost_candles: DataFrame with predicted candles (same format as ohlc_data)
        - quantum_zones: List of quantum probability zones
        - void_levels: List of void price levels
        - signals: List of trading signals
        - title: Chart title
        - height: Chart height
        - width: Chart width
        
        Returns:
        - Plotly figure object or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None
            
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in ohlc_data.columns:
                    logger.error(f"Missing required column: {col}")
                    return None
                    
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.8, 0.2],
                subplot_titles=(title, "Volume")
            )
            
            fig.add_trace(
                go.Candlestick(
                    x=ohlc_data['timestamp'],
                    open=ohlc_data['open'],
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close'],
                    name="Price",
                    increasing_line=dict(color=self.up_color),
                    decreasing_line=dict(color=self.down_color)
                ),
                row=1, col=1
            )
            
            colors = [self.up_color if ohlc_data['close'][i] >= ohlc_data['open'][i] else self.down_color for i in range(len(ohlc_data))]
            
            fig.add_trace(
                go.Bar(
                    x=ohlc_data['timestamp'],
                    y=ohlc_data['volume'],
                    name="Volume",
                    marker=dict(color=colors, opacity=0.8)
                ),
                row=2, col=1
            )
            
            if self.show_ghost_candles and ghost_candles is not None:
                for col in required_columns:
                    if col not in ghost_candles.columns:
                        logger.warning(f"Missing column in ghost_candles: {col}")
                        break
                else:
                    fig.add_trace(
                        go.Candlestick(
                            x=ghost_candles['timestamp'],
                            open=ghost_candles['open'],
                            high=ghost_candles['high'],
                            low=ghost_candles['low'],
                            close=ghost_candles['close'],
                            name="Ghost Candles",
                            opacity=0.5,
                            increasing_line=dict(color=self.ghost_up_color),
                            decreasing_line=dict(color=self.ghost_down_color)
                        ),
                        row=1, col=1
                    )
                    
            if self.show_quantum_zones and quantum_zones is not None:
                for i, zone in enumerate(quantum_zones):
                    if 'start_time' in zone and 'end_time' in zone and 'lower' in zone and 'upper' in zone:
                        fig.add_trace(
                            go.Scatter(
                                x=[zone['start_time'], zone['start_time'], zone['end_time'], zone['end_time']],
                                y=[zone['lower'], zone['upper'], zone['upper'], zone['lower']],
                                fill="toself",
                                fillcolor=self.quantum_zone_color,
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f"Quantum Zone {i+1}",
                                showlegend=i==0  # Show only one in legend
                            ),
                            row=1, col=1
                        )
                        
            if self.show_void_levels and void_levels is not None:
                for i, level in enumerate(void_levels):
                    if 'price' in level and 'start_time' in level and 'end_time' in level:
                        fig.add_trace(
                            go.Scatter(
                                x=[level['start_time'], level['end_time']],
                                y=[level['price'], level['price']],
                                mode='lines',
                                line=dict(
                                    color=self.void_level_color,
                                    width=2,
                                    dash='dash'
                                ),
                                name=f"Void Level {i+1}",
                                showlegend=i==0  # Show only one in legend
                            ),
                            row=1, col=1
                        )
                        
            if self.show_fibonacci_grid:
                price_min = ohlc_data['low'].min()
                price_max = ohlc_data['high'].max()
                price_range = price_max - price_min
                
                fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
                fib_prices = [price_max - level * price_range for level in fib_levels]
                
                for i, (level, price) in enumerate(zip(fib_levels, fib_prices)):
                    if i < len(self.fibonacci_colors):
                        color = self.fibonacci_colors[i % len(self.fibonacci_colors)]
                    else:
                        color = self.fibonacci_colors[0]
                        
                    fig.add_trace(
                        go.Scatter(
                            x=[ohlc_data['timestamp'].iloc[0], ohlc_data['timestamp'].iloc[-1]],
                            y=[price, price],
                            mode='lines',
                            line=dict(
                                color=color.replace('0.3', '0.8'),  # More visible line
                                width=1
                            ),
                            name=f"Fib {level}",
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                    
            if signals is not None:
                buy_times = []
                buy_prices = []
                sell_times = []
                sell_prices = []
                
                for signal in signals:
                    if 'timestamp' in signal and 'price' in signal and 'type' in signal:
                        if signal['type'].lower() == 'buy':
                            buy_times.append(signal['timestamp'])
                            buy_prices.append(signal['price'])
                        elif signal['type'].lower() == 'sell':
                            sell_times.append(signal['timestamp'])
                            sell_prices.append(signal['price'])
                            
                if buy_times:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_times,
                            y=buy_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=self.up_color,
                                line=dict(width=1, color=self.bg_color)
                            ),
                            name="Buy Signals"
                        ),
                        row=1, col=1
                    )
                    
                if sell_times:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_times,
                            y=sell_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=self.down_color,
                                line=dict(width=1, color=self.bg_color)
                            ),
                            name="Sell Signals"
                        ),
                        row=1, col=1
                    )
                    
            fig.update_layout(
                height=height,
                width=width,
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                font=dict(color=self.text_color),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True,
                    rangeslider=dict(visible=False)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                ),
                xaxis2=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                ),
                yaxis2=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            return fig
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return None
            
    def create_quantum_heatmap(
        self,
        price_levels: np.ndarray,
        time_points: List[datetime],
        probability_matrix: np.ndarray,
        title: str = "Quantum Probability Heatmap",
        height: int = 600,
        width: int = 1000
    ) -> Optional[go.Figure]:
        """
        Create a quantum probability heatmap.
        
        Parameters:
        - price_levels: Array of price levels
        - time_points: List of time points
        - probability_matrix: 2D array of probabilities (price_levels x time_points)
        - title: Chart title
        - height: Chart height
        - width: Chart width
        
        Returns:
        - Plotly figure object or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None
            
        try:
            if probability_matrix.shape != (len(price_levels), len(time_points)):
                logger.error(f"Probability matrix shape {probability_matrix.shape} does not match price_levels {len(price_levels)} and time_points {len(time_points)}")
                return None
                
            fig = go.Figure(data=go.Heatmap(
                z=probability_matrix.T,
                x=price_levels,
                y=[t.strftime('%Y-%m-%d %H:%M') for t in time_points],
                colorscale='Viridis',
                colorbar=dict(title='Probability'),
                hovertemplate='Price: %{x}<br>Time: %{y}<br>Probability: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                height=height,
                width=width,
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                font=dict(color=self.text_color),
                xaxis=dict(
                    title='Price',
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                ),
                yaxis=dict(
                    title='Time',
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                )
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating quantum heatmap: {str(e)}")
            return None
            
    def create_void_trader_dashboard(
        self,
        ohlc_data: pd.DataFrame,
        ghost_candles: Optional[pd.DataFrame] = None,
        quantum_zones: Optional[List[Dict[str, Any]]] = None,
        void_levels: Optional[List[Dict[str, Any]]] = None,
        signals: Optional[List[Dict[str, Any]]] = None,
        probability_matrix: Optional[Tuple[np.ndarray, List[datetime], np.ndarray]] = None,
        title: str = "Void Trader Dashboard",
        height: int = 1000,
        width: int = 1200
    ) -> Optional[go.Figure]:
        """
        Create a complete Void Trader dashboard with multiple visualizations.
        
        Parameters:
        - ohlc_data: DataFrame with OHLC data
        - ghost_candles: DataFrame with predicted candles
        - quantum_zones: List of quantum probability zones
        - void_levels: List of void price levels
        - signals: List of trading signals
        - probability_matrix: Tuple of (price_levels, time_points, probability_matrix) for heatmap
        - title: Dashboard title
        - height: Dashboard height
        - width: Dashboard width
        
        Returns:
        - Plotly figure object or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None
            
        try:
            num_rows = 2  # Minimum: price and volume
            
            if probability_matrix is not None:
                num_rows += 1
                
            fig = make_subplots(
                rows=num_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6] + [0.4 / (num_rows - 1)] * (num_rows - 1),
                subplot_titles=[title, "Volume"] + (["Quantum Probability"] if probability_matrix is not None else [])
            )
            
            fig.add_trace(
                go.Candlestick(
                    x=ohlc_data['timestamp'],
                    open=ohlc_data['open'],
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close'],
                    name="Price",
                    increasing_line=dict(color=self.up_color),
                    decreasing_line=dict(color=self.down_color)
                ),
                row=1, col=1
            )
            
            if self.show_ghost_candles and ghost_candles is not None:
                required_columns = ['timestamp', 'open', 'high', 'low', 'close']
                for col in required_columns:
                    if col not in ghost_candles.columns:
                        logger.warning(f"Missing column in ghost_candles: {col}")
                        break
                else:
                    fig.add_trace(
                        go.Candlestick(
                            x=ghost_candles['timestamp'],
                            open=ghost_candles['open'],
                            high=ghost_candles['high'],
                            low=ghost_candles['low'],
                            close=ghost_candles['close'],
                            name="Ghost Candles",
                            opacity=0.5,
                            increasing_line=dict(color=self.ghost_up_color),
                            decreasing_line=dict(color=self.ghost_down_color)
                        ),
                        row=1, col=1
                    )
                    
            if self.show_quantum_zones and quantum_zones is not None:
                for i, zone in enumerate(quantum_zones):
                    if 'start_time' in zone and 'end_time' in zone and 'lower' in zone and 'upper' in zone:
                        fig.add_trace(
                            go.Scatter(
                                x=[zone['start_time'], zone['start_time'], zone['end_time'], zone['end_time']],
                                y=[zone['lower'], zone['upper'], zone['upper'], zone['lower']],
                                fill="toself",
                                fillcolor=self.quantum_zone_color,
                                line=dict(color='rgba(0,0,0,0)'),
                                name=f"Quantum Zone {i+1}",
                                showlegend=i==0  # Show only one in legend
                            ),
                            row=1, col=1
                        )
                        
            if self.show_void_levels and void_levels is not None:
                for i, level in enumerate(void_levels):
                    if 'price' in level and 'start_time' in level and 'end_time' in level:
                        fig.add_trace(
                            go.Scatter(
                                x=[level['start_time'], level['end_time']],
                                y=[level['price'], level['price']],
                                mode='lines',
                                line=dict(
                                    color=self.void_level_color,
                                    width=2,
                                    dash='dash'
                                ),
                                name=f"Void Level {i+1}",
                                showlegend=i==0  # Show only one in legend
                            ),
                            row=1, col=1
                        )
                        
            if signals is not None:
                buy_times = []
                buy_prices = []
                sell_times = []
                sell_prices = []
                
                for signal in signals:
                    if 'timestamp' in signal and 'price' in signal and 'type' in signal:
                        if signal['type'].lower() == 'buy':
                            buy_times.append(signal['timestamp'])
                            buy_prices.append(signal['price'])
                        elif signal['type'].lower() == 'sell':
                            sell_times.append(signal['timestamp'])
                            sell_prices.append(signal['price'])
                            
                if buy_times:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_times,
                            y=buy_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=self.up_color,
                                line=dict(width=1, color=self.bg_color)
                            ),
                            name="Buy Signals"
                        ),
                        row=1, col=1
                    )
                    
                if sell_times:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_times,
                            y=sell_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=self.down_color,
                                line=dict(width=1, color=self.bg_color)
                            ),
                            name="Sell Signals"
                        ),
                        row=1, col=1
                    )
                    
            colors = [self.up_color if ohlc_data['close'][i] >= ohlc_data['open'][i] else self.down_color for i in range(len(ohlc_data))]
            
            fig.add_trace(
                go.Bar(
                    x=ohlc_data['timestamp'],
                    y=ohlc_data['volume'],
                    name="Volume",
                    marker=dict(color=colors, opacity=0.8)
                ),
                row=2, col=1
            )
            
            if probability_matrix is not None:
                price_levels, time_points, prob_matrix = probability_matrix
                
                fig.add_trace(
                    go.Heatmap(
                        z=prob_matrix.T,
                        x=price_levels,
                        y=[t.strftime('%Y-%m-%d %H:%M') for t in time_points],
                        colorscale='Viridis',
                        colorbar=dict(title='Probability'),
                        hovertemplate='Price: %{x}<br>Time: %{y}<br>Probability: %{z}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
            fig.update_layout(
                height=height,
                width=width,
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                font=dict(color=self.text_color),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True,
                    rangeslider=dict(visible=False)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=self.grid_color,
                    showticklabels=True
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            return fig
        except Exception as e:
            logger.error(f"Error creating Void Trader dashboard: {str(e)}")
            return None
            
    def generate_ghost_candles(
        self,
        ohlc_data: pd.DataFrame,
        num_candles: int = 5,
        volatility_factor: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate ghost (predicted) candles based on historical data.
        This is a simple example - in a real system, this would use the Quantum LSTM or other predictive models.
        
        Parameters:
        - ohlc_data: DataFrame with OHLC data
        - num_candles: Number of ghost candles to generate
        - volatility_factor: Factor to adjust volatility of predictions
        
        Returns:
        - DataFrame with ghost candles
        """
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in ohlc_data.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
                    
            last_timestamp = ohlc_data['timestamp'].iloc[-1]
            
            if len(ohlc_data) < 2:
                logger.error("Not enough data to calculate time delta")
                return pd.DataFrame()
                
            time_delta = ohlc_data['timestamp'].iloc[-1] - ohlc_data['timestamp'].iloc[-2]
            
            recent_data = ohlc_data.tail(20)
            price_changes = (recent_data['close'] - recent_data['open']).abs()
            avg_change = price_changes.mean()
            std_change = price_changes.std()
            
            high_low_ranges = recent_data['high'] - recent_data['low']
            avg_range = high_low_ranges.mean()
            
            ghost_data = []
            
            last_close = ohlc_data['close'].iloc[-1]
            last_volume = ohlc_data['volume'].iloc[-1]
            
            for i in range(num_candles):
                next_timestamp = last_timestamp + time_delta * (i + 1)
                
                price_change = np.random.normal(0, std_change * volatility_factor)
                
                if price_change >= 0:
                    ghost_open = last_close
                    ghost_close = ghost_open + price_change
                    ghost_high = ghost_close + np.random.uniform(0, avg_range * 0.5 * volatility_factor)
                    ghost_low = ghost_open - np.random.uniform(0, avg_range * 0.3 * volatility_factor)
                else:
                    ghost_open = last_close
                    ghost_close = ghost_open + price_change
                    ghost_high = ghost_open + np.random.uniform(0, avg_range * 0.3 * volatility_factor)
                    ghost_low = ghost_close - np.random.uniform(0, avg_range * 0.5 * volatility_factor)
                    
                ghost_volume = last_volume * np.random.uniform(0.7, 1.3)
                
                ghost_data.append({
                    'timestamp': next_timestamp,
                    'open': ghost_open,
                    'high': ghost_high,
                    'low': ghost_low,
                    'close': ghost_close,
                    'volume': ghost_volume
                })
                
                last_close = ghost_close
                last_volume = ghost_volume
                
            ghost_df = pd.DataFrame(ghost_data)
            
            return ghost_df
        except Exception as e:
            logger.error(f"Error generating ghost candles: {str(e)}")
            return pd.DataFrame()
            
    def generate_quantum_zones(
        self,
        ohlc_data: pd.DataFrame,
        ghost_candles: pd.DataFrame,
        num_zones: int = 3,
        confidence_levels: List[float] = [0.7, 0.5, 0.3]
    ) -> List[Dict[str, Any]]:
        """
        Generate quantum probability zones based on historical data and ghost candles.
        
        Parameters:
        - ohlc_data: DataFrame with OHLC data
        - ghost_candles: DataFrame with ghost candles
        - num_zones: Number of zones to generate
        - confidence_levels: Confidence levels for each zone
        
        Returns:
        - List of quantum zones
        """
        try:
            if len(ghost_candles) == 0:
                logger.error("No ghost candles provided")
                return []
                
            last_real = ohlc_data.iloc[-1]
            first_ghost = ghost_candles.iloc[0]
            
            price_range = ohlc_data['high'].max() - ohlc_data['low'].min()
            
            zones = []
            
            for i in range(min(num_zones, len(confidence_levels))):
                confidence = confidence_levels[i]
                
                center = (first_ghost['high'] + first_ghost['low']) / 2
                
                width = price_range * (1 - confidence) * 0.5
                
                zone = {
                    'start_time': last_real['timestamp'],
                    'end_time': ghost_candles['timestamp'].iloc[-1],
                    'lower': center - width,
                    'upper': center + width,
                    'confidence': confidence
                }
                
                zones.append(zone)
                
            return zones
        except Exception as e:
            logger.error(f"Error generating quantum zones: {str(e)}")
            return []
            
    def generate_void_levels(
        self,
        ohlc_data: pd.DataFrame,
        num_levels: int = 3,
        lookback_periods: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate void price levels based on historical data.
        
        Parameters:
        - ohlc_data: DataFrame with OHLC data
        - num_levels: Number of void levels to generate
        - lookback_periods: Number of periods to look back
        
        Returns:
        - List of void levels
        """
        try:
            if len(ohlc_data) < lookback_periods:
                logger.error(f"Not enough data. Need at least {lookback_periods} periods.")
                return []
                
            recent_data = ohlc_data.tail(lookback_periods)
            
            prices = np.concatenate([
                recent_data['open'].values,
                recent_data['high'].values,
                recent_data['low'].values,
                recent_data['close'].values
            ])
            
            hist, bin_edges = np.histogram(prices, bins=50)
            
            void_indices = np.argsort(hist)[:num_levels]
            void_prices = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in void_indices]
            
            levels = []
            
            for price in void_prices:
                level = {
                    'price': price,
                    'start_time': recent_data['timestamp'].iloc[0],
                    'end_time': recent_data['timestamp'].iloc[-1] + (recent_data['timestamp'].iloc[-1] - recent_data['timestamp'].iloc[-2]) * 5,
                    'strength': 1.0 - (hist[np.digitize(price, bin_edges) - 1] / hist.max())
                }
                
                levels.append(level)
                
            return levels
        except Exception as e:
            logger.error(f"Error generating void levels: {str(e)}")
            return []
            
    def save_chart_to_image(self, fig: go.Figure, filepath: str, format: str = 'png') -> bool:
        """
        Save a Plotly figure to an image file.
        
        Parameters:
        - fig: Plotly figure object
        - filepath: Path to save the image
        - format: Image format ('png', 'jpg', 'svg', 'pdf')
        
        Returns:
        - Success status
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return False
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            fig.write_image(filepath, format=format)
            
            logger.info(f"Chart saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving chart to image: {str(e)}")
            return False
            
    def get_chart_as_base64(self, fig: go.Figure, format: str = 'png') -> Optional[str]:
        """
        Get a Plotly figure as a base64-encoded string.
        
        Parameters:
        - fig: Plotly figure object
        - format: Image format ('png', 'jpg', 'svg')
        
        Returns:
        - Base64-encoded image string or None if error
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None
            
        try:
            img_bytes = fig.to_image(format=format)
            
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/{format};base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error converting chart to base64: {str(e)}")
            return None

if __name__ == "__main__":
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
    opens = closes - np.random.normal(0, 0.5, 100)
    highs = np.maximum(closes, opens) + np.random.uniform(0, 1, 100)
    lows = np.minimum(closes, opens) - np.random.uniform(0, 1, 100)
    volumes = np.random.uniform(100, 1000, 100)
    
    ohlc_df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    overlays = VoidTraderOverlays(theme="dark")
    
    ghost_df = overlays.generate_ghost_candles(ohlc_df, num_candles=5)
    
    quantum_zones = overlays.generate_quantum_zones(ohlc_df, ghost_df)
    
    void_levels = overlays.generate_void_levels(ohlc_df)
    
    fig = overlays.create_candlestick_chart(
        ohlc_df,
        ghost_candles=ghost_df,
        quantum_zones=quantum_zones,
        void_levels=void_levels
    )
    
    if fig:
        overlays.save_chart_to_image(fig, "void_trader_chart.png")
        
        base64_str = overlays.get_chart_as_base64(fig)
        if base64_str:
            print(f"Base64 string length: {len(base64_str)}")
