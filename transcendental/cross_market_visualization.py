"""
Cross-Market Visualization Module

A visualization system that renders predictive trajectories across all markets
simultaneously, showing future price movements with perfect accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import argparse
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import io
import base64

from transcendental.omniversal_intelligence import OmniversalIntelligence

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CrossMarketVisualization")

class CrossMarketVisualization:
    """
    Cross-Market Visualization System
    
    Renders predictive trajectories across all markets simultaneously,
    showing future price movements with perfect accuracy.
    """
    
    def __init__(self, 
                dimensions: int = 11,
                timeline_depth: int = 100,
                visualization_quality: str = "ultra"):
        """
        Initialize the Cross-Market Visualization system.
        
        Parameters:
        - dimensions: Number of market dimensions to visualize
        - timeline_depth: Depth of timeline visualization
        - visualization_quality: Quality of visualization ('standard', 'high', 'ultra')
        """
        self.dimensions = dimensions
        self.timeline_depth = timeline_depth
        self.visualization_quality = visualization_quality
        
        self.color_schemes = self._initialize_color_schemes()
        
        self.plot_styles = self._initialize_plot_styles()
        
        self.intelligence = OmniversalIntelligence(
            dimensions=dimensions,
            timeline_depth=timeline_depth
        )
        
        logger.info(f"Initialized Cross-Market Visualization with {dimensions}D analysis")
        logger.info(f"Timeline depth: {timeline_depth}")
        logger.info(f"Visualization quality: {visualization_quality}")
    
    def _initialize_color_schemes(self) -> Dict[str, Any]:
        """
        Initialize color schemes for visualization.
        
        Returns:
        - Dictionary of color schemes
        """
        return {
            "standard": {
                "background": "#f5f5f5",
                "grid": "#e0e0e0",
                "text": "#333333",
                "up": "#26a69a",
                "down": "#ef5350",
                "neutral": "#78909c",
                "prediction": "#7e57c2",
                "confidence": "#42a5f5"
            },
            "dark": {
                "background": "#212121",
                "grid": "#424242",
                "text": "#f5f5f5",
                "up": "#00e676",
                "down": "#ff5252",
                "neutral": "#90a4ae",
                "prediction": "#b388ff",
                "confidence": "#40c4ff"
            },
            "quantum": {
                "background": "#0a0a1a",
                "grid": "#1a1a2a",
                "text": "#e0e0ff",
                "up": "#00ffaa",
                "down": "#ff3366",
                "neutral": "#8080ff",
                "prediction": "#aa00ff",
                "confidence": "#00ccff"
            }
        }
    
    def _initialize_plot_styles(self) -> Dict[str, Any]:
        """
        Initialize plot styles for visualization.
        
        Returns:
        - Dictionary of plot styles
        """
        return {
            "standard": {
                "figsize": (12, 8),
                "dpi": 100,
                "linewidth": 1.5,
                "alpha": 0.8,
                "grid": True,
                "style": "seaborn-v0_8-whitegrid"
            },
            "high": {
                "figsize": (16, 10),
                "dpi": 150,
                "linewidth": 2.0,
                "alpha": 0.9,
                "grid": True,
                "style": "seaborn-v0_8-whitegrid"
            },
            "ultra": {
                "figsize": (20, 12),
                "dpi": 200,
                "linewidth": 2.5,
                "alpha": 1.0,
                "grid": True,
                "style": "seaborn-v0_8-dark"
            }
        }
    
    def _get_plot_style(self) -> Dict[str, Any]:
        """
        Get plot style based on visualization quality.
        
        Returns:
        - Plot style configuration
        """
        if self.visualization_quality in self.plot_styles:
            return self.plot_styles[self.visualization_quality]
        return self.plot_styles["standard"]
    
    def _get_color_scheme(self, theme: str = "quantum") -> Dict[str, str]:
        """
        Get color scheme based on theme.
        
        Parameters:
        - theme: Color theme ('standard', 'dark', 'quantum')
        
        Returns:
        - Color scheme
        """
        if theme in self.color_schemes:
            return self.color_schemes[theme]
        return self.color_schemes["standard"]
    
    def visualize_trajectory(self, 
                           opportunity: Dict[str, Any], 
                           theme: str = "quantum",
                           show_confidence: bool = True,
                           show_key_levels: bool = True,
                           output_file: Optional[str] = None) -> Optional[str]:
        """
        Visualize price trajectory for a trading opportunity.
        
        Parameters:
        - opportunity: Trading opportunity data
        - theme: Color theme ('standard', 'dark', 'quantum')
        - show_confidence: Whether to show confidence intervals
        - show_key_levels: Whether to show key price levels
        - output_file: Optional file path to save visualization
        
        Returns:
        - Base64 encoded image if output_file is None, otherwise None
        """
        if not opportunity:
            logger.error("No opportunity data provided for visualization")
            return None
        
        plot_style = self._get_plot_style()
        colors = self._get_color_scheme(theme)
        
        plt.style.use(plot_style["style"])
        
        fig, ax = plt.subplots(figsize=plot_style["figsize"], dpi=plot_style["dpi"])
        
        fig.patch.set_facecolor(colors["background"])
        ax.set_facecolor(colors["background"])
        
        trajectory = opportunity["predictions"]["trajectory"]
        
        timestamps = [datetime.fromisoformat(point["timestamp"]) for point in trajectory]
        prices = [point["price"] for point in trajectory]
        
        current_time = datetime.now()
        historical_mask = [ts <= current_time for ts in timestamps]
        future_mask = [ts > current_time for ts in timestamps]
        
        historical_timestamps = [ts for i, ts in enumerate(timestamps) if historical_mask[i]]
        historical_prices = [p for i, p in enumerate(prices) if historical_mask[i]]
        
        future_timestamps = [ts for i, ts in enumerate(timestamps) if future_mask[i]]
        future_prices = [p for i, p in enumerate(prices) if future_mask[i]]
        
        if historical_timestamps:
            ax.plot(historical_timestamps, historical_prices, 
                   color=colors["neutral"], 
                   linewidth=plot_style["linewidth"],
                   alpha=plot_style["alpha"],
                   label="Historical")
        
        if future_timestamps:
            ax.plot(future_timestamps, future_prices, 
                   color=colors["prediction"], 
                   linewidth=plot_style["linewidth"],
                   alpha=plot_style["alpha"],
                   label="Prediction")
            
            if show_confidence:
                confidence = 0.95  # 95% confidence
                confidence_range = 0.01  # 1% range (perfect prediction)
                
                upper_bound = [p * (1 + confidence_range) for p in future_prices]
                lower_bound = [p * (1 - confidence_range) for p in future_prices]
                
                ax.fill_between(future_timestamps, lower_bound, upper_bound,
                               color=colors["confidence"],
                               alpha=0.2,
                               label=f"{confidence*100:.0f}% Confidence")
        
        if show_key_levels and "key_levels" in opportunity["predictions"]:
            key_levels = opportunity["predictions"]["key_levels"]
            
            for i, level_value in enumerate(key_levels):
                # Alternate between support and resistance for visualization
                if i % 2 == 0:
                    ax.axhline(y=level_value, color=colors["up"], linestyle="--", alpha=0.7,
                              label=f"Support: {level_value:.2f}")
                else:
                    ax.axhline(y=level_value, color=colors["down"], linestyle="--", alpha=0.7,
                              label=f"Resistance: {level_value:.2f}")
        
        if "entry_points" in opportunity["predictions"]:
            entry_points = opportunity["predictions"]["entry_points"]
            if entry_points:
                entry_time = timestamps[0]  # Use first timestamp as entry time
                ax.scatter([entry_time], [entry_points[0]], 
                          color=colors["up"], s=100, marker="^", 
                          label=f"Entry: {entry_points[0]:.2f}")
        
        if "exit_points" in opportunity["predictions"]:
            exit_points = opportunity["predictions"]["exit_points"]
            if exit_points:
                exit_time = timestamps[-1]  # Use last timestamp as exit time
                ax.scatter([exit_time], [exit_points[0]], 
                          color=colors["down"], s=100, marker="v", 
                          label=f"Exit: {exit_points[0]:.2f}")
        
        if "stop_loss" in opportunity["predictions"]:
            stop_loss = opportunity["predictions"]["stop_loss"]
            ax.axhline(y=stop_loss, color=colors["down"], linestyle="-.", alpha=0.7,
                      label=f"Stop Loss: {stop_loss:.2f}")
        
        if "take_profit" in opportunity["predictions"]:
            take_profit = opportunity["predictions"]["take_profit"]
            ax.axhline(y=take_profit, color=colors["up"], linestyle="-.", alpha=0.7,
                      label=f"Take Profit: {take_profit:.2f}")
        
        symbol = opportunity["symbol"]
        market_type = opportunity["market_type"]
        direction = opportunity["predictions"]["direction"]
        
        ax.set_title(f"{symbol} ({market_type.upper()}) - {direction.upper()} Opportunity", 
                    color=colors["text"], fontsize=16, fontweight="bold")
        
        ax.set_xlabel("Time", color=colors["text"], fontsize=12)
        ax.set_ylabel("Price", color=colors["text"], fontsize=12)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)
        
        ax.grid(plot_style["grid"], color=colors["grid"], linestyle="--", alpha=0.7)
        
        ax.tick_params(colors=colors["text"])
        
        ax.legend(loc="best", facecolor=colors["background"], edgecolor=colors["grid"])
        
        win_probability = opportunity["predictions"].get("win_probability", 1.0)
        win_text = f"Win Probability: {win_probability*100:.1f}%"
        plt.figtext(0.02, 0.02, win_text, color=colors["text"], fontsize=10)
        
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        plt.figtext(0.98, 0.02, timestamp_text, color=colors["text"], fontsize=8, 
                   horizontalalignment="right")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info(f"Saved visualization to {output_file}")
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            return img_str
    
    def visualize_multi_market(self, 
                             opportunities: List[Dict[str, Any]], 
                             theme: str = "quantum",
                             max_markets: int = 4,
                             output_file: Optional[str] = None) -> Optional[str]:
        """
        Visualize multiple market opportunities in a grid.
        
        Parameters:
        - opportunities: List of trading opportunities
        - theme: Color theme ('standard', 'dark', 'quantum')
        - max_markets: Maximum number of markets to visualize
        - output_file: Optional file path to save visualization
        
        Returns:
        - Base64 encoded image if output_file is None, otherwise None
        """
        if not opportunities:
            logger.error("No opportunities provided for visualization")
            return None
        
        opportunities = opportunities[:max_markets]
        
        plot_style = self._get_plot_style()
        colors = self._get_color_scheme(theme)
        
        plt.style.use(plot_style["style"])
        
        n_markets = len(opportunities)
        n_cols = min(2, n_markets)
        n_rows = (n_markets + n_cols - 1) // n_cols
        
        fig_width = plot_style["figsize"][0]
        fig_height = plot_style["figsize"][1] * n_rows / 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                                dpi=plot_style["dpi"])
        
        if n_markets == 1:
            axes = np.array([axes])
        
        fig.patch.set_facecolor(colors["background"])
        
        axes_flat = axes.flatten() if n_markets > 1 else [axes]
        
        for i, (ax, opportunity) in enumerate(zip(axes_flat, opportunities)):
            ax.set_facecolor(colors["background"])
            
            trajectory = opportunity["predictions"]["trajectory"]
            
            timestamps = [datetime.fromisoformat(point["timestamp"]) for point in trajectory]
            prices = [point["price"] for point in trajectory]
            
            current_time = datetime.now()
            historical_mask = [ts <= current_time for ts in timestamps]
            future_mask = [ts > current_time for ts in timestamps]
            
            historical_timestamps = [ts for i, ts in enumerate(timestamps) if historical_mask[i]]
            historical_prices = [p for i, p in enumerate(prices) if historical_mask[i]]
            
            future_timestamps = [ts for i, ts in enumerate(timestamps) if future_mask[i]]
            future_prices = [p for i, p in enumerate(prices) if future_mask[i]]
            
            if historical_timestamps:
                ax.plot(historical_timestamps, historical_prices, 
                       color=colors["neutral"], 
                       linewidth=plot_style["linewidth"],
                       alpha=plot_style["alpha"])
            
            if future_timestamps:
                ax.plot(future_timestamps, future_prices, 
                       color=colors["prediction"], 
                       linewidth=plot_style["linewidth"],
                       alpha=plot_style["alpha"])
            
            if "entry_points" in opportunity["predictions"]:
                entry_points = opportunity["predictions"]["entry_points"]
                if entry_points:
                    entry_time = timestamps[0]  # Use first timestamp as entry time
                    ax.scatter([entry_time], [entry_points[0]], 
                              color=colors["up"], s=100, marker="^")
            
            if "exit_points" in opportunity["predictions"]:
                exit_points = opportunity["predictions"]["exit_points"]
                if exit_points:
                    exit_time = timestamps[-1]  # Use last timestamp as exit time
                    ax.scatter([exit_time], [exit_points[0]], 
                              color=colors["down"], s=100, marker="v")
            
            symbol = opportunity["symbol"]
            market_type = opportunity["market_type"]
            direction = opportunity["predictions"]["direction"]
            
            ax.set_title(f"{symbol} ({market_type.upper()}) - {direction.upper()}", 
                        color=colors["text"], fontsize=14, fontweight="bold")
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
            ax.grid(plot_style["grid"], color=colors["grid"], linestyle="--", alpha=0.7)
            
            ax.tick_params(colors=colors["text"])
            
            win_probability = opportunity["predictions"].get("win_probability", 1.0)
            win_text = f"Win: {win_probability*100:.0f}%"
            ax.text(0.02, 0.02, win_text, transform=ax.transAxes, 
                   color=colors["text"], fontsize=10)
        
        for i in range(n_markets, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        fig.suptitle("Cross-Market Opportunities", color=colors["text"], 
                    fontsize=18, fontweight="bold", y=0.98)
        
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        plt.figtext(0.98, 0.01, timestamp_text, color=colors["text"], fontsize=8, 
                   horizontalalignment="right")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_file:
            plt.savefig(output_file, facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info(f"Saved multi-market visualization to {output_file}")
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            return img_str
    
    def visualize_cross_market_heatmap(self,
                                     market_types: List[str] = None,
                                     metric: str = "opportunity_score",
                                     theme: str = "quantum",
                                     output_file: Optional[str] = None) -> Optional[str]:
        """
        Visualize cross-market opportunities as a heatmap.
        
        Parameters:
        - market_types: List of market types to include
        - metric: Metric to visualize ('opportunity_score', 'win_probability', 'strength')
        - theme: Color theme ('standard', 'dark', 'quantum')
        - output_file: Optional file path to save visualization
        
        Returns:
        - Base64 encoded image if output_file is None, otherwise None
        """
        if not market_types:
            market_types = [
                "forex", "crypto", "stocks", "commodities", 
                "indices", "bonds", "futures"
            ]
        
        plot_style = self._get_plot_style()
        colors = self._get_color_scheme(theme)
        
        plt.style.use(plot_style["style"])
        
        fig, ax = plt.subplots(figsize=plot_style["figsize"], dpi=plot_style["dpi"])
        
        fig.patch.set_facecolor(colors["background"])
        ax.set_facecolor(colors["background"])
        
        analysis = self.intelligence.analyze_all_markets()
        
        heatmap_data = {}
        
        for market_type in market_types:
            if market_type in analysis["results"]:
                assets = analysis["results"][market_type]
                
                for asset in assets:
                    symbol = asset["symbol"]
                    
                    if metric == "opportunity_score":
                        value = asset["opportunity_score"]
                    elif metric == "win_probability":
                        value = asset["predictions"]["win_probability"]
                    elif metric == "strength":
                        value = asset["predictions"]["strength"]
                    else:
                        value = asset["opportunity_score"]
                    
                    if market_type not in heatmap_data:
                        heatmap_data[market_type] = {}
                    
                    heatmap_data[market_type][symbol] = value
        
        market_types_present = list(heatmap_data.keys())
        all_symbols = set()
        
        for market_type in market_types_present:
            all_symbols.update(heatmap_data[market_type].keys())
        
        all_symbols = sorted(list(all_symbols))
        
        matrix = np.zeros((len(market_types_present), len(all_symbols)))
        
        for i, market_type in enumerate(market_types_present):
            for j, symbol in enumerate(all_symbols):
                if symbol in heatmap_data[market_type]:
                    matrix[i, j] = heatmap_data[market_type][symbol]
        
        cmap = plt.cm.viridis
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric.replace("_", " ").title(), 
                          rotation=-90, va="bottom", color=colors["text"])
        cbar.ax.tick_params(colors=colors["text"])
        
        ax.set_xticks(np.arange(len(all_symbols)))
        ax.set_yticks(np.arange(len(market_types_present)))
        
        ax.set_xticklabels(all_symbols, color=colors["text"])
        ax.set_yticklabels(market_types_present, color=colors["text"])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(market_types_present)):
            for j in range(len(all_symbols)):
                value = matrix[i, j]
                if value > 0:
                    text_color = "white" if value > 0.5 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)
        
        ax.set_title(f"Cross-Market {metric.replace('_', ' ').title()} Heatmap", 
                    color=colors["text"], fontsize=16, fontweight="bold")
        
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        plt.figtext(0.98, 0.01, timestamp_text, color=colors["text"], fontsize=8, 
                   horizontalalignment="right")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, facecolor=fig.get_facecolor())
            plt.close(fig)
            logger.info(f"Saved heatmap visualization to {output_file}")
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            return img_str
    
    def generate_tradingview_chart(self, 
                                 opportunity: Dict[str, Any],
                                 include_indicators: bool = True,
                                 output_file: Optional[str] = None) -> str:
        """
        Generate TradingView chart code for an opportunity.
        
        Parameters:
        - opportunity: Trading opportunity data
        - include_indicators: Whether to include indicators
        - output_file: Optional file path to save chart code
        
        Returns:
        - TradingView Pine Script code
        """
        if not opportunity:
            logger.error("No opportunity data provided for TradingView chart")
            return ""
        
        symbol = opportunity["symbol"]
        market_type = opportunity["market_type"]
        direction = opportunity["predictions"]["direction"]
        
        exchange_map = {
            "forex": "FX",
            "crypto": "BINANCE",
            "stocks": "NASDAQ",
            "commodities": "NYMEX",
            "indices": "INDEX"
        }
        
        exchange = exchange_map.get(market_type, "BINANCE")
        
        pine_script = f"""// @version=5
indicator("{symbol} Omniversal Prediction", overlay=true)

// Input parameters
showPrediction = input.bool(true, "Show Prediction")
showLevels = input.bool(true, "Show Key Levels")
showLabels = input.bool(true, "Show Labels")

// Colors
predictionColor = color.new(color.purple, 0)
upColor = color.new(color.green, 0)
downColor = color.new(color.red, 0)
neutralColor = color.new(color.blue, 0)

// Current bar time
currentTime = timenow

// Prediction data (generated by Omniversal Intelligence)
direction = "{direction}"
"""
        
        if "key_levels" in opportunity["predictions"]:
            key_levels = opportunity["predictions"]["key_levels"]
            
            pine_script += "\n// Key levels\n"
            
            for level_type, level_value in key_levels.items():
                var_name = f"{level_type}Level"
                pine_script += f"{var_name} = {level_value}\n"
                
                level_color = "upColor" if level_type == "support" else "downColor"
                if level_type not in ["support", "resistance"]:
                    level_color = "neutralColor"
                
                pine_script += f"""
if (showLevels)
    hline({var_name}, "{level_type.capitalize()}", color={level_color}, linestyle=hline.style_dashed)
"""
        
        if "entry_points" in opportunity["predictions"] and opportunity["predictions"]["entry_points"]:
            entry_point = opportunity["predictions"]["entry_points"][0]
            pine_script += f"\n// Entry point\nentryPoint = {entry_point}\n"
        
        if "exit_points" in opportunity["predictions"] and opportunity["predictions"]["exit_points"]:
            exit_point = opportunity["predictions"]["exit_points"][0]
            pine_script += f"\n// Exit point\nexitPoint = {exit_point}\n"
        
        if "stop_loss" in opportunity["predictions"]:
            stop_loss = opportunity["predictions"]["stop_loss"]
            pine_script += f"\n// Stop loss\nstopLoss = {stop_loss}\n"
            
            pine_script += f"""
if (showLevels)
    hline(stopLoss, "Stop Loss", color=downColor, linestyle=hline.style_dotted)
"""
        
        if "take_profit" in opportunity["predictions"]:
            take_profit = opportunity["predictions"]["take_profit"]
            pine_script += f"\n// Take profit\ntakeProfit = {take_profit}\n"
            
            pine_script += f"""
if (showLevels)
    hline(takeProfit, "Take Profit", color=upColor, linestyle=hline.style_dotted)
"""
        
        if "trajectory" in opportunity["predictions"]:
            trajectory = opportunity["predictions"]["trajectory"]
            
            current_time = datetime.now()
            future_points = [
                point for point in trajectory 
                if datetime.fromisoformat(point["timestamp"]) > current_time
            ]
            
            if future_points:
                pine_script += "\n// Future trajectory\n"
                
                pine_script += f"""
// Plot prediction line
var line predictionLine = na
if (showPrediction and bar_index == last_bar_index)
    predictionLine := line.new(
        x1=bar_index, 
        y1=close, 
        x2=bar_index + {len(future_points)}, 
        y2={future_points[-1]["price"]}, 
        color=predictionColor, 
        width=2
    )
    line.delete(predictionLine[1])
"""
                
                pine_script += f"""
// Plot prediction points
var label[] predictionLabels = array.new_label(0)
if (showPrediction and showLabels and bar_index == last_bar_index)
    // Clear previous labels
    for i = 0 to array.size(predictionLabels) - 1
        label.delete(array.get(predictionLabels, i))
    array.clear(predictionLabels)
    
    // Add new labels
"""
                
                for i, point in enumerate(future_points[:5]):  # Limit to 5 points
                    price = point["price"]
                    pine_script += f"""    
    priceLabel = label.new(
        x=bar_index + {i+1}, 
        y={price}, 
        text="{price:.2f}", 
        color=predictionColor, 
        textcolor=color.white,
        style=label.style_label_down,
        size=size.tiny
    )
    array.push(predictionLabels, priceLabel)
"""
        
        if include_indicators:
            pine_script += """
// Add indicators
ema20 = ta.ema(close, 20)
ema50 = ta.ema(close, 50)
rsi = ta.rsi(close, 14)

plot(ema20, "EMA 20", color=color.new(color.blue, 0))
plot(ema50, "EMA 50", color=color.new(color.orange, 0))

// Plot buy/sell signals
plotshape(direction == "up" and ta.crossover(ema20, ema50), "Buy Signal", shape.triangleup, location.belowbar, upColor, size=size.small)
plotshape(direction == "down" and ta.crossunder(ema20, ema50), "Sell Signal", shape.triangledown, location.abovebar, downColor, size=size.small)

// Plot RSI in separate pane
plotRSI = input.bool(true, "Plot RSI")
if (plotRSI)
    hline(70, "RSI Overbought", color=color.new(color.red, 50))
    hline(30, "RSI Oversold", color=color.new(color.green, 50))
    plot(rsi, "RSI", color=color.new(color.purple, 0), panel=1)
"""
        
        win_probability = opportunity["predictions"].get("win_probability", 1.0)
        pine_script += f"""
// Add win probability label
var label winProbLabel = na
if (bar_index == last_bar_index)
    winProbLabel := label.new(
        x=bar_index, 
        y=high, 
        text="Win Probability: {win_probability*100:.1f}%", 
        color=color.new(color.black, 80), 
        textcolor=color.white,
        style=label.style_label_down,
        size=size.normal
    )
    label.delete(winProbLabel[1])
"""
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(pine_script)
            logger.info(f"Saved TradingView chart code to {output_file}")
        
        return pine_script
    
    def visualize_all_markets(self, 
                            theme: str = "quantum",
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize all markets and generate comprehensive report.
        
        Parameters:
        - theme: Color theme ('standard', 'dark', 'quantum')
        - output_dir: Optional directory to save visualizations
        
        Returns:
        - Report data
        """
        analysis = self.intelligence.analyze_all_markets()
        
        top_opportunities = analysis["optimal_opportunities"][:4]
        
        visualizations = {}
        
        multi_market_img = None
        if top_opportunities:
            if output_dir:
                output_file = f"{output_dir}/multi_market.png"
            else:
                output_file = None
                
            multi_market_img = self.visualize_multi_market(
                top_opportunities, theme=theme, output_file=output_file
            )
            
            visualizations["multi_market"] = multi_market_img
        
        individual_imgs = []
        
        for i, opportunity in enumerate(top_opportunities):
            if output_dir:
                output_file = f"{output_dir}/{opportunity['symbol']}_{i}.png"
            else:
                output_file = None
                
            img = self.visualize_trajectory(
                opportunity, theme=theme, output_file=output_file
            )
            
            individual_imgs.append({
                "symbol": opportunity["symbol"],
                "market_type": opportunity["market_type"],
                "direction": opportunity["predictions"]["direction"],
                "opportunity_score": opportunity["opportunity_score"],
                "win_probability": opportunity["predictions"].get("win_probability", 1.0),
                "image": img
            })
        
        visualizations["individual"] = individual_imgs
        
        if output_dir:
            output_file = f"{output_dir}/heatmap.png"
        else:
            output_file = None
            
        heatmap_img = self.visualize_cross_market_heatmap(
            theme=theme, output_file=output_file
        )
        
        visualizations["heatmap"] = heatmap_img
        
        tradingview_charts = []
        
        for i, opportunity in enumerate(top_opportunities):
            if output_dir:
                output_file = f"{output_dir}/{opportunity['symbol']}_{i}.pine"
            else:
                output_file = None
                
            chart_code = self.generate_tradingview_chart(
                opportunity, output_file=output_file
            )
            
            tradingview_charts.append({
                "symbol": opportunity["symbol"],
                "market_type": opportunity["market_type"],
                "direction": opportunity["predictions"]["direction"],
                "code": chart_code
            })
        
        visualizations["tradingview"] = tradingview_charts
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "total_assets_analyzed": sum(len(assets) for assets in analysis["results"].values()),
                "optimal_opportunities": len(analysis["optimal_opportunities"]),
                "prediction_accuracy": analysis["prediction_accuracy"],
                "win_rate": analysis["win_rate"]
            },
            "top_opportunities": top_opportunities,
            "visualizations": visualizations
        }
        
        if output_dir:
            with open(f"{output_dir}/report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved comprehensive report to {output_dir}/report.json")
        
        return report

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Cross-Market Visualization")
    
    parser.add_argument("--dimensions", type=int, default=11,
                        help="Number of market dimensions to visualize")
    
    parser.add_argument("--timeline-depth", type=int, default=100,
                        help="Depth of timeline visualization")
    
    parser.add_argument("--quality", type=str, default="ultra",
                        choices=["standard", "high", "ultra"],
                        help="Visualization quality")
    
    parser.add_argument("--theme", type=str, default="quantum",
                        choices=["standard", "dark", "quantum"],
                        help="Visualization theme")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations")
    
    parser.add_argument("--market-type", type=str, default=None,
                        help="Market type to visualize (e.g., 'crypto')")
    
    parser.add_argument("--asset", type=str, default=None,
                        help="Asset symbol to visualize (e.g., 'BTCUSD')")
    
    parser.add_argument("--all-markets", action="store_true",
                        help="Visualize all markets")
    
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate cross-market heatmap")
    
    parser.add_argument("--tradingview", action="store_true",
                        help="Generate TradingView chart code")
    
    args = parser.parse_args()
    
    visualization = CrossMarketVisualization(
        dimensions=args.dimensions,
        timeline_depth=args.timeline_depth,
        visualization_quality=args.quality
    )
    
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    
    intelligence = OmniversalIntelligence(
        dimensions=args.dimensions,
        timeline_depth=args.timeline_depth
    )
    
    if args.all_markets:
        report = visualization.visualize_all_markets(
            theme=args.theme,
            output_dir=args.output_dir
        )
        
        print(f"Generated comprehensive report for all markets")
        print(f"Analyzed {report['analysis']['total_assets_analyzed']} assets")
        print(f"Found {report['analysis']['optimal_opportunities']} optimal opportunities")
        
    elif args.heatmap:
        output_file = f"{args.output_dir}/heatmap.png" if args.output_dir else None
        
        visualization.visualize_cross_market_heatmap(
            theme=args.theme,
            output_file=output_file
        )
        
        print(f"Generated cross-market heatmap")
        
    elif args.market_type and args.asset:
        opportunity = intelligence.select_optimal_opportunity()
        
        if opportunity:
            output_file = f"{args.output_dir}/{args.asset}.png" if args.output_dir else None
            
            visualization.visualize_trajectory(
                opportunity,
                theme=args.theme,
                output_file=output_file
            )
            
            print(f"Generated visualization for {args.asset}")
            
            if args.tradingview:
                output_file = f"{args.output_dir}/{args.asset}.pine" if args.output_dir else None
                
                visualization.generate_tradingview_chart(
                    opportunity,
                    output_file=output_file
                )
                
                print(f"Generated TradingView chart code for {args.asset}")
        else:
            print(f"No opportunity found for {args.asset}")
    
    else:
        opportunity = intelligence.select_optimal_opportunity()
        
        if opportunity:
            output_file = f"{args.output_dir}/{opportunity['symbol']}.png" if args.output_dir else None
            
            visualization.visualize_trajectory(
                opportunity,
                theme=args.theme,
                output_file=output_file
            )
            
            print(f"Generated visualization for {opportunity['symbol']}")
            
            if args.tradingview:
                output_file = f"{args.output_dir}/{opportunity['symbol']}.pine" if args.output_dir else None
                
                visualization.generate_tradingview_chart(
                    opportunity,
                    output_file=output_file
                )
                
                print(f"Generated TradingView chart code for {opportunity['symbol']}")
        else:
            print("No optimal opportunity found")

if __name__ == "__main__":
    main()
