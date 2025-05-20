"""
Trade Logger Module

This module implements the Trade Logger for the QMP Overrider system.
It provides a comprehensive logging system for trades with detailed information.
"""

import os
import json
import logging
import datetime
import pandas as pd
from pathlib import Path
import uuid

class TradeLogger:
    """
    Trade Logger for the QMP Overrider system.
    
    This class provides a comprehensive logging system for trades with detailed information
    including entry/exit points, reasoning, confidence scores, and performance metrics.
    """
    
    def __init__(self, log_dir=None, log_format="csv"):
        """
        Initialize the Trade Logger.
        
        Parameters:
        - log_dir: Directory to store trade logs (or None for default)
        - log_format: Format for trade logs ("csv" or "json")
        """
        self.logger = logging.getLogger("TradeLogger")
        
        if log_dir is None:
            self.log_dir = Path("logs/trades")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_format = log_format.lower()
        if self.log_format not in ["csv", "json"]:
            self.logger.warning(f"Invalid log format: {log_format}. Using CSV.")
            self.log_format = "csv"
        
        self._init_trade_log()
        
        self.logger.info(f"Trade Logger initialized with format: {self.log_format}")
    
    def _init_trade_log(self):
        """Initialize the trade log file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        
        if self.log_format == "csv":
            self.log_file = self.log_dir / f"trade_log_{timestamp}.csv"
            
            if not self.log_file.exists():
                headers = [
                    "trade_id", "symbol", "direction", "entry_time", "entry_price",
                    "exit_time", "exit_price", "quantity", "pnl", "pnl_percent",
                    "trade_duration", "confidence_score", "signal_quality", "strategy",
                    "timeframe", "market_regime", "gate_scores", "reasoning",
                    "entry_candle_pattern", "exit_candle_pattern", "stop_loss",
                    "take_profit", "risk_reward_ratio", "slippage", "fees",
                    "market_conditions", "volatility", "volume", "spread",
                    "ai_prediction", "ai_confidence", "quantum_state", "emotion_state",
                    "timeline_state", "multiverse_alignment", "sacred_date",
                    "divine_timing", "big_move_compression", "macro_environment",
                    "tags", "notes"
                ]
                
                with open(self.log_file, "w") as f:
                    f.write(",".join(headers) + "\n")
        else:  # JSON format
            self.log_file = self.log_dir / f"trade_log_{timestamp}.json"
            
            if not self.log_file.exists():
                with open(self.log_file, "w") as f:
                    json.dump([], f)
    
    def log_trade(self, trade_data):
        """
        Log a trade to the trade log.
        
        Parameters:
        - trade_data: Dictionary containing trade data
        
        Returns:
        - trade_id: ID of the logged trade
        """
        if "trade_id" not in trade_data:
            trade_data["trade_id"] = str(uuid.uuid4())
        
        if "entry_time" not in trade_data:
            trade_data["entry_time"] = datetime.datetime.now().isoformat()
        
        if "pnl" not in trade_data and "entry_price" in trade_data and "exit_price" in trade_data:
            direction = trade_data.get("direction", "").upper()
            entry_price = float(trade_data["entry_price"])
            exit_price = float(trade_data["exit_price"])
            
            if direction == "BUY" or direction == "LONG":
                trade_data["pnl"] = exit_price - entry_price
                if entry_price > 0:
                    trade_data["pnl_percent"] = (exit_price - entry_price) / entry_price * 100
            elif direction == "SELL" or direction == "SHORT":
                trade_data["pnl"] = entry_price - exit_price
                if entry_price > 0:
                    trade_data["pnl_percent"] = (entry_price - exit_price) / entry_price * 100
        
        if "trade_duration" not in trade_data and "entry_time" in trade_data and "exit_time" in trade_data:
            try:
                entry_time = datetime.datetime.fromisoformat(trade_data["entry_time"])
                exit_time = datetime.datetime.fromisoformat(trade_data["exit_time"])
                duration = exit_time - entry_time
                trade_data["trade_duration"] = str(duration)
            except (ValueError, TypeError):
                pass
        
        if self.log_format == "csv":
            self._log_trade_csv(trade_data)
        else:  # JSON format
            self._log_trade_json(trade_data)
        
        self.logger.info(f"Trade logged: {trade_data['trade_id']} - {trade_data.get('symbol', 'Unknown')} - {trade_data.get('direction', 'Unknown')}")
        
        return trade_data["trade_id"]
    
    def _log_trade_csv(self, trade_data):
        """Log a trade to CSV file"""
        with open(self.log_file, "r") as f:
            headers = f.readline().strip().split(",")
        
        row_data = []
        for header in headers:
            if header in trade_data:
                value = trade_data[header]
                if isinstance(value, dict):
                    value = json.dumps(value).replace(",", ";").replace("\"", "'")
                elif isinstance(value, list):
                    value = json.dumps(value).replace(",", ";").replace("\"", "'")
                elif isinstance(value, bool):
                    value = "1" if value else "0"
                row_data.append(str(value))
            else:
                row_data.append("")
        
        with open(self.log_file, "a") as f:
            f.write(",".join(row_data) + "\n")
    
    def _log_trade_json(self, trade_data):
        """Log a trade to JSON file"""
        with open(self.log_file, "r") as f:
            trades = json.load(f)
        
        trades.append(trade_data)
        
        with open(self.log_file, "w") as f:
            json.dump(trades, f, indent=2)
    
    def get_trades(self, symbol=None, start_date=None, end_date=None, direction=None, profitable=None):
        """
        Get trades from the trade log.
        
        Parameters:
        - symbol: Filter by symbol
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        - direction: Filter by direction ("BUY" or "SELL")
        - profitable: Filter by profitability (True or False)
        
        Returns:
        - List of trades or DataFrame
        """
        if self.log_format == "csv":
            return self._get_trades_csv(symbol, start_date, end_date, direction, profitable)
        else:  # JSON format
            return self._get_trades_json(symbol, start_date, end_date, direction, profitable)
    
    def _get_trades_csv(self, symbol, start_date, end_date, direction, profitable):
        """Get trades from CSV file"""
        try:
            df = pd.read_csv(self.log_file)
            
            if symbol:
                df = df[df["symbol"] == symbol]
            
            if start_date:
                df = df[pd.to_datetime(df["entry_time"]) >= pd.to_datetime(start_date)]
            
            if end_date:
                df = df[pd.to_datetime(df["entry_time"]) <= pd.to_datetime(end_date)]
            
            if direction:
                df = df[df["direction"].str.upper() == direction.upper()]
            
            if profitable is not None:
                if profitable:
                    df = df[df["pnl"] > 0]
                else:
                    df = df[df["pnl"] <= 0]
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting trades from CSV: {e}")
            return pd.DataFrame()
    
    def _get_trades_json(self, symbol, start_date, end_date, direction, profitable):
        """Get trades from JSON file"""
        try:
            with open(self.log_file, "r") as f:
                trades = json.load(f)
            
            filtered_trades = trades
            
            if symbol:
                filtered_trades = [t for t in filtered_trades if t.get("symbol") == symbol]
            
            if start_date:
                start_dt = datetime.datetime.fromisoformat(start_date)
                filtered_trades = [t for t in filtered_trades if "entry_time" in t and datetime.datetime.fromisoformat(t["entry_time"]) >= start_dt]
            
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date)
                filtered_trades = [t for t in filtered_trades if "entry_time" in t and datetime.datetime.fromisoformat(t["entry_time"]) <= end_dt]
            
            if direction:
                filtered_trades = [t for t in filtered_trades if t.get("direction", "").upper() == direction.upper()]
            
            if profitable is not None:
                if profitable:
                    filtered_trades = [t for t in filtered_trades if "pnl" in t and float(t["pnl"]) > 0]
                else:
                    filtered_trades = [t for t in filtered_trades if "pnl" in t and float(t["pnl"]) <= 0]
            
            return filtered_trades
        except Exception as e:
            self.logger.error(f"Error getting trades from JSON: {e}")
            return []
    
    def get_trade_statistics(self, symbol=None, start_date=None, end_date=None):
        """
        Get statistics for trades.
        
        Parameters:
        - symbol: Filter by symbol
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        
        Returns:
        - Dictionary of trade statistics
        """
        trades = self.get_trades(symbol, start_date, end_date)
        
        if isinstance(trades, pd.DataFrame):
            if trades.empty:
                return self._empty_statistics()
            
            total_trades = len(trades)
            winning_trades = len(trades[trades["pnl"] > 0])
            losing_trades = len(trades[trades["pnl"] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = trades[trades["pnl"] > 0]["pnl"].sum()
            total_loss = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_profit = trades[trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades[trades["pnl"] <= 0]["pnl"].mean()) if losing_trades > 0 else 0
            
            expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
            
            by_symbol = {}
            for sym, group in trades.groupby("symbol"):
                sym_total = len(group)
                sym_winning = len(group[group["pnl"] > 0])
                sym_win_rate = sym_winning / sym_total if sym_total > 0 else 0
                sym_profit = group["pnl"].sum()
                
                by_symbol[sym] = {
                    "total_trades": sym_total,
                    "winning_trades": sym_winning,
                    "win_rate": sym_win_rate,
                    "total_profit": sym_profit
                }
            
            by_strategy = {}
            if "strategy" in trades.columns:
                for strat, group in trades.groupby("strategy"):
                    if pd.isna(strat):
                        continue
                    
                    strat_total = len(group)
                    strat_winning = len(group[group["pnl"] > 0])
                    strat_win_rate = strat_winning / strat_total if strat_total > 0 else 0
                    strat_profit = group["pnl"].sum()
                    
                    by_strategy[strat] = {
                        "total_trades": strat_total,
                        "winning_trades": strat_winning,
                        "win_rate": strat_win_rate,
                        "total_profit": strat_profit
                    }
        else:  # JSON format
            if not trades:
                return self._empty_statistics()
            
            total_trades = len(trades)
            winning_trades = len([t for t in trades if "pnl" in t and float(t["pnl"]) > 0])
            losing_trades = len([t for t in trades if "pnl" in t and float(t["pnl"]) <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum([float(t["pnl"]) for t in trades if "pnl" in t and float(t["pnl"]) > 0])
            total_loss = abs(sum([float(t["pnl"]) for t in trades if "pnl" in t and float(t["pnl"]) <= 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
            
            by_symbol = {}
            symbols = set([t.get("symbol") for t in trades if "symbol" in t])
            for sym in symbols:
                sym_trades = [t for t in trades if t.get("symbol") == sym]
                sym_total = len(sym_trades)
                sym_winning = len([t for t in sym_trades if "pnl" in t and float(t["pnl"]) > 0])
                sym_win_rate = sym_winning / sym_total if sym_total > 0 else 0
                sym_profit = sum([float(t["pnl"]) for t in sym_trades if "pnl" in t])
                
                by_symbol[sym] = {
                    "total_trades": sym_total,
                    "winning_trades": sym_winning,
                    "win_rate": sym_win_rate,
                    "total_profit": sym_profit
                }
            
            by_strategy = {}
            strategies = set([t.get("strategy") for t in trades if "strategy" in t and t.get("strategy")])
            for strat in strategies:
                strat_trades = [t for t in trades if t.get("strategy") == strat]
                strat_total = len(strat_trades)
                strat_winning = len([t for t in strat_trades if "pnl" in t and float(t["pnl"]) > 0])
                strat_win_rate = strat_winning / strat_total if strat_total > 0 else 0
                strat_profit = sum([float(t["pnl"]) for t in strat_trades if "pnl" in t])
                
                by_strategy[strat] = {
                    "total_trades": strat_total,
                    "winning_trades": strat_winning,
                    "win_rate": strat_win_rate,
                    "total_profit": strat_profit
                }
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": float(total_profit),
            "total_loss": float(total_loss),
            "net_profit": float(total_profit - total_loss),
            "profit_factor": profit_factor,
            "avg_profit": float(avg_profit),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy
        }
    
    def _empty_statistics(self):
        """Return empty statistics"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "total_loss": 0,
            "net_profit": 0,
            "profit_factor": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "expectancy": 0,
            "by_symbol": {},
            "by_strategy": {}
        }
    
    def export_trades(self, output_file, format=None):
        """
        Export trades to a file.
        
        Parameters:
        - output_file: Path to output file
        - format: Format to export to (or None for same as log_format)
        
        Returns:
        - True if successful, False otherwise
        """
        if format is None:
            format = self.log_format
        
        format = format.lower()
        if format not in ["csv", "json", "excel"]:
            self.logger.warning(f"Invalid export format: {format}. Using {self.log_format}.")
            format = self.log_format
        
        trades = self.get_trades()
        
        try:
            if isinstance(trades, pd.DataFrame):
                if format == "csv":
                    trades.to_csv(output_file, index=False)
                elif format == "json":
                    trades.to_json(output_file, orient="records", indent=2)
                elif format == "excel":
                    trades.to_excel(output_file, index=False)
            else:  # JSON format
                if format == "csv":
                    pd.DataFrame(trades).to_csv(output_file, index=False)
                elif format == "json":
                    with open(output_file, "w") as f:
                        json.dump(trades, f, indent=2)
                elif format == "excel":
                    pd.DataFrame(trades).to_excel(output_file, index=False)
            
            self.logger.info(f"Trades exported to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting trades: {e}")
            return False
    
    def get_trade_by_id(self, trade_id):
        """
        Get a trade by ID.
        
        Parameters:
        - trade_id: ID of the trade
        
        Returns:
        - Trade data or None if not found
        """
        trades = self.get_trades()
        
        if isinstance(trades, pd.DataFrame):
            trade = trades[trades["trade_id"] == trade_id]
            if trade.empty:
                return None
            return trade.iloc[0].to_dict()
        else:  # JSON format
            for trade in trades:
                if trade.get("trade_id") == trade_id:
                    return trade
            return None
    
    def update_trade(self, trade_id, update_data):
        """
        Update a trade in the trade log.
        
        Parameters:
        - trade_id: ID of the trade to update
        - update_data: Dictionary of data to update
        
        Returns:
        - True if successful, False otherwise
        """
        if self.log_format == "csv":
            return self._update_trade_csv(trade_id, update_data)
        else:  # JSON format
            return self._update_trade_json(trade_id, update_data)
    
    def _update_trade_csv(self, trade_id, update_data):
        """Update a trade in CSV file"""
        try:
            df = pd.read_csv(self.log_file)
            
            trade_idx = df[df["trade_id"] == trade_id].index
            if len(trade_idx) == 0:
                self.logger.warning(f"Trade not found: {trade_id}")
                return False
            
            for key, value in update_data.items():
                if key in df.columns:
                    df.loc[trade_idx, key] = value
            
            df.to_csv(self.log_file, index=False)
            
            self.logger.info(f"Trade updated: {trade_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating trade in CSV: {e}")
            return False
    
    def _update_trade_json(self, trade_id, update_data):
        """Update a trade in JSON file"""
        try:
            with open(self.log_file, "r") as f:
                trades = json.load(f)
            
            for i, trade in enumerate(trades):
                if trade.get("trade_id") == trade_id:
                    trades[i].update(update_data)
                    
                    with open(self.log_file, "w") as f:
                        json.dump(trades, f, indent=2)
                    
                    self.logger.info(f"Trade updated: {trade_id}")
                    return True
            
            self.logger.warning(f"Trade not found: {trade_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating trade in JSON: {e}")
            return False
    
    def delete_trade(self, trade_id):
        """
        Delete a trade from the trade log.
        
        Parameters:
        - trade_id: ID of the trade to delete
        
        Returns:
        - True if successful, False otherwise
        """
        if self.log_format == "csv":
            return self._delete_trade_csv(trade_id)
        else:  # JSON format
            return self._delete_trade_json(trade_id)
    
    def _delete_trade_csv(self, trade_id):
        """Delete a trade from CSV file"""
        try:
            df = pd.read_csv(self.log_file)
            
            trade_idx = df[df["trade_id"] == trade_id].index
            if len(trade_idx) == 0:
                self.logger.warning(f"Trade not found: {trade_id}")
                return False
            
            df = df.drop(trade_idx)
            
            df.to_csv(self.log_file, index=False)
            
            self.logger.info(f"Trade deleted: {trade_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting trade from CSV: {e}")
            return False
    
    def _delete_trade_json(self, trade_id):
        """Delete a trade from JSON file"""
        try:
            with open(self.log_file, "r") as f:
                trades = json.load(f)
            
            for i, trade in enumerate(trades):
                if trade.get("trade_id") == trade_id:
                    del trades[i]
                    
                    with open(self.log_file, "w") as f:
                        json.dump(trades, f, indent=2)
                    
                    self.logger.info(f"Trade deleted: {trade_id}")
                    return True
            
            self.logger.warning(f"Trade not found: {trade_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting trade from JSON: {e}")
            return False
