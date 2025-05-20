"""
Heat-Latency Pipeline Module

This module implements a combined pipeline for Regulatory Heat Imprints and Transactional Latency Fingerprints,
providing a unified system for detecting both regulatory attention and hidden liquidity patterns.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import time

from market_intelligence.regulatory_heat_imprints import RegulatoryHeatImprints
from market_intelligence.transactional_latency_fingerprints import TransactionalLatencyFingerprints

class HeatLatencyPipeline:
    """
    Combined Heat-Latency Pipeline for the QMP Overrider system.
    
    This class integrates Regulatory Heat Imprints with Transactional Latency Fingerprints,
    providing a unified system for detecting both regulatory attention and hidden liquidity patterns.
    """
    
    def __init__(self, log_dir=None, heat_threshold=0.65, latency_threshold=2.5):
        """
        Initialize the Heat-Latency Pipeline.
        
        Parameters:
        - log_dir: Directory to store pipeline logs (or None for default)
        - heat_threshold: Threshold for high regulatory heat detection (0.0 to 1.0)
        - latency_threshold: Z-score threshold for latency anomaly detection
        """
        self.logger = logging.getLogger("HeatLatencyPipeline")
        
        if log_dir is None:
            self.log_dir = Path("logs/heat_latency_pipeline")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.heat_system = RegulatoryHeatImprints(
            log_dir=self.log_dir / "regulatory_heat",
            heat_threshold=heat_threshold
        )
        
        self.latency_system = TransactionalLatencyFingerprints(
            log_dir=self.log_dir / "latency_fingerprints",
            anomaly_threshold=latency_threshold
        )
        
        self.combined_alerts = []
        self.symbol_risk_scores = {}
        self.risk_history = {}
        
        self._load_data()
        
        self.logger.info(f"Heat-Latency Pipeline initialized with heat threshold: {heat_threshold}, latency threshold: {latency_threshold}")
    
    def _load_data(self):
        """Load existing pipeline data"""
        alerts_file = self.log_dir / "combined_alerts.json"
        scores_file = self.log_dir / "risk_scores.json"
        history_file = self.log_dir / "risk_history.json"
        
        if alerts_file.exists():
            try:
                with open(alerts_file, "r") as f:
                    self.combined_alerts = json.load(f)
                
                self.logger.info(f"Loaded {len(self.combined_alerts)} combined alerts")
            except Exception as e:
                self.logger.error(f"Error loading combined alerts: {e}")
        
        if scores_file.exists():
            try:
                with open(scores_file, "r") as f:
                    self.symbol_risk_scores = json.load(f)
                
                self.logger.info(f"Loaded risk scores for {len(self.symbol_risk_scores)} symbols")
            except Exception as e:
                self.logger.error(f"Error loading risk scores: {e}")
        
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.risk_history = json.load(f)
                
                self.logger.info(f"Loaded risk history for {len(self.risk_history)} symbols")
            except Exception as e:
                self.logger.error(f"Error loading risk history: {e}")
    
    def _save_data(self):
        """Save pipeline data to file"""
        alerts_file = self.log_dir / "combined_alerts.json"
        scores_file = self.log_dir / "risk_scores.json"
        history_file = self.log_dir / "risk_history.json"
        
        try:
            with open(alerts_file, "w") as f:
                json.dump(self.combined_alerts, f, indent=2)
            
            with open(scores_file, "w") as f:
                json.dump(self.symbol_risk_scores, f, indent=2)
            
            with open(history_file, "w") as f:
                json.dump(self.risk_history, f, indent=2)
            
            self.logger.info("Saved pipeline data")
        except Exception as e:
            self.logger.error(f"Error saving pipeline data: {e}")
    
    def analyze_symbol(self, symbol: str, transaction_data: Optional[Dict[str, Any]] = None, 
                      heat_source_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze a symbol using both regulatory heat and latency fingerprints.
        
        Parameters:
        - symbol: Symbol to analyze
        - transaction_data: Optional transaction data for latency analysis
        - heat_source_data: Optional heat source data for regulatory heat analysis
        
        Returns:
        - Dictionary with combined analysis results
        """
        timestamp = datetime.now().isoformat()
        
        heat_result = self.heat_system.calculate_heat_score(symbol, timestamp, heat_source_data)
        
        latency_result = None
        if transaction_data and "timestamps" in transaction_data:
            timestamps = transaction_data.get("timestamps", [])
            venues = transaction_data.get("venues", None)
            
            if len(timestamps) >= 3:
                latency_result = self.latency_system.analyze_transaction_latencies(
                    symbol, timestamps, venues
                )
        
        analysis = {
            "symbol": symbol,
            "timestamp": timestamp,
            "heat_analysis": heat_result,
            "latency_analysis": latency_result
        }
        
        risk_score = self._calculate_combined_risk(heat_result, latency_result)
        analysis["risk_score"] = risk_score
        
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        elif risk_score >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        analysis["risk_level"] = risk_level
        
        alerts = self._check_for_combined_alerts(symbol, heat_result, latency_result, risk_score)
        analysis["alerts"] = alerts
        
        self.symbol_risk_scores[symbol] = {
            "timestamp": timestamp,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "heat_score": heat_result["total_score"] if heat_result else 0.0,
            "latency_anomalies": latency_result["anomaly_count"] if latency_result else 0
        }
        
        if symbol not in self.risk_history:
            self.risk_history[symbol] = []
        
        history_entry = {
            "timestamp": timestamp,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "heat_score": heat_result["total_score"] if heat_result else 0.0,
            "latency_anomalies": latency_result["anomaly_count"] if latency_result else 0
        }
        
        self.risk_history[symbol].append(history_entry)
        
        if len(self.risk_history[symbol]) > 1000:
            self.risk_history[symbol] = self.risk_history[symbol][-1000:]
        
        self._save_data()
        
        return analysis
    
    def _calculate_combined_risk(self, heat_result: Optional[Dict[str, Any]], 
                                latency_result: Optional[Dict[str, Any]]) -> float:
        """
        Calculate combined risk score from heat and latency results.
        
        Parameters:
        - heat_result: Regulatory heat analysis result
        - latency_result: Latency fingerprints analysis result
        
        Returns:
        - Combined risk score (0.0 to 1.0)
        """
        heat_score = 0.0
        latency_score = 0.0
        
        if heat_result:
            heat_score = heat_result["total_score"]
        
        if latency_result:
            anomaly_count = latency_result["anomaly_count"]
            total_transactions = latency_result["num_transactions"]
            
            if total_transactions > 0:
                anomaly_ratio = min(1.0, anomaly_count / (total_transactions * 0.2))
                latency_score = anomaly_ratio
                
                if "patterns" in latency_result:
                    patterns = latency_result["patterns"]
                    
                    if "periodicity" in patterns and patterns["periodicity"].get("detected", False):
                        latency_score += 0.2
                    
                    if "clustering" in patterns and patterns["clustering"].get("detected", False):
                        latency_score += 0.2
                
                latency_score = min(1.0, latency_score)
        
        if heat_result and latency_result:
            combined_score = (heat_score * 0.6) + (latency_score * 0.4)
        elif heat_result:
            combined_score = heat_score * 0.8  # Reduce confidence without latency data
        elif latency_result:
            combined_score = latency_score * 0.7  # Reduce confidence without heat data
        else:
            combined_score = 0.0
        
        return combined_score
    
    def _check_for_combined_alerts(self, symbol: str, heat_result: Optional[Dict[str, Any]], 
                                  latency_result: Optional[Dict[str, Any]], 
                                  risk_score: float) -> List[Dict[str, Any]]:
        """
        Check for combined alerts based on heat and latency results.
        
        Parameters:
        - symbol: Symbol being analyzed
        - heat_result: Regulatory heat analysis result
        - latency_result: Latency fingerprints analysis result
        - risk_score: Combined risk score
        
        Returns:
        - List of alert dictionaries
        """
        alerts = []
        
        if risk_score >= 0.8:
            alerts.append({
                "type": "critical_combined_risk",
                "description": f"Critical combined risk detected for {symbol}",
                "risk_score": risk_score,
                "timestamp": datetime.now().isoformat()
            })
        elif risk_score >= 0.6:
            alerts.append({
                "type": "high_combined_risk",
                "description": f"High combined risk detected for {symbol}",
                "risk_score": risk_score,
                "timestamp": datetime.now().isoformat()
            })
        
        if heat_result and latency_result:
            if heat_result["total_score"] >= 0.7 and latency_result["anomaly_count"] > 0:
                alerts.append({
                    "type": "heat_latency_correlation",
                    "description": f"Correlated high heat and latency anomalies for {symbol}",
                    "heat_score": heat_result["total_score"],
                    "latency_anomalies": latency_result["anomaly_count"],
                    "timestamp": datetime.now().isoformat()
                })
        
        if symbol in self.risk_history and len(self.risk_history[symbol]) >= 2:
            prev_entries = self.risk_history[symbol][-5:-1]  # Last 4 entries excluding current
            if prev_entries:
                avg_prev_heat = sum(entry["heat_score"] for entry in prev_entries) / len(prev_entries)
                current_heat = heat_result["total_score"] if heat_result else 0.0
                
                if current_heat > avg_prev_heat * 1.5 and current_heat >= 0.5:
                    alerts.append({
                        "type": "rapid_heat_increase",
                        "description": f"Rapid increase in regulatory heat for {symbol}",
                        "current_heat": current_heat,
                        "previous_avg_heat": avg_prev_heat,
                        "percent_increase": (current_heat - avg_prev_heat) / avg_prev_heat * 100 if avg_prev_heat > 0 else 100,
                        "timestamp": datetime.now().isoformat()
                    })
        
        for alert in alerts:
            alert["symbol"] = symbol
            self.combined_alerts.append(alert)
        
        if len(self.combined_alerts) > 1000:
            self.combined_alerts = self.combined_alerts[-1000:]
        
        return alerts
    
    def detect_dark_pool_leakage_with_heat(self, symbol: str, timestamps: List[float], 
                                          venues: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect dark pool leakage with regulatory heat context.
        
        Parameters:
        - symbol: Symbol to analyze
        - timestamps: List of transaction timestamps (in milliseconds)
        - venues: Optional list of venues corresponding to each timestamp
        
        Returns:
        - Dictionary with dark pool leakage detection results including regulatory context
        """
        leakage_result = self.latency_system.detect_dark_pool_leakage(symbol, timestamps, venues)
        
        heat_result = self.heat_system.calculate_heat_score(symbol)
        
        combined_result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "leakage_detection": leakage_result,
            "regulatory_heat": heat_result
        }
        
        leakage_score = leakage_result["leakage_score"]
        heat_score = heat_result["total_score"]
        
        combined_score = (leakage_score * 0.7) + (heat_score * 0.3)
        combined_result["combined_score"] = combined_score
        
        high_risk = combined_score >= 0.7
        combined_result["high_risk"] = high_risk
        
        risk_factors = []
        
        if leakage_result["leakage_detected"]:
            risk_factors.append({
                "type": "dark_pool_leakage",
                "description": "Dark pool leakage detected",
                "score": leakage_score,
                "contribution": leakage_score * 0.7
            })
        
        if heat_result["high_heat"]:
            risk_factors.append({
                "type": "high_regulatory_heat",
                "description": "High regulatory attention detected",
                "score": heat_score,
                "contribution": heat_score * 0.3
            })
        
        combined_result["risk_factors"] = risk_factors
        
        if high_risk:
            alert = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "type": "dark_pool_leakage_with_heat",
                "description": f"Dark pool leakage detected with regulatory context for {symbol}",
                "combined_score": combined_score,
                "leakage_score": leakage_score,
                "heat_score": heat_score
            }
            
            self.combined_alerts.append(alert)
            
            if len(self.combined_alerts) > 1000:
                self.combined_alerts = self.combined_alerts[-1000:]
            
            self._save_data()
        
        return combined_result
    
    def predict_regulatory_action_with_latency(self, symbol: str, days_ahead: int = 30,
                                             transaction_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict regulatory action with latency fingerprint context.
        
        Parameters:
        - symbol: Symbol to predict for
        - days_ahead: Number of days ahead to predict
        - transaction_data: Optional transaction data for latency analysis
        
        Returns:
        - Dictionary with regulatory action prediction including latency context
        """
        prediction = self.heat_system.predict_regulatory_action(symbol, days_ahead)
        
        latency_result = None
        if transaction_data and "timestamps" in transaction_data:
            timestamps = transaction_data.get("timestamps", [])
            venues = transaction_data.get("venues", None)
            
            if len(timestamps) >= 3:
                latency_result = self.latency_system.analyze_transaction_latencies(
                    symbol, timestamps, venues
                )
        
        combined_result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "regulatory_prediction": prediction,
            "latency_analysis": latency_result
        }
        
        if latency_result and "anomaly_count" in latency_result and latency_result["anomaly_count"] > 0:
            anomaly_ratio = min(1.0, latency_result["anomaly_count"] / (latency_result["num_transactions"] * 0.2))
            
            original_probability = prediction["action_probability"]
            adjusted_probability = min(0.99, original_probability * (1 + anomaly_ratio * 0.5))
            
            combined_result["original_probability"] = original_probability
            combined_result["adjusted_probability"] = adjusted_probability
            combined_result["adjustment_factor"] = 1 + anomaly_ratio * 0.5
            combined_result["adjustment_reason"] = f"Latency anomalies detected ({latency_result['anomaly_count']} anomalies)"
        else:
            combined_result["adjusted_probability"] = prediction["action_probability"]
            combined_result["adjustment_factor"] = 1.0
        
        return combined_result
    
    def get_combined_alerts(self, symbol: Optional[str] = None, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           min_risk: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get combined alerts.
        
        Parameters:
        - symbol: Symbol to filter by (or None for all symbols)
        - start_date: Start date for alerts (ISO format)
        - end_date: End date for alerts (ISO format)
        - min_risk: Minimum risk score to include
        
        Returns:
        - List of alert dictionaries
        """
        filtered_alerts = []
        
        for alert in self.combined_alerts:
            if symbol and alert["symbol"] != symbol:
                continue
            
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
                if datetime.fromisoformat(alert["timestamp"]) < start_dt:
                    continue
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
                if datetime.fromisoformat(alert["timestamp"]) > end_dt:
                    continue
            
            if "combined_score" in alert and alert["combined_score"] < min_risk:
                continue
            elif "risk_score" in alert and alert["risk_score"] < min_risk:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_risk_history(self, symbol: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get risk history for a symbol.
        
        Parameters:
        - symbol: Symbol to get history for
        - start_date: Start date for history (ISO format)
        - end_date: End date for history (ISO format)
        
        Returns:
        - List of risk history entries
        """
        if symbol not in self.risk_history:
            return []
        
        filtered_history = []
        
        for entry in self.risk_history[symbol]:
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
                if datetime.fromisoformat(entry["timestamp"]) < start_dt:
                    continue
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
                if datetime.fromisoformat(entry["timestamp"]) > end_dt:
                    continue
            
            filtered_history.append(entry)
        
        return filtered_history
    
    def plot_combined_risk(self, symbol: str, output_file: Optional[str] = None, 
                          show: bool = False) -> bool:
        """
        Plot combined risk history for a symbol.
        
        Parameters:
        - symbol: Symbol to plot history for
        - output_file: Path to save plot (or None to not save)
        - show: Whether to show the plot
        
        Returns:
        - True if successful, False otherwise
        """
        if symbol not in self.risk_history or not self.risk_history[symbol]:
            self.logger.warning(f"No risk history for symbol: {symbol}")
            return False
        
        try:
            history = self.risk_history[symbol]
            timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in history]
            risk_scores = [entry["risk_score"] for entry in history]
            heat_scores = [entry["heat_score"] for entry in history]
            latency_anomalies = [entry["latency_anomalies"] for entry in history]
            
            max_anomalies = max(latency_anomalies) if latency_anomalies else 1
            normalized_anomalies = [a / max_anomalies if max_anomalies > 0 else 0 for a in latency_anomalies]
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            ax1.plot(timestamps, risk_scores, 'r-', linewidth=2, label='Combined Risk Score')
            ax1.plot(timestamps, heat_scores, 'b--', alpha=0.7, label='Regulatory Heat Score')
            
            ax2 = ax1.twinx()
            ax2.bar(timestamps, normalized_anomalies, alpha=0.3, color='g', label='Latency Anomalies')
            
            alerts = self.get_combined_alerts(symbol)
            if alerts:
                alert_times = [datetime.fromisoformat(alert["timestamp"]) for alert in alerts]
                alert_scores = []
                
                for alert in alerts:
                    if "risk_score" in alert:
                        alert_scores.append(alert["risk_score"])
                    elif "combined_score" in alert:
                        alert_scores.append(alert["combined_score"])
                    else:
                        alert_time = datetime.fromisoformat(alert["timestamp"])
                        closest_idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - alert_time).total_seconds()))
                        alert_scores.append(risk_scores[closest_idx])
                
                ax1.scatter(alert_times, alert_scores, color='red', s=100, zorder=5, label='Alerts')
            
            ax1.set_xlabel('Timestamp')
            ax1.set_ylabel('Risk/Heat Score', color='r')
            ax2.set_ylabel('Normalized Latency Anomalies', color='g')
            
            plt.title(f'Combined Risk Analysis for {symbol}')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Saved combined risk plot to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error plotting combined risk: {e}")
            return False
    
    def export_data(self, output_dir: Optional[str] = None, 
                   format: str = "json") -> bool:
        """
        Export pipeline data.
        
        Parameters:
        - output_dir: Directory to save exported data (or None for default)
        - format: Format to export data ("json", "csv", or "excel")
        
        Returns:
        - True if successful, False otherwise
        """
        if output_dir is None:
            output_dir = self.log_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "json":
                alerts_file = output_dir / "combined_alerts.json"
                with open(alerts_file, "w") as f:
                    json.dump(self.combined_alerts, f, indent=2)
                
                scores_file = output_dir / "risk_scores.json"
                with open(scores_file, "w") as f:
                    json.dump(self.symbol_risk_scores, f, indent=2)
                
                history_file = output_dir / "risk_history.json"
                with open(history_file, "w") as f:
                    json.dump(self.risk_history, f, indent=2)
            
            elif format.lower() == "csv":
                alerts_df = pd.DataFrame(self.combined_alerts)
                alerts_file = output_dir / "combined_alerts.csv"
                alerts_df.to_csv(alerts_file, index=False)
                
                scores_data = []
                for symbol, data in self.symbol_risk_scores.items():
                    entry = data.copy()
                    entry["symbol"] = symbol
                    scores_data.append(entry)
                
                scores_df = pd.DataFrame(scores_data)
                scores_file = output_dir / "risk_scores.csv"
                scores_df.to_csv(scores_file, index=False)
                
                history_data = []
                for symbol, entries in self.risk_history.items():
                    for entry in entries:
                        history_entry = entry.copy()
                        history_entry["symbol"] = symbol
                        history_data.append(history_entry)
                
                history_df = pd.DataFrame(history_data)
                history_file = output_dir / "risk_history.csv"
                history_df.to_csv(history_file, index=False)
            
            elif format.lower() == "excel":
                excel_file = output_dir / "heat_latency_pipeline.xlsx"
                with pd.ExcelWriter(excel_file) as writer:
                    alerts_df = pd.DataFrame(self.combined_alerts)
                    alerts_df.to_excel(writer, sheet_name="Combined Alerts", index=False)
                    
                    scores_data = []
                    for symbol, data in self.symbol_risk_scores.items():
                        entry = data.copy()
                        entry["symbol"] = symbol
                        scores_data.append(entry)
                    
                    scores_df = pd.DataFrame(scores_data)
                    scores_df.to_excel(writer, sheet_name="Risk Scores", index=False)
                    
                    history_data = []
                    for symbol, entries in self.risk_history.items():
                        for entry in entries:
                            history_entry = entry.copy()
                            history_entry["symbol"] = symbol
                            history_data.append(history_entry)
                    
                    history_df = pd.DataFrame(history_data)
                    history_df.to_excel(writer, sheet_name="Risk History", index=False)
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Exported pipeline data to {output_dir} in {format} format")
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting pipeline data: {e}")
            return False
