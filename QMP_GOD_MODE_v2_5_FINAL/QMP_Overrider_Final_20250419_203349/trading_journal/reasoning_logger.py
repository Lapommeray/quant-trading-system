"""
Reasoning Logger Module

This module implements the Reasoning Logger for the QMP Overrider system.
It provides a comprehensive logging system for trade reasoning with detailed information.
"""

import os
import json
import logging
import datetime
from pathlib import Path
import uuid

class ReasoningLogger:
    """
    Reasoning Logger for the QMP Overrider system.
    
    This class provides a comprehensive logging system for trade reasoning with detailed information
    about the decision-making process, including gate scores, AI predictions, and market conditions.
    """
    
    def __init__(self, log_dir=None, log_format="json"):
        """
        Initialize the Reasoning Logger.
        
        Parameters:
        - log_dir: Directory to store reasoning logs (or None for default)
        - log_format: Format for reasoning logs ("json" or "txt")
        """
        self.logger = logging.getLogger("ReasoningLogger")
        
        if log_dir is None:
            self.log_dir = Path("logs/reasoning")
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_format = log_format.lower()
        if self.log_format not in ["json", "txt"]:
            self.logger.warning(f"Invalid log format: {log_format}. Using JSON.")
            self.log_format = "json"
        
        self._init_reasoning_log()
        
        self.logger.info(f"Reasoning Logger initialized with format: {self.log_format}")
    
    def _init_reasoning_log(self):
        """Initialize the reasoning log file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        
        if self.log_format == "json":
            self.log_file = self.log_dir / f"reasoning_log_{timestamp}.json"
            
            if not self.log_file.exists():
                with open(self.log_file, "w") as f:
                    json.dump([], f)
        else:  # TXT format
            self.log_file = self.log_dir / f"reasoning_log_{timestamp}.txt"
            
            if not self.log_file.exists():
                with open(self.log_file, "w") as f:
                    f.write(f"# QMP Overrider Reasoning Log - {timestamp}\n\n")
    
    def log_reasoning(self, reasoning_data):
        """
        Log reasoning to the reasoning log.
        
        Parameters:
        - reasoning_data: Dictionary containing reasoning data
        
        Returns:
        - reasoning_id: ID of the logged reasoning
        """
        if "reasoning_id" not in reasoning_data:
            reasoning_data["reasoning_id"] = str(uuid.uuid4())
        
        if "timestamp" not in reasoning_data:
            reasoning_data["timestamp"] = datetime.datetime.now().isoformat()
        
        if self.log_format == "json":
            self._log_reasoning_json(reasoning_data)
        else:  # TXT format
            self._log_reasoning_txt(reasoning_data)
        
        self.logger.info(f"Reasoning logged: {reasoning_data['reasoning_id']} - {reasoning_data.get('symbol', 'Unknown')} - {reasoning_data.get('decision', 'Unknown')}")
        
        return reasoning_data["reasoning_id"]
    
    def _log_reasoning_json(self, reasoning_data):
        """Log reasoning to JSON file"""
        with open(self.log_file, "r") as f:
            reasoning_entries = json.load(f)
        
        reasoning_entries.append(reasoning_data)
        
        with open(self.log_file, "w") as f:
            json.dump(reasoning_entries, f, indent=2)
    
    def _log_reasoning_txt(self, reasoning_data):
        """Log reasoning to TXT file"""
        timestamp = reasoning_data.get("timestamp", datetime.datetime.now().isoformat())
        symbol = reasoning_data.get("symbol", "Unknown")
        decision = reasoning_data.get("decision", "Unknown")
        reasoning_id = reasoning_data.get("reasoning_id", "Unknown")
        
        header = f"## Reasoning ID: {reasoning_id}\n"
        header += f"## Timestamp: {timestamp}\n"
        header += f"## Symbol: {symbol}\n"
        header += f"## Decision: {decision}\n\n"
        
        gate_scores = reasoning_data.get("gate_scores", {})
        gate_scores_text = "### Gate Scores:\n"
        for gate, score in gate_scores.items():
            gate_scores_text += f"- {gate}: {score}\n"
        gate_scores_text += "\n"
        
        ai_predictions = reasoning_data.get("ai_predictions", {})
        ai_predictions_text = "### AI Predictions:\n"
        for model, prediction in ai_predictions.items():
            ai_predictions_text += f"- {model}: {prediction}\n"
        ai_predictions_text += "\n"
        
        market_conditions = reasoning_data.get("market_conditions", {})
        market_conditions_text = "### Market Conditions:\n"
        for condition, value in market_conditions.items():
            market_conditions_text += f"- {condition}: {value}\n"
        market_conditions_text += "\n"
        
        reasoning_steps = reasoning_data.get("reasoning_steps", [])
        reasoning_steps_text = "### Reasoning Steps:\n"
        for i, step in enumerate(reasoning_steps, 1):
            reasoning_steps_text += f"{i}. {step}\n"
        reasoning_steps_text += "\n"
        
        final_reasoning = reasoning_data.get("final_reasoning", "")
        final_reasoning_text = f"### Final Reasoning:\n{final_reasoning}\n\n"
        
        confidence = reasoning_data.get("confidence", 0.0)
        confidence_text = f"### Confidence: {confidence}\n\n"
        
        additional_data = reasoning_data.get("additional_data", {})
        additional_data_text = "### Additional Data:\n"
        for key, value in additional_data.items():
            if isinstance(value, dict) or isinstance(value, list):
                value = json.dumps(value, indent=2)
            additional_data_text += f"- {key}: {value}\n"
        additional_data_text += "\n"
        
        entry = header + gate_scores_text + ai_predictions_text + market_conditions_text + reasoning_steps_text + final_reasoning_text + confidence_text + additional_data_text
        entry += "---\n\n"
        
        with open(self.log_file, "a") as f:
            f.write(entry)
    
    def get_reasoning(self, reasoning_id=None, symbol=None, start_date=None, end_date=None, decision=None):
        """
        Get reasoning from the reasoning log.
        
        Parameters:
        - reasoning_id: Filter by reasoning ID
        - symbol: Filter by symbol
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        - decision: Filter by decision
        
        Returns:
        - List of reasoning entries or single entry if reasoning_id is provided
        """
        if self.log_format == "json":
            return self._get_reasoning_json(reasoning_id, symbol, start_date, end_date, decision)
        else:  # TXT format
            return self._get_reasoning_txt(reasoning_id, symbol, start_date, end_date, decision)
    
    def _get_reasoning_json(self, reasoning_id, symbol, start_date, end_date, decision):
        """Get reasoning from JSON file"""
        try:
            with open(self.log_file, "r") as f:
                reasoning_entries = json.load(f)
            
            if reasoning_id:
                for entry in reasoning_entries:
                    if entry.get("reasoning_id") == reasoning_id:
                        return entry
                return None
            
            filtered_entries = reasoning_entries
            
            if symbol:
                filtered_entries = [e for e in filtered_entries if e.get("symbol") == symbol]
            
            if start_date:
                start_dt = datetime.datetime.fromisoformat(start_date)
                filtered_entries = [e for e in filtered_entries if "timestamp" in e and datetime.datetime.fromisoformat(e["timestamp"]) >= start_dt]
            
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date)
                filtered_entries = [e for e in filtered_entries if "timestamp" in e and datetime.datetime.fromisoformat(e["timestamp"]) <= end_dt]
            
            if decision:
                filtered_entries = [e for e in filtered_entries if e.get("decision") == decision]
            
            return filtered_entries
        except Exception as e:
            self.logger.error(f"Error getting reasoning from JSON: {e}")
            return []
    
    def _get_reasoning_txt(self, reasoning_id, symbol, start_date, end_date, decision):
        """Get reasoning from TXT file"""
        self.logger.warning("Filtering reasoning from TXT format is not fully supported. Consider using JSON format.")
        
        try:
            with open(self.log_file, "r") as f:
                content = f.read()
            
            entries = content.split("---\n\n")
            
            parsed_entries = []
            for entry in entries:
                if not entry.strip():
                    continue
                
                parsed_entry = {}
                
                lines = entry.strip().split("\n")
                for line in lines:
                    if line.startswith("## Reasoning ID:"):
                        parsed_entry["reasoning_id"] = line.replace("## Reasoning ID:", "").strip()
                    elif line.startswith("## Timestamp:"):
                        parsed_entry["timestamp"] = line.replace("## Timestamp:", "").strip()
                    elif line.startswith("## Symbol:"):
                        parsed_entry["symbol"] = line.replace("## Symbol:", "").strip()
                    elif line.startswith("## Decision:"):
                        parsed_entry["decision"] = line.replace("## Decision:", "").strip()
                
                parsed_entry["raw_content"] = entry.strip()
                
                parsed_entries.append(parsed_entry)
            
            if reasoning_id:
                for entry in parsed_entries:
                    if entry.get("reasoning_id") == reasoning_id:
                        return entry
                return None
            
            filtered_entries = parsed_entries
            
            if symbol:
                filtered_entries = [e for e in filtered_entries if e.get("symbol") == symbol]
            
            if start_date:
                start_dt = datetime.datetime.fromisoformat(start_date)
                filtered_entries = [e for e in filtered_entries if "timestamp" in e and datetime.datetime.fromisoformat(e["timestamp"]) >= start_dt]
            
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date)
                filtered_entries = [e for e in filtered_entries if "timestamp" in e and datetime.datetime.fromisoformat(e["timestamp"]) <= end_dt]
            
            if decision:
                filtered_entries = [e for e in filtered_entries if e.get("decision") == decision]
            
            return filtered_entries
        except Exception as e:
            self.logger.error(f"Error getting reasoning from TXT: {e}")
            return []
    
    def analyze_reasoning_patterns(self, symbol=None, start_date=None, end_date=None):
        """
        Analyze reasoning patterns.
        
        Parameters:
        - symbol: Filter by symbol
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        
        Returns:
        - Dictionary with reasoning analysis
        """
        reasoning_entries = self.get_reasoning(None, symbol, start_date, end_date, None)
        
        if not reasoning_entries:
            return {
                "total_entries": 0,
                "symbols": {},
                "decisions": {},
                "avg_confidence": 0.0,
                "gate_score_averages": {},
                "common_market_conditions": {},
                "reasoning_patterns": []
            }
        
        analysis = {
            "total_entries": len(reasoning_entries),
            "symbols": {},
            "decisions": {},
            "avg_confidence": 0.0,
            "gate_score_averages": {},
            "common_market_conditions": {},
            "reasoning_patterns": []
        }
        
        total_confidence = 0.0
        gate_scores_sum = {}
        gate_scores_count = {}
        market_conditions_count = {}
        reasoning_texts = []
        
        for entry in reasoning_entries:
            symbol = entry.get("symbol", "Unknown")
            if symbol not in analysis["symbols"]:
                analysis["symbols"][symbol] = 0
            analysis["symbols"][symbol] += 1
            
            decision = entry.get("decision", "Unknown")
            if decision not in analysis["decisions"]:
                analysis["decisions"][decision] = 0
            analysis["decisions"][decision] += 1
            
            confidence = entry.get("confidence", 0.0)
            total_confidence += confidence
            
            gate_scores = entry.get("gate_scores", {})
            for gate, score in gate_scores.items():
                if gate not in gate_scores_sum:
                    gate_scores_sum[gate] = 0.0
                    gate_scores_count[gate] = 0
                gate_scores_sum[gate] += score
                gate_scores_count[gate] += 1
            
            market_conditions = entry.get("market_conditions", {})
            for condition, value in market_conditions.items():
                condition_key = f"{condition}:{value}"
                if condition_key not in market_conditions_count:
                    market_conditions_count[condition_key] = 0
                market_conditions_count[condition_key] += 1
            
            final_reasoning = entry.get("final_reasoning", "")
            if final_reasoning:
                reasoning_texts.append(final_reasoning)
        
        analysis["avg_confidence"] = total_confidence / len(reasoning_entries) if reasoning_entries else 0.0
        
        for gate, total in gate_scores_sum.items():
            analysis["gate_score_averages"][gate] = total / gate_scores_count[gate] if gate_scores_count[gate] > 0 else 0.0
        
        sorted_conditions = sorted(market_conditions_count.items(), key=lambda x: x[1], reverse=True)
        for condition, count in sorted_conditions[:10]:  # Top 10
            analysis["common_market_conditions"][condition] = count
        
        try:
            from collections import Counter
            import re
            
            all_text = " ".join(reasoning_texts)
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_counts = Counter(words)
            
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were", "be", "been", "being"}
            filtered_counts = {word: count for word, count in word_counts.items() if word not in common_words and len(word) > 3}
            
            top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            top_word_list = [word for word, _ in top_words]
            phrases = []
            
            for text in reasoning_texts:
                for word in top_word_list:
                    pattern = r'[^.!?]*\b' + word + r'\b[^.!?]*[.!?]'
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    phrases.extend(matches)
            
            phrase_counts = Counter(phrases)
            top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            analysis["reasoning_patterns"] = [{"phrase": phrase, "count": count} for phrase, count in top_phrases]
            analysis["top_words"] = [{"word": word, "count": count} for word, count in top_words]
        except Exception as e:
            self.logger.error(f"Error analyzing reasoning patterns: {e}")
            analysis["reasoning_patterns"] = []
            analysis["top_words"] = []
        
        return analysis
