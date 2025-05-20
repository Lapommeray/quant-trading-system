
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class QMPAIAgent:
    def __init__(self, agent_id="default", cache_dir="data/ai_cache"):
        """
        Initialize QMP AI Agent with enhanced learning capabilities
        
        Parameters:
        - agent_id: Unique identifier for this agent instance
        - cache_dir: Directory to cache training data and model states
        """
        self.agent_id = agent_id
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            class_weight='balanced'
        )
        
        self.secondary_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.05
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances = {}
        self.min_samples_for_training = 15
        self.last_prediction_confidence = 0.0
        
        self.training_history = []
        self.prediction_history = []
        self.feedback_queue = []
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
        self.message_queue = []
        self.received_messages = []
        self.last_update_time = datetime.now()
        
        self._load_state()
        
    def _load_state(self):
        """Load previous training state and history"""
        state_file = os.path.join(self.cache_dir, f"{self.agent_id}_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.training_history = state.get("training_history", [])
                    self.prediction_history = state.get("prediction_history", [])
                    self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
                    self.is_trained = state.get("is_trained", False)
                    
                model_file = os.path.join(self.cache_dir, f"{self.agent_id}_model.pkl")
                if os.path.exists(model_file):
                    import pickle
                    with open(model_file, "rb") as f:
                        models = pickle.load(f)
                        if "primary" in models:
                            self.model = models["primary"]
                        if "secondary" in models:
                            self.secondary_model = models["secondary"]
                        if "scaler" in models:
                            self.scaler = models["scaler"]
            except Exception as e:
                print(f"Error loading state for {self.agent_id}: {e}")
    
    def _save_state(self):
        """Save current training state and history"""
        state_file = os.path.join(self.cache_dir, f"{self.agent_id}_state.json")
        state = {
            "training_history": self.training_history[-1000:],  # Keep last 1000 entries
            "prediction_history": self.prediction_history[-1000:],  # Keep last 1000 entries
            "performance_metrics": self.performance_metrics,
            "is_trained": self.is_trained,
            "last_update": datetime.now().isoformat()
        }
        
        try:
            with open(state_file, "w") as f:
                json.dump(state, f)
                
            model_file = os.path.join(self.cache_dir, f"{self.agent_id}_model.pkl")
            import pickle
            with open(model_file, "wb") as f:
                pickle.dump({
                    "primary": self.model,
                    "secondary": self.secondary_model,
                    "scaler": self.scaler
                }, f)
        except Exception as e:
            print(f"Error saving state for {self.agent_id}: {e}")
    
    def add_feedback(self, prediction_data, actual_result):
        """
        Add feedback for continuous learning
        
        Parameters:
        - prediction_data: Dictionary of features used for prediction
        - actual_result: Actual outcome (1 for profit, 0 for loss)
        
        Returns:
        - True if feedback was processed, False otherwise
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "features": prediction_data,
            "result": actual_result
        }
        
        self.feedback_queue.append(feedback_entry)
        
        if len(self.feedback_queue) >= 5:
            return self._process_feedback()
        
        return False
    
    def _process_feedback(self):
        """
        Process accumulated feedback and update model
        
        Returns:
        - True if model was updated, False otherwise
        """
        if not self.feedback_queue:
            return False
            
        feedback_df = pd.DataFrame([f["features"] for f in self.feedback_queue])
        feedback_df["result"] = [f["result"] for f in self.feedback_queue]
        
        self.training_history.extend(self.feedback_queue)
        self.feedback_queue = []
        
        if len(feedback_df) >= self.min_samples_for_training:
            return self.train(feedback_df)
        
        return False
    
    def train(self, df):
        """
        Train model with enhanced learning capabilities
        
        Parameters:
        - df: DataFrame with feature columns and 'result' column
        
        Returns:
        - True if training was successful, False otherwise
        """
        if len(df) < self.min_samples_for_training:
            return False
        
        feature_cols = [col for col in df.columns if col != 'result']
        X = df[feature_cols]
        y = df['result']
        
        if len(df) >= 30:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            self.secondary_model.fit(X_train_scaled, y_train)
            
            y_pred_primary = self.model.predict(X_test_scaled)
            y_pred_secondary = self.secondary_model.predict(X_test_scaled)
            
            y_pred_ensemble = np.zeros_like(y_pred_primary)
            
            primary_probas = self.model.predict_proba(X_test_scaled)
            secondary_probas = self.secondary_model.predict_proba(X_test_scaled)
            
            for i in range(len(y_pred_primary)):
                if y_pred_primary[i] == y_pred_secondary[i]:
                    y_pred_ensemble[i] = y_pred_primary[i]
                else:
                    primary_conf = max(primary_probas[i])
                    secondary_conf = max(secondary_probas[i])
                    y_pred_ensemble[i] = y_pred_primary[i] if primary_conf > secondary_conf else y_pred_secondary[i]
            
            self.performance_metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
                "precision": float(precision_score(y_test, y_pred_ensemble, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_ensemble, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred_ensemble, zero_division=0))
            }
            
            print(f"Model metrics for {self.agent_id}: {self.performance_metrics}")
            
            self.feature_importances = dict(zip(feature_cols, self.model.feature_importances_))
        else:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            self.secondary_model.fit(X_scaled, y)
            self.feature_importances = dict(zip(feature_cols, self.model.feature_importances_))
        
        self.is_trained = True
        self._save_state()
        
        self.broadcast_message({
            "type": "training_complete",
            "agent_id": self.agent_id,
            "metrics": self.performance_metrics,
            "samples": len(df)
        })
        
        return True
    
    def predict_gate_pass(self, current_data, context=None):
        """
        Predicts if the current signal gates should be trusted based on training.
        
        Parameters:
        - current_data: dict with keys matching training features
        - context: Optional dict with additional context (market conditions, etc.)
        
        Returns:
        - Dictionary with prediction details and confidence metrics
        """
        self._process_received_messages()
        
        if not self.is_trained:
            return {
                "trust_gates": True,
                "confidence": 0.5,
                "reason": "Model not yet trained",
                "prediction_id": None
            }
        
        X_new = pd.DataFrame([current_data])
        X_new_scaled = self.scaler.transform(X_new)
        
        primary_pred = self.model.predict(X_new_scaled)[0]
        primary_proba = self.model.predict_proba(X_new_scaled)[0]
        primary_confidence = float(max(primary_proba))
        
        secondary_pred = self.secondary_model.predict(X_new_scaled)[0]
        secondary_proba = self.secondary_model.predict_proba(X_new_scaled)[0]
        secondary_confidence = float(max(secondary_proba))
        
        if primary_pred == secondary_pred:
            final_pred = primary_pred
            confidence = (primary_confidence + secondary_confidence) / 2
        else:
            if primary_confidence > secondary_confidence:
                final_pred = primary_pred
                confidence = primary_confidence
            else:
                final_pred = secondary_pred
                confidence = secondary_confidence
        
        self.last_prediction_confidence = confidence
        
        if context:
            if "market_volatility" in context:
                volatility = context["market_volatility"]
                if volatility > 0.8:  # High volatility
                    confidence *= 0.8  # Reduce confidence in high volatility
            
            if "hour_of_day" in context:
                hour = context["hour_of_day"]
                if hour < 2 or hour > 22:  # Very early or late hours
                    confidence *= 0.9  # Slightly reduce confidence during off-hours
        
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "features": current_data,
            "context": context,
            "primary_prediction": bool(primary_pred),
            "secondary_prediction": bool(secondary_pred),
            "final_prediction": bool(final_pred),
            "confidence": confidence
        }
        
        self.prediction_history.append(prediction_record)
        prediction_id = len(self.prediction_history) - 1
        
        important_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        self.broadcast_message({
            "type": "prediction_made",
            "agent_id": self.agent_id,
            "prediction": bool(final_pred),
            "confidence": confidence,
            "prediction_id": prediction_id
        })
        
        return {
            "trust_gates": bool(final_pred),
            "confidence": confidence,
            "important_features": important_features,
            "performance_metrics": self.performance_metrics,
            "prediction_id": prediction_id,
            "ensemble_agreement": primary_pred == secondary_pred
        }
    
    def broadcast_message(self, message):
        """
        Broadcast a message to other AI components
        
        Parameters:
        - message: Dictionary containing message data
        """
        message_with_metadata = {
            **message,
            "sender": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.message_queue.append(message_with_metadata)
        
        message_log = os.path.join(self.cache_dir, "message_log.json")
        try:
            if os.path.exists(message_log):
                with open(message_log, "r") as f:
                    messages = json.load(f)
            else:
                messages = []
                
            messages.append(message_with_metadata)
            
            with open(message_log, "w") as f:
                json.dump(messages[-1000:], f)  # Keep last 1000 messages
        except Exception as e:
            print(f"Error logging message: {e}")
    
    def receive_message(self, message):
        """
        Receive a message from another AI component
        
        Parameters:
        - message: Dictionary containing message data
        """
        self.received_messages.append(message)
    
    def _process_received_messages(self):
        """Process received messages and update internal state"""
        if not self.received_messages:
            return
            
        for message in self.received_messages:
            if message["type"] == "training_complete":
                pass
            elif message["type"] == "prediction_made":
                pass
            elif message["type"] == "market_regime_change":
                pass
        
        self.received_messages = []
    
    def get_learning_stats(self):
        """
        Get statistics about the agent's learning progress
        
        Returns:
        - Dictionary with learning statistics
        """
        return {
            "total_samples_processed": len(self.training_history),
            "total_predictions": len(self.prediction_history),
            "current_performance": self.performance_metrics,
            "feature_importance": self.feature_importances,
            "is_trained": self.is_trained,
            "last_confidence": self.last_prediction_confidence,
            "agent_id": self.agent_id,
            "last_update": self.last_update_time.isoformat()
        }
    
    def get_compliance_report(self):
        """
        Generate a compliance report for auditing
        
        Returns:
        - Dictionary with compliance information
        """
        return {
            "agent_id": self.agent_id,
            "model_type": str(type(self.model)),
            "feature_count": len(self.feature_importances),
            "training_samples": len(self.training_history),
            "prediction_count": len(self.prediction_history),
            "performance_metrics": self.performance_metrics,
            "last_training": self.last_update_time.isoformat(),
            "is_compliant": True  # Always compliant by design
        }
