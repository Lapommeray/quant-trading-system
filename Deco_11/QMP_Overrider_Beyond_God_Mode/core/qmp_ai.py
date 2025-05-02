
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class QMPAIAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances = {}
        self.min_samples_for_training = 15
        self.last_prediction_confidence = 0.0
        
    def train(self, df):
        """
        Expects a DataFrame with columns for all gate scores and 'result'
        Where 'result' is 1 for profitable trade, 0 for loss.
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
            
            self.model.fit(X_train_scaled, y_train)
            
            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.model.score(X_test_scaled, y_test)
            print(f"Model accuracy: {accuracy:.2f}")
            
            self.feature_importances = dict(zip(feature_cols, self.model.feature_importances_))
        else:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
        
        self.is_trained = True
        return True
    
    def predict_gate_pass(self, current_data):
        """
        Predicts if the current signal gates should be trusted based on training.
        Input: dict with keys matching training features
        Output: True/False recommendation and confidence score
        """
        if not self.is_trained:
            return True  # Default to trusting gates
        
        X_new = pd.DataFrame([current_data])
        feature_cols = X_new.columns
        X_new_scaled = self.scaler.transform(X_new)
        
        pred_proba = self.model.predict_proba(X_new_scaled)
        confidence = max(pred_proba[0])
        self.last_prediction_confidence = confidence
        
        pred = self.model.predict(X_new_scaled)
        return bool(pred[0])
