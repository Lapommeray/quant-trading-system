"""
SHAP Interpreter Module

This module provides SHAP (SHapley Additive exPlanations) value interpretation
for machine learning models in the Quantum Trading System. It enables real-time
SHAP value streaming to the dashboard for model explainability.

Dependencies:
- shap
- numpy
- pandas
- matplotlib
- plotly (for dashboard integration)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime
import json
import threading
import time
import queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('shap_interpreter.log')
    ]
)

logger = logging.getLogger("SHAPInterpreter")

try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library loaded successfully")
except ImportError:
    logger.warning("SHAP library not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Some visualization features will be disabled.")
    MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly loaded successfully for dashboard integration")
except ImportError:
    logger.warning("Plotly not available. Dashboard integration will be disabled.")
    PLOTLY_AVAILABLE = False

class SHAPInterpreter:
    """
    SHAP value interpreter for model explainability.
    Provides real-time SHAP value streaming to the dashboard.
    """
    
    def __init__(
        self,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = "tree",
        max_display: int = 10,
        stream_interval: float = 1.0
    ):
        """
        Initialize the SHAP interpreter.
        
        Parameters:
        - background_data: Background data for SHAP explainer
        - feature_names: Names of features for better visualization
        - model_type: Type of model ('tree', 'deep', 'kernel', 'linear')
        - max_display: Maximum number of features to display
        - stream_interval: Interval (in seconds) for streaming SHAP values
        """
        self.background_data = background_data
        self.feature_names = feature_names
        self.model_type = model_type
        self.max_display = max_display
        self.stream_interval = stream_interval
        
        self.explainer = None
        self.shap_values = None
        self.model = None
        
        self.streaming = False
        self.stream_thread = None
        self.stream_queue = queue.Queue()
        self.stream_callback = None
        
        if not SHAP_AVAILABLE:
            logger.error("SHAP library is required for this module")
            
    def set_model(self, model: Any) -> bool:
        """
        Set the model to explain.
        
        Parameters:
        - model: The machine learning model to explain
        
        Returns:
        - Success status
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP library not available")
            return False
            
        try:
            self.model = model
            self._initialize_explainer()
            return True
        except Exception as e:
            logger.error(f"Error setting model: {str(e)}")
            return False
            
    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type"""
        if not SHAP_AVAILABLE or self.model is None:
            return
            
        try:
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model, self.background_data)
            elif self.model_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
            elif self.model_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            else:
                logger.warning(f"Unknown model type: {self.model_type}. Using KernelExplainer.")
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                
            logger.info(f"SHAP {self.model_type} explainer initialized")
        except Exception as e:
            logger.error(f"Error initializing explainer: {str(e)}")
            self.explainer = None
            
    def explain(self, data: np.ndarray) -> np.ndarray:
        """
        Generate SHAP values for the given data.
        
        Parameters:
        - data: Input data to explain
        
        Returns:
        - SHAP values
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return np.array([])
            
        try:
            self.shap_values = self.explainer.shap_values(data)
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return np.array([])
            
    def plot_summary(self, output_file: Optional[str] = None) -> None:
        """
        Plot SHAP summary plot.
        
        Parameters:
        - output_file: Path to save the plot (optional)
        """
        if not SHAP_AVAILABLE or not MPL_AVAILABLE or self.shap_values is None:
            logger.error("Cannot create summary plot: missing dependencies or SHAP values")
            return
            
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values, 
                feature_names=self.feature_names,
                max_display=self.max_display
            )
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                logger.info(f"Summary plot saved to {output_file}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Error creating summary plot: {str(e)}")
            
    def plot_waterfall(self, instance_index: int = 0, output_file: Optional[str] = None) -> None:
        """
        Plot SHAP waterfall plot for a specific instance.
        
        Parameters:
        - instance_index: Index of the instance to explain
        - output_file: Path to save the plot (optional)
        """
        if not SHAP_AVAILABLE or not MPL_AVAILABLE or self.shap_values is None:
            logger.error("Cannot create waterfall plot: missing dependencies or SHAP values")
            return
            
        try:
            plt.figure(figsize=(10, 8))
            
            if isinstance(self.shap_values, list):
                shap.plots.waterfall(
                    shap.Explanation(
                        values=self.shap_values[0][instance_index],
                        base_values=self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                        data=self.background_data[instance_index] if self.background_data is not None else None,
                        feature_names=self.feature_names
                    )
                )
            else:
                shap.plots.waterfall(
                    shap.Explanation(
                        values=self.shap_values[instance_index],
                        base_values=self.explainer.expected_value,
                        data=self.background_data[instance_index] if self.background_data is not None else None,
                        feature_names=self.feature_names
                    )
                )
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to {output_file}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            
    def get_dashboard_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP data for dashboard integration.
        
        Parameters:
        - data: Input data to explain
        
        Returns:
        - Dictionary with SHAP data for dashboard
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return {"error": "SHAP explainer not initialized"}
            
        try:
            shap_values = self.explain(data)
            
            if isinstance(shap_values, list):
                importance_values = np.abs(shap_values[0]).mean(axis=0)
            else:
                importance_values = np.abs(shap_values).mean(axis=0)
                
            if self.feature_names:
                feature_importance = sorted(
                    zip(self.feature_names, importance_values),
                    key=lambda x: x[1],
                    reverse=True
                )
                features = [f[0] for f in feature_importance[:self.max_display]]
                importance = [f[1] for f in feature_importance[:self.max_display]]
            else:
                indices = np.argsort(importance_values)[::-1][:self.max_display]
                features = [f"Feature {i}" for i in indices]
                importance = importance_values[indices]
                
            explanations = []
            for i in range(min(5, len(data))):  # Limit to 5 examples
                if isinstance(shap_values, list):
                    values = shap_values[0][i]
                    base_value = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
                else:
                    values = shap_values[i]
                    base_value = self.explainer.expected_value
                    
                if self.feature_names:
                    sorted_indices = np.argsort(np.abs(values))[::-1][:self.max_display]
                    feature_names = [self.feature_names[j] for j in sorted_indices]
                    feature_values = values[sorted_indices]
                else:
                    sorted_indices = np.argsort(np.abs(values))[::-1][:self.max_display]
                    feature_names = [f"Feature {j}" for j in sorted_indices]
                    feature_values = values[sorted_indices]
                    
                explanations.append({
                    "instance": i,
                    "base_value": float(base_value),
                    "features": feature_names,
                    "values": feature_values.tolist(),
                    "prediction": float(base_value + np.sum(values))
                })
                
            return {
                "feature_importance": {
                    "features": features,
                    "importance": importance.tolist() if isinstance(importance, np.ndarray) else importance
                },
                "explanations": explanations,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
            return {"error": str(e)}
            
    def create_plotly_summary(self, data: np.ndarray) -> Optional[go.Figure]:
        """
        Create a Plotly figure for SHAP summary visualization.
        
        Parameters:
        - data: Input data to explain
        
        Returns:
        - Plotly figure object or None if error
        """
        if not PLOTLY_AVAILABLE or not SHAP_AVAILABLE:
            logger.error("Plotly or SHAP not available")
            return None
            
        try:
            shap_values = self.explain(data)
            
            if isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values
                
            mean_abs_shap = np.abs(values).mean(axis=0)
            
            if self.feature_names:
                feature_importance = sorted(
                    zip(self.feature_names, mean_abs_shap),
                    key=lambda x: x[1]
                )
                features = [f[0] for f in feature_importance[-self.max_display:]]
                importance = [f[1] for f in feature_importance[-self.max_display:]]
            else:
                indices = np.argsort(mean_abs_shap)[-self.max_display:]
                features = [f"Feature {i}" for i in indices]
                importance = mean_abs_shap[indices]
                
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=features,
                    x=importance,
                    orientation='h',
                    marker=dict(
                        color='rgba(50, 171, 96, 0.7)',
                        line=dict(color='rgba(50, 171, 96, 1.0)', width=2)
                    )
                )
            )
            
            fig.update_layout(
                title='Feature Importance (mean |SHAP value|)',
                xaxis_title='mean |SHAP value|',
                yaxis_title='Feature',
                height=500,
                width=700,
                margin=dict(l=100, r=20, t=70, b=70),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating Plotly summary: {str(e)}")
            return None
            
    def create_plotly_waterfall(self, data: np.ndarray, instance_index: int = 0) -> Optional[go.Figure]:
        """
        Create a Plotly figure for SHAP waterfall visualization.
        
        Parameters:
        - data: Input data to explain
        - instance_index: Index of the instance to explain
        
        Returns:
        - Plotly figure object or None if error
        """
        if not PLOTLY_AVAILABLE or not SHAP_AVAILABLE:
            logger.error("Plotly or SHAP not available")
            return None
            
        try:
            shap_values = self.explain(data)
            
            if isinstance(shap_values, list):
                values = shap_values[0][instance_index]
                base_value = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            else:
                values = shap_values[instance_index]
                base_value = self.explainer.expected_value
                
            sorted_indices = np.argsort(np.abs(values))[::-1][:self.max_display]
            
            if self.feature_names:
                features = [self.feature_names[i] for i in sorted_indices]
            else:
                features = [f"Feature {i}" for i in sorted_indices]
                
            sorted_values = values[sorted_indices]
            
            cumulative = [base_value]
            for val in sorted_values:
                cumulative.append(cumulative[-1] + val)
                
            colors = ['rgba(50, 171, 96, 0.7)' if val > 0 else 'rgba(219, 64, 82, 0.7)' for val in sorted_values]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    name='Base Value',
                    y=['Base Value'],
                    x=[base_value],
                    orientation='h',
                    marker=dict(color='rgba(0, 0, 255, 0.7)')
                )
            )
            
            fig.add_trace(
                go.Bar(
                    name='Feature Impact',
                    y=features,
                    x=sorted_values,
                    orientation='h',
                    marker=dict(color=colors)
                )
            )
            
            fig.add_trace(
                go.Bar(
                    name='Final Prediction',
                    y=['Prediction'],
                    x=[cumulative[-1]],
                    orientation='h',
                    marker=dict(color='rgba(128, 0, 128, 0.7)')
                )
            )
            
            fig.update_layout(
                title=f'SHAP Waterfall Plot for Instance {instance_index}',
                xaxis_title='SHAP Value',
                height=600,
                width=800,
                margin=dict(l=100, r=20, t=70, b=70),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                barmode='relative'
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating Plotly waterfall: {str(e)}")
            return None
            
    def start_streaming(self, callback: Callable[[Dict[str, Any]], None], data_provider: Callable[[], np.ndarray]) -> bool:
        """
        Start streaming SHAP values to the dashboard.
        
        Parameters:
        - callback: Function to call with SHAP data
        - data_provider: Function that returns current data to explain
        
        Returns:
        - Success status
        """
        if self.streaming:
            logger.warning("SHAP streaming already active")
            return False
            
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return False
            
        try:
            self.streaming = True
            self.stream_callback = callback
            
            self.stream_thread = threading.Thread(
                target=self._stream_loop,
                args=(data_provider,),
                daemon=True
            )
            self.stream_thread.start()
            
            logger.info("SHAP streaming started")
            return True
        except Exception as e:
            logger.error(f"Error starting SHAP streaming: {str(e)}")
            self.streaming = False
            return False
            
    def stop_streaming(self) -> bool:
        """
        Stop streaming SHAP values.
        
        Returns:
        - Success status
        """
        if not self.streaming:
            logger.warning("SHAP streaming not active")
            return False
            
        try:
            self.streaming = False
            
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5)
                
            logger.info("SHAP streaming stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping SHAP streaming: {str(e)}")
            return False
            
    def _stream_loop(self, data_provider: Callable[[], np.ndarray]) -> None:
        """
        Background thread for streaming SHAP values.
        
        Parameters:
        - data_provider: Function that returns current data to explain
        """
        while self.streaming:
            try:
                data = data_provider()
                
                if data is None or len(data) == 0:
                    time.sleep(self.stream_interval)
                    continue
                    
                dashboard_data = self.get_dashboard_data(data)
                
                if self.stream_callback and dashboard_data:
                    self.stream_callback(dashboard_data)
                    
                time.sleep(self.stream_interval)
            except Exception as e:
                logger.error(f"Error in SHAP streaming loop: {str(e)}")
                time.sleep(self.stream_interval)
                
    def save_explanation(self, data: np.ndarray, filepath: str) -> bool:
        """
        Save SHAP explanation to a file.
        
        Parameters:
        - data: Input data to explain
        - filepath: Path to save the explanation
        
        Returns:
        - Success status
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return False
            
        try:
            shap_values = self.explain(data)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            explanation_data = {
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "base_value": float(self.explainer.expected_value) if not isinstance(self.explainer.expected_value, list) else [float(v) for v in self.explainer.expected_value],
                "shap_values": shap_values.tolist() if not isinstance(shap_values, list) else [sv.tolist() for sv in shap_values],
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(explanation_data, f)
                
            logger.info(f"SHAP explanation saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving SHAP explanation: {str(e)}")
            return False
            
    @classmethod
    def load_explanation(cls, filepath: str) -> Dict[str, Any]:
        """
        Load SHAP explanation from a file.
        
        Parameters:
        - filepath: Path to the saved explanation
        
        Returns:
        - Dictionary with SHAP explanation data
        """
        try:
            with open(filepath, 'r') as f:
                explanation_data = json.load(f)
                
            logger.info(f"SHAP explanation loaded from {filepath}")
            return explanation_data
        except Exception as e:
            logger.error(f"Error loading SHAP explanation: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    try:
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        interpreter = SHAPInterpreter(
            background_data=X[:100],
            feature_names=feature_names,
            model_type="tree"
        )
        
        interpreter.set_model(model)
        shap_values = interpreter.explain(X[:10])
        
        dashboard_data = interpreter.get_dashboard_data(X[:10])
        print(f"Dashboard data generated with {len(dashboard_data['explanations'])} explanations")
        
        if MPL_AVAILABLE:
            interpreter.plot_summary(output_file="shap_summary.png")
            interpreter.plot_waterfall(output_file="shap_waterfall.png")
            print("Plots saved to shap_summary.png and shap_waterfall.png")
            
        if PLOTLY_AVAILABLE:
            summary_fig = interpreter.create_plotly_summary(X[:10])
            waterfall_fig = interpreter.create_plotly_waterfall(X[:10])
            print("Plotly figures created successfully")
            
    except ImportError:
        print("Example requires scikit-learn. Install with: pip install scikit-learn")
