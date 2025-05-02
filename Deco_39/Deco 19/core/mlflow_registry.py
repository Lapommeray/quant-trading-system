"""
MLFlow Model Registry

This module implements the MLFlow Model Registry for the QMP Overrider system.
It provides a unified interface for model versioning, tracking, and deployment.
"""

import os
import json
import logging
import datetime
from pathlib import Path
import numpy as np

class MLFlowModelRegistry:
    """
    MLFlow Model Registry for the QMP Overrider system.
    
    This class provides a unified interface for model versioning, tracking, and deployment
    using MLFlow as the backend. It handles model registration, versioning, and deployment
    for all AI components in the system.
    """
    
    def __init__(self, tracking_uri=None, experiment_name="qmp_overrider"):
        """
        Initialize the MLFlow Model Registry.
        
        Parameters:
        - tracking_uri: URI for the MLFlow tracking server
        - experiment_name: Name of the experiment to use
        """
        self.logger = logging.getLogger("MLFlowModelRegistry")
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            self.mlflow = mlflow
            self.client = MlflowClient()
            
            self.experiment_name = experiment_name
            self.experiment = self._get_or_create_experiment()
            
            self.logger.info(f"MLFlow Model Registry initialized with experiment '{experiment_name}'")
            self.mlflow_available = True
        except ImportError:
            self.logger.warning("MLFlow not installed. Using fallback local model registry.")
            self.mlflow_available = False
            self._setup_local_registry()
    
    def _get_or_create_experiment(self):
        """Get or create the experiment"""
        experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            return experiment
        
        experiment_id = self.mlflow.create_experiment(self.experiment_name)
        return self.mlflow.get_experiment(experiment_id)
    
    def _setup_local_registry(self):
        """Set up local model registry as fallback"""
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.models_dir / "registry.json"
        
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "runs": []
            }
            self._save_registry()
    
    def _save_registry(self):
        """Save the local registry to disk"""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def log_model(self, model, model_name, model_type, metadata=None, framework="sklearn"):
        """
        Log a model to the registry.
        
        Parameters:
        - model: The model object to log
        - model_name: Name of the model
        - model_type: Type of the model (e.g., "classifier", "regressor")
        - metadata: Additional metadata for the model
        - framework: Framework used for the model (e.g., "sklearn", "pytorch")
        
        Returns:
        - Model info dictionary
        """
        if metadata is None:
            metadata = {}
        
        timestamp = datetime.datetime.now().isoformat()
        
        if self.mlflow_available:
            return self._log_model_mlflow(model, model_name, model_type, metadata, framework, timestamp)
        else:
            return self._log_model_local(model, model_name, model_type, metadata, framework, timestamp)
    
    def _log_model_mlflow(self, model, model_name, model_type, metadata, framework, timestamp):
        """Log a model using MLFlow"""
        with self.mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            self.mlflow.log_params({
                "model_type": model_type,
                "framework": framework,
                "timestamp": timestamp,
                **metadata
            })
            
            if framework == "sklearn":
                self.mlflow.sklearn.log_model(model, model_name)
            elif framework == "pytorch":
                self.mlflow.pytorch.log_model(model, model_name)
            elif framework == "tensorflow":
                self.mlflow.tensorflow.log_model(model, model_name)
            else:
                self.mlflow.pyfunc.log_model(model, model_name)
            
            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            registered_model = self.mlflow.register_model(model_uri, model_name)
            
            return {
                "model_name": model_name,
                "model_type": model_type,
                "framework": framework,
                "timestamp": timestamp,
                "run_id": run.info.run_id,
                "model_version": registered_model.version,
                "metadata": metadata
            }
    
    def _log_model_local(self, model, model_name, model_type, metadata, framework, timestamp):
        """Log a model locally"""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        if model_name in self.registry["models"]:
            version = self.registry["models"][model_name]["latest_version"] + 1
        else:
            version = 1
            self.registry["models"][model_name] = {
                "versions": [],
                "latest_version": 0
            }
        
        model_path = model_dir / f"v{version}"
        model_path.mkdir(exist_ok=True)
        
        try:
            if framework == "sklearn":
                import joblib
                joblib.dump(model, model_path / "model.pkl")
            elif framework == "pytorch":
                import torch
                torch.save(model, model_path / "model.pt")
            elif framework == "tensorflow":
                model.save(model_path / "model")
            else:
                import pickle
                with open(model_path / "model.pkl", "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None
        
        model_info = {
            "model_name": model_name,
            "model_type": model_type,
            "framework": framework,
            "timestamp": timestamp,
            "version": version,
            "metadata": metadata
        }
        
        with open(model_path / "metadata.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        self.registry["models"][model_name]["latest_version"] = version
        self.registry["models"][model_name]["versions"].append(model_info)
        
        run_id = f"local-{timestamp}"
        run_info = {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": version,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        self.registry["runs"].append(run_info)
        self._save_registry()
        
        return model_info
    
    def load_model(self, model_name, version="latest", framework="sklearn"):
        """
        Load a model from the registry.
        
        Parameters:
        - model_name: Name of the model to load
        - version: Version of the model to load (or "latest")
        - framework: Framework used for the model
        
        Returns:
        - The loaded model
        """
        if self.mlflow_available:
            return self._load_model_mlflow(model_name, version, framework)
        else:
            return self._load_model_local(model_name, version, framework)
    
    def _load_model_mlflow(self, model_name, version, framework):
        """Load a model using MLFlow"""
        if version == "latest":
            version = "latest"
        else:
            version = str(version)
        
        model_uri = f"models:/{model_name}/{version}"
        
        try:
            if framework == "sklearn":
                return self.mlflow.sklearn.load_model(model_uri)
            elif framework == "pytorch":
                return self.mlflow.pytorch.load_model(model_uri)
            elif framework == "tensorflow":
                return self.mlflow.tensorflow.load_model(model_uri)
            else:
                return self.mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def _load_model_local(self, model_name, version, framework):
        """Load a model locally"""
        if model_name not in self.registry["models"]:
            self.logger.error(f"Model '{model_name}' not found in registry")
            return None
        
        if version == "latest":
            version = self.registry["models"][model_name]["latest_version"]
        
        model_path = self.models_dir / model_name / f"v{version}"
        
        if not model_path.exists():
            self.logger.error(f"Model '{model_name}' version {version} not found")
            return None
        
        try:
            if framework == "sklearn":
                import joblib
                return joblib.load(model_path / "model.pkl")
            elif framework == "pytorch":
                import torch
                return torch.load(model_path / "model.pt")
            elif framework == "tensorflow":
                import tensorflow as tf
                return tf.keras.models.load_model(model_path / "model")
            else:
                import pickle
                with open(model_path / "model.pkl", "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def get_model_info(self, model_name, version="latest"):
        """
        Get information about a model.
        
        Parameters:
        - model_name: Name of the model
        - version: Version of the model (or "latest")
        
        Returns:
        - Model info dictionary
        """
        if self.mlflow_available:
            return self._get_model_info_mlflow(model_name, version)
        else:
            return self._get_model_info_local(model_name, version)
    
    def _get_model_info_mlflow(self, model_name, version):
        """Get model info using MLFlow"""
        if version == "latest":
            version = None
        
        try:
            model_version = self.client.get_latest_versions(model_name, stages=["None"])[0] if version is None else \
                           self.client.get_model_version(model_name, version)
            
            run = self.client.get_run(model_version.run_id)
            
            return {
                "model_name": model_name,
                "model_version": model_version.version,
                "timestamp": model_version.creation_timestamp,
                "run_id": model_version.run_id,
                "metadata": run.data.params
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def _get_model_info_local(self, model_name, version):
        """Get model info locally"""
        if model_name not in self.registry["models"]:
            self.logger.error(f"Model '{model_name}' not found in registry")
            return None
        
        if version == "latest":
            version = self.registry["models"][model_name]["latest_version"]
        
        for model_info in self.registry["models"][model_name]["versions"]:
            if model_info["version"] == version:
                return model_info
        
        self.logger.error(f"Model '{model_name}' version {version} not found")
        return None
    
    def list_models(self):
        """
        List all models in the registry.
        
        Returns:
        - List of model names
        """
        if self.mlflow_available:
            return self._list_models_mlflow()
        else:
            return self._list_models_local()
    
    def _list_models_mlflow(self):
        """List models using MLFlow"""
        try:
            return [model.name for model in self.client.list_registered_models()]
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def _list_models_local(self):
        """List models locally"""
        return list(self.registry["models"].keys())
    
    def list_model_versions(self, model_name):
        """
        List all versions of a model.
        
        Parameters:
        - model_name: Name of the model
        
        Returns:
        - List of version numbers
        """
        if self.mlflow_available:
            return self._list_model_versions_mlflow(model_name)
        else:
            return self._list_model_versions_local(model_name)
    
    def _list_model_versions_mlflow(self, model_name):
        """List model versions using MLFlow"""
        try:
            return [int(version.version) for version in self.client.get_latest_versions(model_name)]
        except Exception as e:
            self.logger.error(f"Error listing model versions: {e}")
            return []
    
    def _list_model_versions_local(self, model_name):
        """List model versions locally"""
        if model_name not in self.registry["models"]:
            self.logger.error(f"Model '{model_name}' not found in registry")
            return []
        
        return [model_info["version"] for model_info in self.registry["models"][model_name]["versions"]]
    
    def log_metrics(self, metrics, run_id=None):
        """
        Log metrics for a model.
        
        Parameters:
        - metrics: Dictionary of metrics to log
        - run_id: Run ID to log metrics for (or None for current run)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.mlflow_available:
            return self._log_metrics_mlflow(metrics, run_id)
        else:
            return self._log_metrics_local(metrics, run_id)
    
    def _log_metrics_mlflow(self, metrics, run_id):
        """Log metrics using MLFlow"""
        try:
            if run_id:
                with self.mlflow.start_run(run_id=run_id):
                    self.mlflow.log_metrics(metrics)
            else:
                self.mlflow.log_metrics(metrics)
            return True
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            return False
    
    def _log_metrics_local(self, metrics, run_id):
        """Log metrics locally"""
        if run_id is None:
            if not self.registry["runs"]:
                self.logger.error("No runs available")
                return False
            run_id = self.registry["runs"][-1]["run_id"]
        
        for run in self.registry["runs"]:
            if run["run_id"] == run_id:
                if "metrics" not in run:
                    run["metrics"] = {}
                
                run["metrics"].update(metrics)
                self._save_registry()
                return True
        
        self.logger.error(f"Run '{run_id}' not found")
        return False
    
    def compare_models(self, model_name, versions=None, metric="accuracy"):
        """
        Compare different versions of a model.
        
        Parameters:
        - model_name: Name of the model
        - versions: List of versions to compare (or None for all)
        - metric: Metric to compare
        
        Returns:
        - Dictionary mapping versions to metric values
        """
        if versions is None:
            versions = self.list_model_versions(model_name)
        
        results = {}
        
        for version in versions:
            model_info = self.get_model_info(model_name, version)
            if model_info:
                run_id = model_info.get("run_id")
                if run_id:
                    if self.mlflow_available:
                        try:
                            run = self.client.get_run(run_id)
                            if metric in run.data.metrics:
                                results[version] = run.data.metrics[metric]
                        except Exception as e:
                            self.logger.error(f"Error getting run metrics: {e}")
                    else:
                        for run in self.registry["runs"]:
                            if run["run_id"] == run_id and "metrics" in run and metric in run["metrics"]:
                                results[version] = run["metrics"][metric]
        
        return results
    
    def get_best_model(self, model_name, metric="accuracy", higher_is_better=True):
        """
        Get the best version of a model based on a metric.
        
        Parameters:
        - model_name: Name of the model
        - metric: Metric to compare
        - higher_is_better: Whether higher metric values are better
        
        Returns:
        - Best model version
        """
        metrics = self.compare_models(model_name, metric=metric)
        
        if not metrics:
            return None
        
        if higher_is_better:
            best_version = max(metrics.items(), key=lambda x: x[1])[0]
        else:
            best_version = min(metrics.items(), key=lambda x: x[1])[0]
        
        return best_version
    
    def delete_model(self, model_name, version=None):
        """
        Delete a model from the registry.
        
        Parameters:
        - model_name: Name of the model
        - version: Version to delete (or None for all)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.mlflow_available:
            return self._delete_model_mlflow(model_name, version)
        else:
            return self._delete_model_local(model_name, version)
    
    def _delete_model_mlflow(self, model_name, version):
        """Delete a model using MLFlow"""
        try:
            if version:
                self.client.delete_model_version(model_name, version)
            else:
                self.client.delete_registered_model(model_name)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    def _delete_model_local(self, model_name, version):
        """Delete a model locally"""
        if model_name not in self.registry["models"]:
            self.logger.error(f"Model '{model_name}' not found in registry")
            return False
        
        if version:
            model_path = self.models_dir / model_name / f"v{version}"
            if not model_path.exists():
                self.logger.error(f"Model '{model_name}' version {version} not found")
                return False
            
            try:
                import shutil
                shutil.rmtree(model_path)
                
                self.registry["models"][model_name]["versions"] = [
                    v for v in self.registry["models"][model_name]["versions"] if v["version"] != version
                ]
                
                if not self.registry["models"][model_name]["versions"]:
                    del self.registry["models"][model_name]
                else:
                    self.registry["models"][model_name]["latest_version"] = max(
                        v["version"] for v in self.registry["models"][model_name]["versions"]
                    )
                
                self._save_registry()
                return True
            except Exception as e:
                self.logger.error(f"Error deleting model: {e}")
                return False
        else:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                self.logger.error(f"Model '{model_name}' not found")
                return False
            
            try:
                import shutil
                shutil.rmtree(model_path)
                
                del self.registry["models"][model_name]
                self._save_registry()
                return True
            except Exception as e:
                self.logger.error(f"Error deleting model: {e}")
                return False
