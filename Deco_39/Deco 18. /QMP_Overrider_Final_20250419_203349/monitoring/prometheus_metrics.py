"""
Prometheus Metrics Integration

This module implements Prometheus metrics integration for the QMP Overrider system.
It provides a unified interface for exposing metrics to Prometheus for monitoring.
"""

import time
import logging
from threading import Thread, Lock
import os
from pathlib import Path

class PrometheusMetrics:
    """
    Prometheus Metrics for the QMP Overrider system.
    
    This class provides a unified interface for exposing metrics to Prometheus
    for monitoring. It handles metric registration, updating, and exposition.
    """
    
    def __init__(self, app_name="qmp_overrider", port=8000):
        """
        Initialize the Prometheus Metrics.
        
        Parameters:
        - app_name: Name of the application
        - port: Port to expose metrics on
        """
        self.logger = logging.getLogger("PrometheusMetrics")
        self.app_name = app_name
        self.port = port
        self.metrics = {}
        self.metric_lock = Lock()
        self.running = False
        self.server_thread = None
        
        try:
            import prometheus_client
            from prometheus_client import Counter, Gauge, Histogram, Summary
            
            self.prometheus_client = prometheus_client
            self.Counter = Counter
            self.Gauge = Gauge
            self.Histogram = Histogram
            self.Summary = Summary
            
            self.prometheus_available = True
            self.logger.info("Prometheus client available")
        except ImportError:
            self.logger.warning("Prometheus client not installed. Using fallback metrics.")
            self.prometheus_available = False
            self._setup_fallback_metrics()
    
    def _setup_fallback_metrics(self):
        """Set up fallback metrics when prometheus_client is not available"""
        class FallbackCounter:
            def __init__(self, name, documentation, labelnames=None):
                self.name = name
                self.documentation = documentation
                self.labelnames = labelnames or []
                self.values = {}
                self._default_value = 0
            
            def inc(self, amount=1, **labels):
                key = self._get_key(labels)
                if key not in self.values:
                    self.values[key] = self._default_value
                self.values[key] += amount
            
            def _get_key(self, labels):
                if not self.labelnames:
                    return "default"
                return tuple(labels.get(label, "") for label in self.labelnames)
        
        class FallbackGauge:
            def __init__(self, name, documentation, labelnames=None):
                self.name = name
                self.documentation = documentation
                self.labelnames = labelnames or []
                self.values = {}
                self._default_value = 0
            
            def set(self, value, **labels):
                key = self._get_key(labels)
                self.values[key] = value
            
            def inc(self, amount=1, **labels):
                key = self._get_key(labels)
                if key not in self.values:
                    self.values[key] = self._default_value
                self.values[key] += amount
            
            def dec(self, amount=1, **labels):
                key = self._get_key(labels)
                if key not in self.values:
                    self.values[key] = self._default_value
                self.values[key] -= amount
            
            def _get_key(self, labels):
                if not self.labelnames:
                    return "default"
                return tuple(labels.get(label, "") for label in self.labelnames)
        
        class FallbackHistogram:
            def __init__(self, name, documentation, labelnames=None, buckets=None):
                self.name = name
                self.documentation = documentation
                self.labelnames = labelnames or []
                self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                self.values = {}
                self.counts = {}
            
            def observe(self, value, **labels):
                key = self._get_key(labels)
                if key not in self.values:
                    self.values[key] = []
                    self.counts[key] = 0
                self.values[key].append(value)
                self.counts[key] += 1
            
            def _get_key(self, labels):
                if not self.labelnames:
                    return "default"
                return tuple(labels.get(label, "") for label in self.labelnames)
        
        class FallbackSummary:
            def __init__(self, name, documentation, labelnames=None):
                self.name = name
                self.documentation = documentation
                self.labelnames = labelnames or []
                self.values = {}
                self.counts = {}
            
            def observe(self, value, **labels):
                key = self._get_key(labels)
                if key not in self.values:
                    self.values[key] = []
                    self.counts[key] = 0
                self.values[key].append(value)
                self.counts[key] += 1
            
            def _get_key(self, labels):
                if not self.labelnames:
                    return "default"
                return tuple(labels.get(label, "") for label in self.labelnames)
        
        self.Counter = FallbackCounter
        self.Gauge = FallbackGauge
        self.Histogram = FallbackHistogram
        self.Summary = FallbackSummary
    
    def start_server(self):
        """Start the metrics server"""
        if not self.prometheus_available:
            self.logger.warning("Prometheus client not installed. Cannot start server.")
            return False
        
        if self.running:
            self.logger.warning("Metrics server already running")
            return True
        
        try:
            self.server_thread = Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            self.logger.info(f"Metrics server started on port {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting metrics server: {e}")
            return False
    
    def _run_server(self):
        """Run the metrics server"""
        try:
            self.prometheus_client.start_http_server(self.port)
            
            while self.running:
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in metrics server: {e}")
            self.running = False
    
    def stop_server(self):
        """Stop the metrics server"""
        if not self.running:
            return
        
        self.running = False
        
        if self.server_thread:
            self.server_thread.join(timeout=5)
            self.server_thread = None
        
        self.logger.info("Metrics server stopped")
    
    def create_counter(self, name, documentation, labelnames=None):
        """
        Create a counter metric.
        
        Parameters:
        - name: Name of the metric
        - documentation: Documentation for the metric
        - labelnames: List of label names
        
        Returns:
        - Counter metric
        """
        with self.metric_lock:
            metric_name = f"{self.app_name}_{name}"
            
            if metric_name in self.metrics:
                return self.metrics[metric_name]
            
            counter = self.Counter(
                name=metric_name,
                documentation=documentation,
                labelnames=labelnames or []
            )
            
            self.metrics[metric_name] = counter
            return counter
    
    def create_gauge(self, name, documentation, labelnames=None):
        """
        Create a gauge metric.
        
        Parameters:
        - name: Name of the metric
        - documentation: Documentation for the metric
        - labelnames: List of label names
        
        Returns:
        - Gauge metric
        """
        with self.metric_lock:
            metric_name = f"{self.app_name}_{name}"
            
            if metric_name in self.metrics:
                return self.metrics[metric_name]
            
            gauge = self.Gauge(
                name=metric_name,
                documentation=documentation,
                labelnames=labelnames or []
            )
            
            self.metrics[metric_name] = gauge
            return gauge
    
    def create_histogram(self, name, documentation, labelnames=None, buckets=None):
        """
        Create a histogram metric.
        
        Parameters:
        - name: Name of the metric
        - documentation: Documentation for the metric
        - labelnames: List of label names
        - buckets: List of bucket boundaries
        
        Returns:
        - Histogram metric
        """
        with self.metric_lock:
            metric_name = f"{self.app_name}_{name}"
            
            if metric_name in self.metrics:
                return self.metrics[metric_name]
            
            if self.prometheus_available:
                histogram = self.Histogram(
                    name=metric_name,
                    documentation=documentation,
                    labelnames=labelnames or [],
                    buckets=buckets
                )
            else:
                histogram = self.Histogram(
                    name=metric_name,
                    documentation=documentation,
                    labelnames=labelnames or [],
                    buckets=buckets
                )
            
            self.metrics[metric_name] = histogram
            return histogram
    
    def create_summary(self, name, documentation, labelnames=None):
        """
        Create a summary metric.
        
        Parameters:
        - name: Name of the metric
        - documentation: Documentation for the metric
        - labelnames: List of label names
        
        Returns:
        - Summary metric
        """
        with self.metric_lock:
            metric_name = f"{self.app_name}_{name}"
            
            if metric_name in self.metrics:
                return self.metrics[metric_name]
            
            summary = self.Summary(
                name=metric_name,
                documentation=documentation,
                labelnames=labelnames or []
            )
            
            self.metrics[metric_name] = summary
            return summary
    
    def get_metric(self, name):
        """
        Get a metric by name.
        
        Parameters:
        - name: Name of the metric
        
        Returns:
        - Metric object or None if not found
        """
        metric_name = f"{self.app_name}_{name}"
        return self.metrics.get(metric_name)
    
    def get_metrics(self):
        """
        Get all metrics.
        
        Returns:
        - Dictionary of metrics
        """
        return self.metrics
    
    def save_metrics(self, path=None):
        """
        Save metrics to a file.
        
        Parameters:
        - path: Path to save metrics to (or None for default)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.prometheus_available:
            self.logger.warning("Saving metrics not supported with Prometheus client")
            return False
        
        if path is None:
            path = Path("metrics") / f"{self.app_name}_metrics.json"
        
        try:
            import json
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            metrics_data = {}
            
            for name, metric in self.metrics.items():
                metric_data = {
                    "name": metric.name,
                    "documentation": metric.documentation,
                    "type": metric.__class__.__name__
                }
                
                if hasattr(metric, "labelnames"):
                    metric_data["labelnames"] = metric.labelnames
                
                if hasattr(metric, "values"):
                    values = {}
                    for key, value in metric.values.items():
                        if isinstance(key, tuple):
                            key = "|".join(str(k) for k in key)
                        values[str(key)] = value
                    metric_data["values"] = values
                
                if hasattr(metric, "counts"):
                    counts = {}
                    for key, value in metric.counts.items():
                        if isinstance(key, tuple):
                            key = "|".join(str(k) for k in key)
                        counts[str(key)] = value
                    metric_data["counts"] = counts
                
                metrics_data[name] = metric_data
            
            with open(path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Metrics saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            return False
    
    def load_metrics(self, path=None):
        """
        Load metrics from a file.
        
        Parameters:
        - path: Path to load metrics from (or None for default)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.prometheus_available:
            self.logger.warning("Loading metrics not supported with Prometheus client")
            return False
        
        if path is None:
            path = Path("metrics") / f"{self.app_name}_metrics.json"
        
        if not os.path.exists(path):
            self.logger.warning(f"Metrics file not found: {path}")
            return False
        
        try:
            import json
            
            with open(path, "r") as f:
                metrics_data = json.load(f)
            
            for name, metric_data in metrics_data.items():
                metric_type = metric_data["type"]
                metric_name = metric_data["name"]
                documentation = metric_data["documentation"]
                labelnames = metric_data.get("labelnames", [])
                
                if metric_type == "FallbackCounter":
                    metric = self.create_counter(
                        name=metric_name.replace(f"{self.app_name}_", ""),
                        documentation=documentation,
                        labelnames=labelnames
                    )
                elif metric_type == "FallbackGauge":
                    metric = self.create_gauge(
                        name=metric_name.replace(f"{self.app_name}_", ""),
                        documentation=documentation,
                        labelnames=labelnames
                    )
                elif metric_type == "FallbackHistogram":
                    metric = self.create_histogram(
                        name=metric_name.replace(f"{self.app_name}_", ""),
                        documentation=documentation,
                        labelnames=labelnames
                    )
                elif metric_type == "FallbackSummary":
                    metric = self.create_summary(
                        name=metric_name.replace(f"{self.app_name}_", ""),
                        documentation=documentation,
                        labelnames=labelnames
                    )
                else:
                    self.logger.warning(f"Unknown metric type: {metric_type}")
                    continue
                
                if "values" in metric_data and hasattr(metric, "values"):
                    for key_str, value in metric_data["values"].items():
                        if "|" in key_str and hasattr(metric, "labelnames") and metric.labelnames:
                            key = tuple(key_str.split("|"))
                        else:
                            key = key_str
                        
                        metric.values[key] = value
                
                if "counts" in metric_data and hasattr(metric, "counts"):
                    for key_str, value in metric_data["counts"].items():
                        if "|" in key_str and hasattr(metric, "labelnames") and metric.labelnames:
                            key = tuple(key_str.split("|"))
                        else:
                            key = key_str
                        
                        metric.counts[key] = value
            
            self.logger.info(f"Metrics loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
            return False
