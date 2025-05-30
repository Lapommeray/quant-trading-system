import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

class DaskParallelProcessor:
    """
    Parallel processing using Dask for high-performance backtesting
    """
    def __init__(self, n_workers=None):
        self.client = None
        self.n_workers = n_workers
        
    def initialize(self):
        """
        Initialize Dask client
        """
        if self.client is None:
            cluster = LocalCluster(n_workers=self.n_workers)
            self.client = Client(cluster)
            print(f"Dask initialized with {self.n_workers} workers")
            return self.client.dashboard_link
            
    def shutdown(self):
        """
        Shutdown Dask client
        """
        if self.client is not None:
            self.client.close()
            self.client = None
            print("Dask client shut down")
            
    def parallelize_dataframe(self, df):
        """
        Convert pandas DataFrame to Dask DataFrame
        """
        return dd.from_pandas(df, npartitions=self.n_workers or 4)
        
    def parallel_backtest(self, strategy_func, data, parameters_list):
        """
        Run multiple backtests in parallel with different parameters
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            raise ValueError("Failed to initialize Dask client")
            
        futures = []
        for params in parameters_list:
            future = self.client.submit(strategy_func, data, **params)
            futures.append(future)
            
        if futures:
            results = self.client.gather(futures)
            return results
        return []
        
    def parallel_optimization(self, objective_func, param_space, n_trials=100):
        """
        Parallel hyperparameter optimization
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            raise ValueError("Failed to initialize Dask client")
            
        param_combinations = []
        for _ in range(n_trials):
            params = {k: np.random.choice(v) for k, v in param_space.items()}
            param_combinations.append(params)
            
        futures = [self.client.submit(objective_func, **params) for params in param_combinations]
        results = self.client.gather(futures)
        
        best_idx = np.argmax(results)
        best_params = param_combinations[best_idx]
        best_score = results[best_idx]
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': list(zip(param_combinations, results))
        }
