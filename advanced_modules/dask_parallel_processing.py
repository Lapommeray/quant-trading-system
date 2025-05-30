import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import logging

class DaskParallelProcessor:
    """
    Parallel processing using Dask for high-performance backtesting
    """
    def __init__(self, n_workers=4, threads_per_worker=2, memory_limit='4GB'):
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.client = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self):
        """
        Initialize Dask client
        """
        try:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit
            )
            self.client = Client(cluster)
            self.logger.info(f"Initialized Dask cluster with {self.n_workers} workers")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Dask cluster: {str(e)}")
            return False
            
    def shutdown(self):
        """
        Shutdown Dask client
        """
        if self.client is not None:
            self.client.close()
            self.client = None
            self.logger.info("Dask client shutdown")
            
    def parallel_backtest(self, strategy_func, data, parameters_list):
        """
        Run parallel backtests with different parameters
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            self.logger.warning("Falling back to sequential processing")
            results = []
            for params in parameters_list:
                result = strategy_func(data, **params)
                results.append(result)
            return results
            
        futures = []
        for params in parameters_list:
            future = self.client.submit(strategy_func, data, **params)
            futures.append(future)
            
        results = self.client.gather(futures)
        return results
        
    def parallel_optimization(self, objective_func, parameter_grid):
        """
        Run parallel optimization of parameters
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            self.logger.warning("Falling back to sequential processing for optimization")
            results = []
            for params in parameter_grid:
                result = objective_func(**params)
                results.append({
                    'parameters': params,
                    'result': result
                })
            results.sort(key=lambda x: x['result'], reverse=True)
            return results
            
        futures = []
        for params in parameter_grid:
            future = self.client.submit(objective_func, **params)
            futures.append((params, future))
            
        results = []
        for params, future in futures:
            result = self.client.gather(future)
            results.append({
                'parameters': params,
                'result': result
            })
            
        results.sort(key=lambda x: x['result'], reverse=True)
        return results
        
    def parallel_map(self, func, items, **kwargs):
        """
        Apply function to items in parallel
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            self.logger.warning("Falling back to sequential processing for map")
            results = []
            for item in items:
                result = func(item, **kwargs)
                results.append(result)
            return results
            
        futures = []
        for item in items:
            future = self.client.submit(func, item, **kwargs)
            futures.append(future)
            
        results = self.client.gather(futures)
        return results
        
    def parallel_dataframe_apply(self, df, func, axis=1, **kwargs):
        """
        Apply function to pandas DataFrame in parallel
        """
        if self.client is None:
            self.initialize()
            
        if self.client is None:
            self.logger.warning("Falling back to sequential processing for DataFrame apply")
            result = df.apply(func, axis=axis, **kwargs)
            return result
            
        chunks = np.array_split(df, self.n_workers * 2)
        
        futures = []
        for chunk in chunks:
            future = self.client.submit(lambda df: df.apply(func, axis=axis, **kwargs), chunk)
            futures.append(future)
            
        results = self.client.gather(futures)
        if axis == 0:
            return pd.concat(results, axis=1)
        else:
            return pd.concat(results, axis=0)
