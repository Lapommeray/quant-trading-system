"""
Test Performance Optimizer

This module tests the memory-efficient Numba-accelerated functions in the PerformanceOptimizer class.
"""

import unittest
import numpy as np
import sys
import os
import time
import psutil
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_optimizer import PerformanceOptimizer

class TestPerformanceOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)  # For reproducibility
        self.prices = np.random.normal(100, 1, 1000)
        self.volumes = np.random.uniform(1000, 10000, 1000)
        self.returns = np.random.normal(0.001, 0.01, 1000)
        
    def test_fast_ema_accuracy(self):
        """Test that fast_ema produces accurate results"""
        window = 20
        alpha = 2.0 / (window + 1.0)
        expected_ema = np.zeros_like(self.prices)
        expected_ema[0] = self.prices[0]
        for i in range(1, len(self.prices)):
            expected_ema[i] = alpha * self.prices[i] + (1.0 - alpha) * expected_ema[i-1]
            
        actual_ema = PerformanceOptimizer.fast_ema(self.prices, window)
        
        np.testing.assert_allclose(actual_ema, expected_ema, rtol=1e-10)
        
    def test_fast_volatility_accuracy(self):
        """Test that fast_volatility produces accurate results"""
        expected_vol = np.std(self.returns, ddof=1) * np.sqrt(252)
        
        actual_vol = PerformanceOptimizer.fast_volatility(self.returns)
        
        self.assertAlmostEqual(actual_vol, expected_vol, places=10)
        
    def test_fast_returns_accuracy(self):
        """Test that fast_returns produces accurate results"""
        expected_returns = np.zeros(len(self.prices) - 1)
        for i in range(1, len(self.prices)):
            expected_returns[i-1] = (self.prices[i] / self.prices[i-1]) - 1.0
            
        actual_returns = PerformanceOptimizer.fast_returns(self.prices)
        
        np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-10)
        
    def test_memory_efficiency(self):
        """Test that functions don't leak memory during repeated calls"""
        gc.collect()
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        large_prices = np.random.normal(100, 1, 1000000)
        
        out_array = np.empty_like(large_prices)
        
        for _ in range(100):
            PerformanceOptimizer.fast_ema(large_prices, 20, out_array)
            
        gc.collect()
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        self.assertLess(memory_increase, 20)  # Less than 20MB increase
        
    def test_performance_improvement(self):
        """Test that Numba functions are significantly faster than pure Python"""
        large_prices = np.random.normal(100, 1, 100000)
        window = 20
        
        def python_ema(prices, window):
            alpha = 2.0 / (window + 1.0)
            result = np.zeros_like(prices)
            result[0] = prices[0]
            for i in range(1, len(prices)):
                result[i] = alpha * prices[i] + (1.0 - alpha) * result[i-1]
            return result
        
        start_time = time.time()
        python_result = python_ema(large_prices, window)
        python_time = time.time() - start_time
        
        _ = PerformanceOptimizer.fast_ema(large_prices[:100], window)  # Warm-up run
        start_time = time.time()
        numba_result = PerformanceOptimizer.fast_ema(large_prices, window)
        numba_time = time.time() - start_time
        
        np.testing.assert_allclose(numba_result, python_result, rtol=1e-10)
        
        self.assertLess(numba_time * 10, python_time)
        
    def test_preallocation_benefit(self):
        """Test that pre-allocated arrays improve performance"""
        large_prices = np.random.normal(100, 1, 1000000)
        window = 20
        
        _ = PerformanceOptimizer.fast_ema(large_prices[:1000], window)
        
        start_time = time.time()
        for _ in range(10):  # Fewer iterations but larger array
            result = PerformanceOptimizer.fast_ema(large_prices, window)
            self.assertIsNotNone(result)
        no_prealloc_time = time.time() - start_time
        
        out_array = np.empty_like(large_prices)
        start_time = time.time()
        for _ in range(10):  # Fewer iterations but larger array
            result = PerformanceOptimizer.fast_ema(large_prices, window, out_array)
            self.assertIsNotNone(result)
        prealloc_time = time.time() - start_time
        
        if abs(prealloc_time - no_prealloc_time) < 0.01:
            self.skipTest("Timing difference too small to be reliable")
        else:
            self.assertLess(prealloc_time, no_prealloc_time)
        
    def test_parallel_processing(self):
        """Test that parallel processing works correctly"""
        arrays = [np.random.normal(100, 1, 1000) for _ in range(5)]
        lookbacks = [10, 20, 30, 40, 50]
        
        results = PerformanceOptimizer.parallel_signal_processing(arrays, lookbacks)
        
        self.assertEqual(len(results), len(arrays))
        
        for i, result in enumerate(results):
            self.assertEqual(len(result), len(arrays[i]))

if __name__ == '__main__':
    unittest.main()
