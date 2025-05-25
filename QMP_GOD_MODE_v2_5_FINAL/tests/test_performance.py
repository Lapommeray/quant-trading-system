#!/usr/bin/env python
"""
Performance test module for QMP Trading System
"""

import sys
import os
import unittest
import time
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.performance_optimizer import PerformanceOptimizer
    from core.async_api_client import AsyncQMPApiClient
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

class TestPerformance(unittest.TestCase):
    
    @unittest.skipIf(not HAS_MODULES, "Required modules not available")
    def test_latency(self):
        """Test order execution latency"""
        optimizer = PerformanceOptimizer()
        
        prices = np.random.normal(100, 1, 10000)
        
        start_time = time.time()
        result = optimizer.fast_volatility_calc(prices)
        latency = time.time() - start_time
        
        self.assertLess(latency, 0.01, f"Latency too high: {latency:.4f}s")
        print(f"Latency test passed: {latency*1000:.2f}ms")
        
    @unittest.skipIf(not HAS_MODULES, "Required modules not available")  
    def test_memory(self):
        """Test for memory leaks in critical components"""
        optimizer = PerformanceOptimizer()
        
        initial_usage = self._get_memory_usage()
        
        for _ in range(100):
            prices = np.random.normal(100, 1, 1000)
            _ = optimizer.fast_volatility_calc(prices)
            
        final_usage = self._get_memory_usage()
        
        memory_increase = final_usage - initial_usage
        self.assertLess(memory_increase, 50, f"Memory leak detected: {memory_increase}MB increase")
        print(f"Memory test passed: {memory_increase:.1f}MB increase")
        
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0

def main():
    parser = argparse.ArgumentParser(description="Performance tests for QMP Trading System")
    parser.add_argument("--test", choices=["latency", "memory", "all"], 
                        default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    suite = unittest.TestSuite()
    
    if args.test in ["latency", "all"]:
        suite.addTest(TestPerformance('test_latency'))
    if args.test in ["memory", "all"]:
        suite.addTest(TestPerformance('test_memory'))
        
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())
