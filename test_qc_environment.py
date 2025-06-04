"""
Test QuantConnect Environment Compatibility
Validates that the system works in a QC-like environment
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import sklearn
import statsmodels
from datetime import datetime
import logging

def test_python_version():
    """Test Python version compatibility"""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    
    if (major, minor) != (3, 10):
        print(f"❌ Python {major}.{minor} detected - QuantConnect requires Python 3.10")
        return False
    
    print("✅ Python 3.10 confirmed")
    return True

def test_package_versions():
    """Test package versions match QuantConnect requirements"""
    packages = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'matplotlib': matplotlib.__version__,
        'sklearn': sklearn.__version__,
        'statsmodels': statsmodels.__version__
    }
    
    print("\nPackage versions:")
    for package, version in packages.items():
        print(f"  {package}: {version}")
    
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version >= (1, 25):
        print("❌ NumPy version too high for QuantConnect")
        return False
    
    pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
    if pandas_version >= (2, 0):
        print("❌ Pandas version too high for QuantConnect")
        return False
    
    print("✅ All package versions compatible")
    return True

def test_no_dask():
    """Test that Dask is not installed (causes conflicts)"""
    try:
        import dask
        print("❌ Dask is installed - this will cause conflicts in QuantConnect")
        return False
    except ImportError:
        print("✅ Dask not installed - no conflicts")
        return True

def test_basic_functionality():
    """Test basic functionality that would be used in QuantConnect"""
    try:
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        
        df = pd.DataFrame({
            'price': [100, 101, 102, 101, 103],
            'volume': [1000, 1100, 900, 1200, 800]
        })
        df['returns'] = df['price'].pct_change()
        
        from scipy import stats
        price_data = df['price'].dropna()
        volume_data = df['volume'][:len(price_data)]
        correlation = stats.pearsonr(price_data, volume_data)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage is reasonable"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    if memory_mb > 500:  # 500MB threshold
        print("⚠️ High memory usage detected")
        return False
    
    print("✅ Memory usage acceptable")
    return True

def main():
    """Run all QuantConnect compatibility tests"""
    print("🚀 QuantConnect Environment Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Versions", test_package_versions),
        ("No Dask Conflicts", test_no_dask),
        ("Basic Functionality", test_basic_functionality),
        ("Memory Usage", test_memory_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready for QuantConnect deployment!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Fix issues before QuantConnect deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
