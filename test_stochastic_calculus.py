#!/usr/bin/env python3
"""
Test script to verify the AdvancedStochasticCalculus implementation
"""

import sys
import numpy as np
from advanced_modules.advanced_stochastic_calculus import AdvancedStochasticCalculus

def main():
    calc = AdvancedStochasticCalculus()
    
    has_method = hasattr(calc, 'calibrate_rough_volatility_model')
    print(f"Has calibrate_rough_volatility_model method: {has_method}")
    
    methods = [method for method in dir(calc) if not method.startswith('__')]
    print(f"Available methods: {methods}")
    
    if has_method:
        prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0])
        
        result = calc.calibrate_rough_volatility_model(prices)
        print(f"Method result: {result}")
    
if __name__ == "__main__":
    main()
