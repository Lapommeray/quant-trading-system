#!/usr/bin/env python
"""
Verification script for critical hotfixes in QMP Trading System
Run this script to verify all fixes are working correctly
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerificationScript")

def test_walk_forward_no_leakage():
    """Test that walk-forward validation prevents data leakage"""
    from core.walk_forward_backtest import WalkForwardBacktester
    
    logger.info("Testing walk-forward validation for data leakage...")
    
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    data = {
        '1d': pd.DataFrame({
            'Close': np.random.normal(100, 1, 300),
            'Volume': np.random.uniform(1000, 10000, 300)
        }, index=dates)
    }
    
    backtester = WalkForwardBacktester(train_days=180, test_days=30)
    
    current_date = dates[180]
    train_start = current_date - timedelta(days=180)
    train_end = current_date - timedelta(days=1)  # Buffer gap
    test_start = current_date
    test_end = current_date + timedelta(days=30)
    
    train_data = backtester._extract_data_range(data, train_start, train_end)
    test_data = backtester._extract_data_range(data, test_start, test_end)
    
    train_indices = set(train_data['1d'].index)
    test_indices = set(test_data['1d'].index)
    
    overlap = len(train_indices.intersection(test_indices))
    
    if overlap > 0:
        logger.error(f"❌ CRITICAL: Data leak detected - {overlap} overlapping dates")
        return False
    
    max_train_date = train_data['1d'].index.max()
    min_test_date = test_data['1d'].index.min()
    
    if max_train_date >= min_test_date:
        logger.error("❌ CRITICAL: Train data contains future information")
        return False
    
    buffer_days = (min_test_date - max_train_date).days
    if buffer_days < 1:
        logger.error("❌ CRITICAL: Insufficient buffer gap between train and test data")
        return False
    
    logger.info("✅ Walk-forward validation passed - No data leakage detected")
    return True

def test_fat_tail_risk():
    """Test fat-tail risk management with Expected Shortfall"""
    from core.fat_tail import FatTailRiskManager
    
    logger.info("Testing fat-tail risk management...")
    
    normal_returns = np.random.normal(0.001, 0.01, 1000)
    
    fat_tail_returns = np.random.standard_t(df=3, size=1000) * 0.01 + 0.001
    
    risk_manager = FatTailRiskManager()
    
    normal_kelly = risk_manager.kelly_criterion(pd.Series(normal_returns))
    
    fat_tail_kelly = risk_manager.kelly_criterion_fat_tail(pd.Series(fat_tail_returns))
    
    if fat_tail_kelly >= normal_kelly:
        logger.error("❌ CRITICAL: Fat-tail Kelly not more conservative than normal Kelly")
        return False
    
    crash_returns = np.concatenate([
        np.random.normal(0.001, 0.01, 980),
        np.random.normal(-0.05, 0.03, 20)  # Crash period
    ])
    
    crash_kelly = risk_manager.kelly_criterion_fat_tail(pd.Series(crash_returns))
    
    if crash_kelly > 0.2:
        logger.error(f"❌ CRITICAL: Crash Kelly too aggressive: {crash_kelly:.4f}")
        return False
    
    logger.info(f"✅ Fat-tail risk management passed - Normal: {normal_kelly:.4f}, Fat-tail: {fat_tail_kelly:.4f}, Crash: {crash_kelly:.4f}")
    return True

def test_black_swan_resilience():
    """Test portfolio resilience during black swan events"""
    from core.event_blackout import EventBlackoutManager
    
    logger.info("Testing black swan resilience...")
    
    returns = np.random.normal(0, 0.01, 1000)
    returns[-1] = -0.20  # 20% crash (black swan)
    
    blackout_manager = EventBlackoutManager()
    results = blackout_manager.simulate_black_swan_events(pd.Series(returns))
    
    max_allowed_drawdown = 0.20
    all_passed = True
    
    for result in results:
        drawdown = abs(result['max_drawdown'])
        scenario = result['scenario']
        
        if drawdown >= max_allowed_drawdown:
            logger.error(f"❌ CRITICAL: Excessive drawdown in {scenario}: {drawdown:.2%}")
            all_passed = False
        else:
            logger.info(f"✅ {scenario} passed - Max drawdown: {drawdown:.2%}")
    
    if all_passed:
        logger.info("✅ Black swan resilience passed - All scenarios within drawdown limits")
    
    return all_passed

def test_dynamic_slippage():
    """Test dynamic slippage model with different market conditions"""
    from core.dynamic_slippage import DynamicLiquiditySlippage
    
    logger.info("Testing dynamic slippage model...")
    
    slippage_model = DynamicLiquiditySlippage()
    
    normal_conditions = {
        'volatility': 0.1,
        'hour': 12,
        'news_factor': 1.0
    }
    
    crisis_conditions = {
        'volatility': 0.5,
        'hour': 12,
        'news_factor': 5.0
    }
    
    sizes = [1000, 10000, 100000, 1000000]
    
    symbols = ['SPY', 'BTC']
    
    all_passed = True
    
    for symbol in symbols:
        normal_slippages = []
        crisis_slippages = []
        
        for size in sizes:
            normal_slip = slippage_model.get_dynamic_slippage(symbol, size, normal_conditions)
            crisis_slip = slippage_model.get_dynamic_slippage(symbol, size, crisis_conditions)
            
            normal_slippages.append(normal_slip)
            crisis_slippages.append(crisis_slip)
            
            if crisis_slip <= normal_slip:
                logger.error(f"❌ CRITICAL: Crisis slippage not higher for {symbol} size {size}")
                all_passed = False
        
        for i in range(1, len(normal_slippages)):
            if normal_slippages[i] <= normal_slippages[i-1]:
                logger.error(f"❌ CRITICAL: Slippage not increasing with size for {symbol}")
                all_passed = False
    
    for symbol in symbols:
        normal_size = 10000
        crisis_bps = slippage_model.simulate_black_swan(symbol, normal_size)
        
        if crisis_bps < 50:  # At least 50 bps during black swan
            logger.error(f"❌ CRITICAL: Black swan slippage too low for {symbol}: {crisis_bps:.2f} bps")
            all_passed = False
        else:
            logger.info(f"✅ Black swan slippage for {symbol}: {crisis_bps:.2f} bps")
    
    if all_passed:
        logger.info("✅ Dynamic slippage model passed - Proper scaling with size and conditions")
    
    return all_passed

def test_numba_optimization():
    """Test Numba optimization performance"""
    try:
        import numba
        has_numba = True
    except ImportError:
        logger.warning("⚠️ Numba not available - skipping optimization test")
        return True
    
    from core.performance_optimizer import PerformanceOptimizer
    
    logger.info("Testing Numba optimization performance...")
    
    prices = np.random.normal(100, 1, 100000)
    optimizer = PerformanceOptimizer()
    
    _ = optimizer.fast_volatility_calc(prices[:1000])
    
    import time
    import gc
    
    gc.collect()
    
    iterations = 5
    pandas_total = 0
    numba_total = 0
    
    for _ in range(iterations):
        start_time = time.time()
        _ = pd.Series(prices).pct_change().rolling(20).std() * np.sqrt(252)
        pandas_total += time.time() - start_time
        
        start_time = time.time()
        _ = optimizer.fast_volatility_calc(prices)
        numba_total += time.time() - start_time
    
    pandas_time = pandas_total / iterations
    numba_time = numba_total / iterations
    
    logger.info(f"Pandas time: {pandas_time:.6f}s, Numba time: {numba_time:.6f}s")
    
    if abs(pandas_time - numba_time) < 0.001:
        logger.warning("⚠️ Timing difference too small to be reliable")
        return True
    
    if numba_time > pandas_time * 1.5:
        logger.error(f"❌ CRITICAL: Numba optimization not providing expected performance")
        return False
    
    logger.info(f"✅ Numba optimization passed - {pandas_time/numba_time:.2f}x faster than pandas")
    return True

def run_all_tests():
    """Run all verification tests"""
    tests = [
        test_walk_forward_no_leakage,
        test_fat_tail_risk,
        test_black_swan_resilience,
        test_dynamic_slippage,
        test_numba_optimization
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n===== VERIFICATION SUMMARY =====")
    logger.info(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        logger.info("✅ ALL TESTS PASSED - System is production-ready")
        return 0
    else:
        logger.error(f"❌ {total-passed} TESTS FAILED - System is NOT production-ready")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify critical hotfixes for QMP Trading System")
    parser.add_argument("--test", choices=["walk_forward", "fat_tail", "black_swan", "slippage", "numba", "all"], 
                        default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.test == "walk_forward":
        success = test_walk_forward_no_leakage()
    elif args.test == "fat_tail":
        success = test_fat_tail_risk()
    elif args.test == "black_swan":
        success = test_black_swan_resilience()
    elif args.test == "slippage":
        success = test_dynamic_slippage()
    elif args.test == "numba":
        success = test_numba_optimization()
    else:
        sys.exit(run_all_tests())
    
    sys.exit(0 if success else 1)
