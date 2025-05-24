import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dynamic_slippage import DynamicLiquiditySlippage

class TestDynamicSlippage(unittest.TestCase):
    def setUp(self):
        """Set up test slippage model"""
        self.slippage_model = DynamicLiquiditySlippage(base_spread_bps=1.0)
        
    def test_normal_market_slippage(self):
        """Test slippage under normal market conditions"""
        normal_conditions = {
            'volatility': 0.1,  # 10% annualized volatility
            'hour': 12,         # Middle of US trading day
            'news_factor': 1.0  # No significant news
        }
        
        small_order_slippage = self.slippage_model.get_dynamic_slippage('SPY', 1000, normal_conditions)
        large_order_slippage = self.slippage_model.get_dynamic_slippage('SPY', 100000, normal_conditions)
        
        self.assertGreater(small_order_slippage, 0)
        self.assertGreater(large_order_slippage, 0)
        
        self.assertGreater(large_order_slippage, small_order_slippage)
        
        self.assertLess(small_order_slippage, 0.001)  # Less than 10 bps for small orders
        
    def test_high_volatility_slippage(self):
        """Test slippage during high volatility"""
        high_vol_conditions = {
            'volatility': 0.3,  # 30% annualized volatility
            'hour': 12,
            'news_factor': 1.5  # Some market-moving news
        }
        
        normal_vol_conditions = {
            'volatility': 0.1,  # 10% annualized volatility
            'hour': 12,
            'news_factor': 1.0
        }
        
        order_size = 10000
        
        high_vol_slippage = self.slippage_model.get_dynamic_slippage('SPY', order_size, high_vol_conditions)
        normal_vol_slippage = self.slippage_model.get_dynamic_slippage('SPY', order_size, normal_vol_conditions)
        
        self.assertGreater(high_vol_slippage, normal_vol_slippage)
        
        ratio = high_vol_slippage / normal_vol_slippage
        self.assertGreater(ratio, 1.5)  # At least 50% higher
        
    def test_order_size_scaling(self):
        """Test that slippage scales properly with order size"""
        conditions = {
            'volatility': 0.1,
            'hour': 12,
            'news_factor': 1.0
        }
        
        sizes = [1000, 10000, 100000, 1000000]
        slippages = [self.slippage_model.get_dynamic_slippage('SPY', size, conditions) for size in sizes]
        
        for i in range(1, len(slippages)):
            self.assertGreater(slippages[i], slippages[i-1])
            
        mid_idx = len(sizes) // 2
        size_ratio = sizes[mid_idx+1] / sizes[mid_idx]
        slippage_ratio = slippages[mid_idx+1] / slippages[mid_idx]
        
        self.assertLess(slippage_ratio, size_ratio)
        
    def test_black_swan_slippage(self):
        """Test slippage during black swan events"""
        normal_size = 10000
        
        crisis_slippage_bps = self.slippage_model.simulate_black_swan('SPY', normal_size)
        
        normal_conditions = {
            'volatility': 0.1,
            'hour': 12,
            'news_factor': 1.0
        }
        normal_slippage = self.slippage_model.get_dynamic_slippage('SPY', normal_size, normal_conditions)
        normal_slippage_bps = normal_slippage * 10000  # Convert to bps
        
        self.assertGreater(crisis_slippage_bps, normal_slippage_bps * 5)
        
        black_swan_conditions = {
            'volatility': 0.5,  # 50% annualized volatility
            'hour': 12,
            'news_factor': 5.0  # Major market event
        }
        
        direct_crisis_slippage = self.slippage_model.get_dynamic_slippage('SPY', normal_size * 10, black_swan_conditions)
        direct_crisis_slippage_bps = direct_crisis_slippage * 10000
        
        ratio = abs(direct_crisis_slippage_bps / crisis_slippage_bps - 1.0)
        self.assertLess(ratio, 0.5)  # Within 50% of each other
        
    def test_liquidity_differences(self):
        """Test that less liquid assets have higher slippage"""
        conditions = {
            'volatility': 0.1,
            'hour': 12,
            'news_factor': 1.0
        }
        
        size = 10000
        
        spy_slippage = self.slippage_model.get_dynamic_slippage('SPY', size, conditions)
        btc_slippage = self.slippage_model.get_dynamic_slippage('BTC', size, conditions)
        
        self.assertGreater(btc_slippage, spy_slippage)
        
    def test_time_of_day_impact(self):
        """Test that time of day affects slippage"""
        size = 10000
        
        us_market_conditions = {
            'volatility': 0.1,
            'hour': 12,  # Middle of US trading
            'news_factor': 1.0
        }
        
        off_hours_conditions = {
            'volatility': 0.1,
            'hour': 22,  # After US market close
            'news_factor': 1.0
        }
        
        us_market_slippage = self.slippage_model.get_dynamic_slippage('SPY', size, us_market_conditions)
        off_hours_slippage = self.slippage_model.get_dynamic_slippage('SPY', size, off_hours_conditions)
        
        self.assertGreater(off_hours_slippage, us_market_slippage)

if __name__ == '__main__':
    unittest.main()
