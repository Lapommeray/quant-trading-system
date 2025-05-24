import numpy as np
from datetime import datetime, time
import logging

class DynamicLiquiditySlippage:
    """Enhanced slippage model with dynamic liquidity adjustment"""
    
    def __init__(self, base_spread_bps=1.0):
        self.base_spread_bps = base_spread_bps
        self.liquidity_cache = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DynamicLiquiditySlippage")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    def get_spread_bps(self, symbol):
        """Get base spread in basis points for a symbol"""
        default_spreads = {
            'BTC': 5.0,   # 5 bps for Bitcoin
            'ETH': 7.0,   # 7 bps for Ethereum
            'SPY': 1.0,   # 1 bps for S&P 500 ETF
            'QQQ': 1.0,   # 1 bps for Nasdaq ETF
            'GLD': 2.0,   # 2 bps for Gold ETF
            'DIA': 1.0    # 1 bps for Dow Jones ETF
        }
        
        ticker = symbol.split('.')[0] if '.' in symbol else symbol
        
        return default_spreads.get(ticker, self.base_spread_bps)
        
    def get_dynamic_slippage(self, symbol, size, market_conditions):
        """Calculate slippage based on simulated order book liquidity"""
        base_spread = self.get_spread_bps(symbol)
        
        # Simulate order book depth and liquidity
        order_book = self._simulate_order_book(symbol, market_conditions)
        mid_price = (order_book['best_bid'] + order_book['best_ask']) / 2
        
        # Calculate market impact based on order size vs available liquidity
        available_liquidity = order_book['total_bid_size'] + order_book['total_ask_size']
        
        size_ratio = size / available_liquidity if available_liquidity > 0 else 10.0
        
        size_impact = min(5.0, np.power(size_ratio, 0.6) * 2.0)
        
        direct_size_component = 0.01 * (size / 10000.0)  # 1 bp per 10k units
        
        volatility_factor = 1.0 + (market_conditions.get('volatility', 0.1) / 0.1)
        time_factor = self._get_time_factor(market_conditions.get('hour', datetime.now().hour))
        news_factor = market_conditions.get('news_factor', 1.0)
        
        # Exponential scaling for large orders during high volatility
        if size_ratio > 0.1 and market_conditions.get('volatility', 0.1) > 0.2:
            volatility_factor *= (1.0 + size_ratio)
        
        dynamic_spread = base_spread * volatility_factor * (1 + size_impact) * time_factor * news_factor
        
        dynamic_spread += direct_size_component * base_spread
        
        max_spread = base_spread * (10 + 5 * market_conditions.get('news_factor', 1.0))
        capped_spread = min(dynamic_spread, max_spread)
        
        self.logger.debug(f"Dynamic slippage for {symbol} (size={size}): {capped_spread:.2f} bps (size impact: {size_impact:.2f}, direct component: {direct_size_component * base_spread:.2f} bps)")
        
        return capped_spread / 10000.0  # Convert bps to decimal(1 bps = 0.0001)
        
    def _get_typical_depth(self, symbol):
        """Get typical order book depth for a symbol"""
        default_depths = {
            'BTC': 1000000,   # $1M for Bitcoin
            'ETH': 500000,    # $500k for Ethereum
            'SPY': 5000000,   # $5M for S&P 500 ETF
            'QQQ': 5000000,   # $5M for Nasdaq ETF
            'GLD': 2000000,   # $2M for Gold ETF
            'DIA': 3000000    # $3M for Dow Jones ETF
        }
        
        ticker = symbol.split('.')[0] if '.' in symbol else symbol
        
        return default_depths.get(ticker, 1000000)  # Default $1M
        
    def _simulate_order_book(self, symbol, market_conditions):
        """Simulate realistic order book conditions"""
        base_depth = self._get_typical_depth(symbol)
        volatility = market_conditions.get('volatility', 0.1)
        
        # Reduce liquidity during high volatility
        liquidity_factor = max(0.2, 1.0 - (volatility - 0.1) * 2)
        
        return {
            'best_bid': 100.0,  # Placeholder - would use real market data
            'best_ask': 100.05,
            'total_bid_size': base_depth * liquidity_factor * 0.5,
            'total_ask_size': base_depth * liquidity_factor * 0.5
        }
        
    def _get_time_factor(self, hour):
        """Get time-of-day liquidity factor"""
        us_market_open = 9  # 9 AM ET
        us_market_close = 16  # 4 PM ET
        
        asia_market_open = 20  # 8 PM ET (9 AM JST next day)
        asia_market_close = 3  # 3 AM ET (4 PM JST)
        
        europe_market_open = 3  # 3 AM ET (9 AM CET)
        europe_market_close = 11  # 11 AM ET (5 PM CET)
        
        in_us_market = us_market_open <= hour < us_market_close
        in_asia_market = asia_market_open <= hour or hour < asia_market_close
        in_europe_market = europe_market_open <= hour < europe_market_close
        
        if in_us_market and (in_asia_market or in_europe_market):
            return 1.0  # Maximum liquidity when multiple markets open
        elif in_us_market or in_asia_market or in_europe_market:
            return 1.2  # Good liquidity when at least one major market open
        else:
            return 1.5  # Lower liquidity during off-hours
            
    def calculate_slippage(self, symbol, price, size, side, market_conditions=None):
        """Calculate slippage for a trade"""
        if market_conditions is None:
            market_conditions = {
                'volatility': 0.1,  # 10% annualized volatility
                'hour': datetime.now().hour,
                'news_factor': 1.0
            }
            
        spread_decimal = self.get_dynamic_slippage(symbol, size, market_conditions)
        
        if side.upper() in ['BUY', 'B']:
            slippage = price * spread_decimal / 2  # Half spread for buys
        else:
            slippage = -price * spread_decimal / 2  # Negative half spread for sells
            
        slippage_amount = slippage * size
        
        self.logger.info(f"Slippage for {side} {size} {symbol} @ {price}: {slippage_amount:.2f} ({spread_decimal*10000:.2f} bps)")
        
        return slippage_amount
        
    def simulate_black_swan(self, symbol, normal_size):
        """Simulate slippage during black swan events"""
        black_swan_conditions = {
            'volatility': 0.5,  # 50% annualized volatility
            'hour': datetime.now().hour,
            'news_factor': 5.0,  # 5x normal impact
            'depth': self._get_typical_depth(symbol) * 0.2  # 80% reduction in liquidity
        }
        
        crisis_size = normal_size * 10
        
        crisis_spread = self.get_dynamic_slippage(symbol, crisis_size, black_swan_conditions)
        
        return crisis_spread * 10000  # Return in basis points
