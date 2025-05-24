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
        """Calculate slippage based on dynamic market liquidity"""
        base_spread = self.get_spread_bps(symbol)
        
        volatility_factor = 1.0 + (market_conditions.get('volatility', 0.1) / 0.1)
        
        typical_depth = self._get_typical_depth(symbol)
        size_impact = min(2.0, size / typical_depth)
        
        current_hour = market_conditions.get('hour', datetime.now().hour)
        time_factor = self._get_time_factor(current_hour)
        
        news_factor = market_conditions.get('news_factor', 1.0)
        
        dynamic_spread = base_spread * volatility_factor * (1 + size_impact) * time_factor * news_factor
        
        capped_spread = min(dynamic_spread, base_spread * 10)
        
        self.logger.debug(f"Dynamic slippage for {symbol}: {capped_spread:.2f} bps (base: {base_spread:.2f})")
        
        return capped_spread / 10000.0  # Convert bps to decimal (1 bps = 0.0001)
        
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
