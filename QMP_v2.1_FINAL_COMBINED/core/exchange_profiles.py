"""
Exchange-Specific Circuit Breaker Tuning Module

This module implements exchange-specific circuit breaker profiles for the QMP Overrider system.
It provides pre-configured exchange profiles with optimized parameters for different market types.
"""

from dataclasses import dataclass

@dataclass
class ExchangeProfile:
    """
    Exchange-specific circuit breaker profile with optimized parameters.
    
    Attributes:
        name: Exchange name identifier
        volatility_threshold: Threshold for volatility-based circuit breaking
        latency_floor_ms: Baseline latency for the exchange in milliseconds
        imbalance_multiplier: Order book imbalance threshold multiplier
        cooling_period: Cooling period after circuit breaker activation in seconds
    """
    name: str
    volatility_threshold: float
    latency_floor_ms: int
    imbalance_multiplier: float
    cooling_period: int

EXCHANGE_PROFILES = {
    "binance": ExchangeProfile(
        name="binance",
        volatility_threshold=0.12,  # Higher tolerance for crypto
        latency_floor_ms=25,
        imbalance_multiplier=2.5,
        cooling_period=45
    ),
    "binance_futures": ExchangeProfile(
        name="binance_futures",
        volatility_threshold=0.15,  # Even higher for futures
        latency_floor_ms=20,
        imbalance_multiplier=3.0,
        cooling_period=30
    ),
    "coinbase": ExchangeProfile(
        name="coinbase",
        volatility_threshold=0.10,
        latency_floor_ms=30,
        imbalance_multiplier=2.2,
        cooling_period=60
    ),
    "kraken": ExchangeProfile(
        name="kraken",
        volatility_threshold=0.11,
        latency_floor_ms=35,
        imbalance_multiplier=2.3,
        cooling_period=50
    ),
    "nyse": ExchangeProfile(
        name="nyse",
        volatility_threshold=0.07,  # Stricter for equities
        latency_floor_ms=5,
        imbalance_multiplier=1.8,
        cooling_period=300
    ),
    "nasdaq": ExchangeProfile(
        name="nasdaq",
        volatility_threshold=0.08,
        latency_floor_ms=4,
        imbalance_multiplier=1.9,
        cooling_period=240
    ),
    "cme": ExchangeProfile(
        name="cme",
        volatility_threshold=0.09,
        latency_floor_ms=8,
        imbalance_multiplier=2.0,
        cooling_period=180
    ),
    "lse": ExchangeProfile(
        name="lse",
        volatility_threshold=0.06,
        latency_floor_ms=10,
        imbalance_multiplier=1.7,
        cooling_period=270
    ),
    "jse": ExchangeProfile(
        name="jse",
        volatility_threshold=0.08,
        latency_floor_ms=15,
        imbalance_multiplier=2.1,
        cooling_period=210
    ),
    "asx": ExchangeProfile(
        name="asx",
        volatility_threshold=0.07,
        latency_floor_ms=18,
        imbalance_multiplier=1.9,
        cooling_period=240
    ),
    "darkpool_x": ExchangeProfile(
        name="darkpool_x",
        volatility_threshold=0.15,  # Less transparent
        latency_floor_ms=50,
        imbalance_multiplier=3.0,
        cooling_period=120
    ),
    "darkpool_y": ExchangeProfile(
        name="darkpool_y",
        volatility_threshold=0.14,
        latency_floor_ms=45,
        imbalance_multiplier=2.8,
        cooling_period=150
    ),
    "otc": ExchangeProfile(
        name="otc",
        volatility_threshold=0.20,  # Highest tolerance
        latency_floor_ms=60,
        imbalance_multiplier=3.5,
        cooling_period=90
    )
}

class AdaptiveBreaker:
    """
    Adaptive circuit breaker that adjusts parameters based on market conditions.
    
    This class extends the basic circuit breaker functionality with dynamic parameter
    adjustment based on current market conditions.
    """
    
    def __init__(self, exchange_name, base_config=None):
        """
        Initialize the adaptive circuit breaker with exchange-specific parameters.
        
        Parameters:
            exchange_name: Name of the exchange to use profile for
            base_config: Optional base configuration to override profile
        """
        from circuit_breakers.hft_guardian import BreakerConfig, HFTCircuitBreaker
        
        self.exchange = exchange_name
        
        if exchange_name in EXCHANGE_PROFILES:
            profile = EXCHANGE_PROFILES[exchange_name]
        else:
            profile = ExchangeProfile(
                name="default",
                volatility_threshold=0.10,
                latency_floor_ms=20,
                imbalance_multiplier=2.0,
                cooling_period=60
            )
        
        config = BreakerConfig(
            volatility_threshold=profile.volatility_threshold,
            latency_spike_ms=profile.latency_floor_ms * 3,  # 3x normal is a spike
            order_imbalance_ratio=profile.imbalance_multiplier,
            cooling_off_period=profile.cooling_period
        )
        
        if base_config:
            for key, value in base_config.__dict__.items():
                if value is not None:
                    setattr(config, key, value)
        
        self.breaker = HFTCircuitBreaker(config)
        self.profile = profile
        self.market_condition = "normal"
    
    def dynamic_adjust(self, market_condition):
        """
        Dynamically adjust circuit breaker parameters based on market conditions.
        
        Parameters:
            market_condition: Current market condition identifier
                (e.g., "normal", "high_volume", "news_event", "earnings", "fomc")
        """
        self.market_condition = market_condition
        
        if market_condition == "high_volume":
            self.breaker.config.volatility_threshold *= 1.3
            self.breaker.config.latency_spike_ms *= 1.2
            
        elif market_condition == "news_event":
            self.breaker.config.order_imbalance_ratio *= 0.7
            self.breaker.config.cooling_off_period *= 1.5
            
        elif market_condition == "earnings":
            self.breaker.config.volatility_threshold *= 1.5
            self.breaker.config.cooling_off_period *= 2.0
            
        elif market_condition == "fomc":
            self.breaker.config.volatility_threshold *= 2.0
            self.breaker.config.order_imbalance_ratio *= 0.5
            self.breaker.config.cooling_off_period *= 3.0
            
        elif market_condition == "low_liquidity":
            self.breaker.config.volatility_threshold *= 0.7
            self.breaker.config.order_imbalance_ratio *= 0.6
            
        elif market_condition == "normal":
            self.breaker.config.volatility_threshold = self.profile.volatility_threshold
            self.breaker.config.latency_spike_ms = self.profile.latency_floor_ms * 3
            self.breaker.config.order_imbalance_ratio = self.profile.imbalance_multiplier
            self.breaker.config.cooling_off_period = self.profile.cooling_period
    
    def check_conditions(self, market_data):
        """
        Check if circuit breaker conditions are met.
        
        Parameters:
            market_data: Market data object with volatility, latency, and bid_ask_imbalance
            
        Returns:
            Boolean indicating if circuit breaker should be triggered
        """
        return self.breaker.check_conditions(market_data)
    
    def trigger(self, strategy):
        """
        Trigger the circuit breaker.
        
        Parameters:
            strategy: Trading strategy object to apply circuit breaker actions to
        """
        self.breaker.trigger(strategy)
    
    def reset(self):
        """Reset the circuit breaker state."""
        self.breaker.reset()
