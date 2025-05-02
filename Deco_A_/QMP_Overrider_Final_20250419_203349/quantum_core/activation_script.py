"""
Quantum Core Activation Script

This script activates the quantum core components for the QMP Overrider system.
"""

import os
import sys
import argparse
from datetime import datetime

from .quantum_oracle import QuantumOracle
from .market_maker_mind_reader import MarketMakerMindReader
from .time_fractal_predictor import TimeFractalPredictor
from .black_swan_hunter import BlackSwanHunter
from .god_mode_trader import GodModeTrader

def activate_god_mode(symbols=None, precision="attosecond"):
    """
    Activate God Mode for the specified symbols
    
    Parameters:
    - symbols: List of symbols to activate God Mode for
    - precision: Precision level for quantum operations
    
    Returns:
    - GodModeTrader instance
    """
    if symbols is None:
        symbols = ["BTCUSD", "SPY", "QQQ"]
    
    print(f"Activating Quantum Core with {precision} precision")
    print(f"Symbols: {', '.join(symbols)}")
    
    god_trader = GodModeTrader()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] God Mode activated successfully")
    
    return god_trader

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Activate Quantum Core")
    parser.add_argument("--symbols", type=str, default="BTCUSD,SPY,QQQ",
                        help="Comma-separated list of symbols")
    parser.add_argument("--precision", type=str, default="attosecond",
                        choices=["picosecond", "femtosecond", "attosecond"],
                        help="Precision level for quantum operations")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    
    god_trader = activate_god_mode(symbols, args.precision)
    
    for symbol in symbols:
        action = god_trader.execute_divine_trade(symbol)
        print(f"Divine action for {symbol}: {action}")

if __name__ == "__main__":
    main()
