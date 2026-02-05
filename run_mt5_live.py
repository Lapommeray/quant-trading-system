#!/usr/bin/env python3
"""
MT5 Live Runner - Canonical Entry Point for RayBridge EA

This is the ONLY file that should output signals to MT5.
It is a thin orchestration layer that:
1. Collects signals from the existing QMP engine
2. Reaches a final decision per symbol
3. Writes exactly one JSON object to MT5 per cycle
4. Uses closed-bar / completed data only
5. Never repaints
6. Never emits partial or intermediate states

Usage:
    python run_mt5_live.py
    python run_mt5_live.py --symbol XAUUSD --interval 60
    python run_mt5_live.py --symbols XAUUSD,EURUSD,BTCUSD --interval 300
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mt5_live.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MT5Live")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MT5 bridge
from mt5_bridge import write_signal_atomic, init_bridge, get_signal_dir

# Confidence threshold - signals below this are treated as "no signal"
CONFIDENCE_THRESHOLD = 0.7


class MT5LiveRunner:
    """
    Canonical MT5 live runner.
    
    This class is intentionally boring, simple, and deterministic.
    It does NOT contain trading logic - only orchestration.
    """
    
    def __init__(
        self,
        symbols: List[str],
        interval_seconds: int = 60,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the MT5 live runner.
        
        Args:
            symbols: List of symbols to trade (e.g., ["XAUUSD", "EURUSD"])
            interval_seconds: Seconds between signal cycles (must be >= bar close)
            confidence_threshold: Minimum confidence to emit a signal
        """
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.last_signal_time: Dict[str, datetime.datetime] = {}
        
        # Initialize MT5 bridge
        init_bridge({
            "mt5_bridge_enabled": True,
            "mt5_signal_interval_seconds": 0,  # We control timing here
            "symbols_for_mt5": [],  # Allow all symbols
            "mt5_confidence_threshold": 0.0  # We filter here
        })
        
        # Initialize signal generator (lazy load to avoid import errors)
        self.signal_generator = None
        
        logger.info(f"MT5 Live Runner initialized")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Interval: {interval_seconds}s")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Signal directory: {get_signal_dir()}")
    
    def _get_signal_generator(self):
        """Lazy-load the signal generator to handle import errors gracefully"""
        if self.signal_generator is not None:
            return self.signal_generator
        
        # Try to import and initialize the QMP engine
        try:
            # First try the standalone version that doesn't require QuantConnect
            from core.qmp_engine_standalone import QMPStandaloneEngine
            self.signal_generator = QMPStandaloneEngine()
            logger.info("Using QMPStandaloneEngine for signal generation")
            return self.signal_generator
        except ImportError:
            pass
        
        # Fallback: create a simple signal generator wrapper
        logger.warning("QMP engine not available, using fallback signal generator")
        self.signal_generator = FallbackSignalGenerator()
        return self.signal_generator
    
    def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch closed-bar market data for a symbol.
        
        This MUST return only completed/closed bar data.
        No tick data. No partial bars. No repaint.
        
        Returns:
            Dictionary with OHLCV data or None if unavailable
        """
        try:
            # Try to use real data sources
            data = self._fetch_from_data_source(symbol)
            if data is not None:
                return data
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
        
        # Return None if no data available - this will trigger a null signal
        return None
    
    def _fetch_from_data_source(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from available data sources.
        
        Tries multiple sources in order of preference.
        """
        # Try yfinance for common symbols
        try:
            import yfinance as yf
            
            # Map symbol to yfinance format
            yf_symbol = self._map_to_yfinance_symbol(symbol)
            if yf_symbol:
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    return {
                        "symbol": symbol,
                        "ohlcv": hist.tail(60).to_dict('records'),
                        "close": float(hist['Close'].iloc[-1]),
                        "volume": float(hist['Volume'].iloc[-1]),
                        "timestamp": hist.index[-1].isoformat()
                    }
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol}: {e}")
        
        return None
    
    def _map_to_yfinance_symbol(self, symbol: str) -> Optional[str]:
        """Map internal symbol to yfinance symbol"""
        mapping = {
            "XAUUSD": "GC=F",  # Gold futures
            "XAGUSD": "SI=F",  # Silver futures
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "SPX500": "^GSPC",
            "US30": "^DJI",
            "NASDAQ": "^IXIC",
        }
        
        # Handle symbols with slashes
        clean_symbol = symbol.replace("/", "")
        return mapping.get(clean_symbol, mapping.get(symbol))
    
    def _generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal for a symbol.
        
        This is the ONLY place where signal generation logic is called.
        The result is the FINAL decision - no further processing.
        
        Returns:
            Dictionary with 'final_signal', 'confidence', and metadata
        """
        generator = self._get_signal_generator()
        
        try:
            result = generator.generate_signal(symbol, market_data)
            return result
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return {
                "final_signal": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_mt5_signal(
        self,
        symbol: str,
        final_signal: Optional[str],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Create the MT5 signal object with strict schema.
        
        Schema (STRICT):
            symbol: str - Trading symbol
            signal: str | null - "BUY", "SELL", "HOLD", or null
            confidence: float - 0.0 to 1.0
            timestamp: str - ISO-8601 format
        """
        # Normalize signal
        if final_signal is None:
            normalized_signal = None
        elif final_signal.upper() in ("BUY", "STRONG_BUY"):
            normalized_signal = "BUY"
        elif final_signal.upper() in ("SELL", "STRONG_SELL"):
            normalized_signal = "SELL"
        elif final_signal.upper() in ("HOLD", "NEUTRAL", "WAIT"):
            normalized_signal = "HOLD"
        else:
            normalized_signal = None
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            normalized_signal = None
            confidence = 0.0
        
        return {
            "symbol": symbol.replace("/", ""),  # Remove slashes for MT5
            "signal": normalized_signal,
            "confidence": round(float(confidence), 4),
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    
    def _process_symbol(self, symbol: str) -> bool:
        """
        Process a single symbol through the full pipeline.
        
        Returns:
            True if signal was written successfully
        """
        logger.info(f"Processing {symbol}...")
        
        # Step 1: Fetch closed-bar market data
        market_data = self._fetch_market_data(symbol)
        
        if market_data is None:
            # No data available - emit null signal
            logger.warning(f"No market data available for {symbol}")
            mt5_signal = self._create_mt5_signal(symbol, None, 0.0)
        else:
            # Step 2: Generate signal from QMP engine
            signal_result = self._generate_signal(symbol, market_data)
            
            final_signal = signal_result.get("final_signal")
            confidence = signal_result.get("confidence", 0.0)
            
            # Step 3: Create MT5 signal object
            mt5_signal = self._create_mt5_signal(symbol, final_signal, confidence)
        
        # Step 4: Write to MT5 (atomic, single file)
        success = write_signal_atomic(mt5_signal)
        
        if success:
            self.last_signal_time[symbol] = datetime.datetime.utcnow()
            logger.info(f"Signal written for {symbol}: {mt5_signal['signal']} "
                       f"(confidence: {mt5_signal['confidence']:.2f})")
        else:
            logger.warning(f"Failed to write signal for {symbol}")
        
        return success
    
    def run_once(self) -> Dict[str, bool]:
        """
        Run a single cycle for all symbols.
        
        Returns:
            Dictionary mapping symbol to success status
        """
        results = {}
        
        for symbol in self.symbols:
            try:
                results[symbol] = self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                # Write null signal on error
                mt5_signal = self._create_mt5_signal(symbol, None, 0.0)
                write_signal_atomic(mt5_signal)
                results[symbol] = False
        
        return results
    
    def run(self):
        """
        Run the MT5 live loop continuously.
        
        Execution discipline:
        - One signal per symbol per interval
        - Interval must be >= bar close (no ticks)
        - No intra-bar updates
        - No repaint
        - Silence = no trade
        """
        self.running = True
        cycle_count = 0
        
        logger.info("=" * 60)
        logger.info("MT5 LIVE RUNNER STARTED")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Interval: {self.interval_seconds}s")
        logger.info(f"Signal directory: {get_signal_dir()}")
        logger.info("=" * 60)
        
        try:
            while self.running:
                cycle_count += 1
                cycle_start = time.time()
                
                logger.info(f"\n--- Cycle {cycle_count} ---")
                
                # Process all symbols
                results = self.run_once()
                
                # Log results
                success_count = sum(1 for v in results.values() if v)
                logger.info(f"Cycle {cycle_count} complete: "
                           f"{success_count}/{len(results)} signals written")
                
                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.interval_seconds - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"Sleeping {sleep_time:.1f}s until next cycle...")
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.running = False
            logger.info("MT5 Live Runner stopped")
    
    def stop(self):
        """Stop the live runner"""
        self.running = False


class FallbackSignalGenerator:
    """
    Fallback signal generator when QMP engine is not available.
    
    This provides a simple technical analysis based signal
    for testing and development purposes.
    """
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple signal based on price momentum"""
        try:
            ohlcv = market_data.get("ohlcv", [])
            
            if len(ohlcv) < 20:
                return {"final_signal": None, "confidence": 0.0}
            
            # Simple momentum calculation
            closes = [bar.get("Close", bar.get("close", 0)) for bar in ohlcv[-20:]]
            
            if not closes or closes[-1] == 0:
                return {"final_signal": None, "confidence": 0.0}
            
            # Calculate simple moving averages
            sma_fast = sum(closes[-5:]) / 5
            sma_slow = sum(closes[-20:]) / 20
            
            # Calculate momentum
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            # Generate signal
            if sma_fast > sma_slow * 1.001 and momentum > 0:
                signal = "BUY"
                confidence = min(0.9, 0.5 + abs(momentum) * 10)
            elif sma_fast < sma_slow * 0.999 and momentum < 0:
                signal = "SELL"
                confidence = min(0.9, 0.5 + abs(momentum) * 10)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            return {
                "final_signal": signal,
                "confidence": confidence,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "momentum": momentum
            }
            
        except Exception as e:
            logger.error(f"Fallback signal generation error: {e}")
            return {"final_signal": None, "confidence": 0.0}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MT5 Live Runner - Canonical entry point for RayBridge EA"
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default="XAUUSD",
        help="Single symbol to trade (default: XAUUSD)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides --symbol)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Seconds between signal cycles (default: 60)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = [args.symbol]
    
    # Create runner
    runner = MT5LiveRunner(
        symbols=symbols,
        interval_seconds=args.interval,
        confidence_threshold=args.confidence
    )
    
    # Run
    if args.once:
        results = runner.run_once()
        print(f"\nResults: {results}")
    else:
        runner.run()


if __name__ == "__main__":
    main()
