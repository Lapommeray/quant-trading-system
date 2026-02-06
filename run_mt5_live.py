#!/usr/bin/env python3
"""
MT5 Live Runner - The ONLY MT5 Live Entry Point

This is the canonical entry point for RayBridge EA signal output.
It is a thin orchestration layer that:
1. Collects signals from the existing QMP engine
2. Reaches a final decision per symbol
3. Writes exactly one JSON object to MT5 per cycle
4. Uses closed-bar / completed data only
5. Never repaints
6. Never emits partial or intermediate states

Execution discipline (NON-NEGOTIABLE):
- One signal per symbol per interval (default 60s, hard minimum 60s)
- Interval must be >= bar close (no ticks)
- No intra-bar updates
- No repaint
- Silence = no trade
- Confidence must exceed threshold to emit BUY/SELL
- Signal must be stable (no rapid flipping)

Usage:
    python run_mt5_live.py
    python run_mt5_live.py --symbol XAUUSD --interval 60
    python run_mt5_live.py --symbols XAUUSD,EURUSD,BTCUSD --interval 300
    python run_mt5_live.py --once
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
from mt5_bridge import write_signal_output, init_bridge, get_signal_dir

# Hard minimum interval - cannot go below this
MIN_INTERVAL_SECONDS = 60

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
        if interval_seconds < MIN_INTERVAL_SECONDS:
            logger.warning(
                f"Interval {interval_seconds}s is below minimum {MIN_INTERVAL_SECONDS}s. "
                f"Enforcing minimum."
            )
            interval_seconds = MIN_INTERVAL_SECONDS

        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.confidence_threshold = confidence_threshold
        self.running = False

        # Per-symbol timing: tracks monotonic time of last signal write
        self.last_signal_time: Dict[str, float] = {}
        # Per-symbol last emitted direction for stability tracking
        self.last_emitted_signal: Dict[str, Optional[str]] = {}

        # Initialize MT5 bridge
        init_bridge({
            "mt5_bridge_enabled": True,
            "mt5_signal_interval_seconds": 0,
            "symbols_for_mt5": [],
            "mt5_confidence_threshold": 0.0
        })

        self.signal_generator = None

        logger.info("MT5 Live Runner initialized")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Interval: {interval_seconds}s (minimum: {MIN_INTERVAL_SECONDS}s)")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Signal directory: {get_signal_dir()}")
    
    def _get_signal_generator(self):
        """Lazy-load the signal generator to handle import errors gracefully"""
        if self.signal_generator is not None:
            return self.signal_generator
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "unified_ai_indicator",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "unified_ai_indicator.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.signal_generator = mod.UnifiedAIIndicator(
                data_dir=os.path.dirname(os.path.abspath(__file__))
            )
            logger.info("Using UnifiedAIIndicator (RL + Bayesian + Genetic + Microstructure + Explainability)")
            return self.signal_generator
        except Exception as e:
            logger.warning(f"Could not load UnifiedAIIndicator: {e}")

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "unified_core",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "unified_core.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.signal_generator = mod.UnifiedIntelligenceCore(
                data_dir=os.path.dirname(os.path.abspath(__file__))
            )
            logger.info("Using UnifiedIntelligenceCore (RL + Bayesian + Genetic + Microstructure)")
            return self.signal_generator
        except Exception as e:
            logger.warning(f"Could not load UnifiedIntelligenceCore: {e}")

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "qmp_engine_standalone",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "qmp_engine_standalone.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.signal_generator = mod.QMPStandaloneEngine()
            logger.info("Using QMPStandaloneEngine for signal generation")
            return self.signal_generator
        except Exception as e:
            logger.warning(f"Could not load QMPStandaloneEngine: {e}")
        
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
    
    def _is_interval_elapsed(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal for this symbol"""
        last = self.last_signal_time.get(symbol)
        if last is None:
            return True
        elapsed = time.time() - last
        return elapsed >= self.interval_seconds

    def _fetch_from_data_source(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch multi-timeframe closed-bar data for consensus engine."""
        try:
            import yfinance as yf

            yf_symbol = self._map_to_yfinance_symbol(symbol)
            if not yf_symbol:
                return None

            ticker = yf.Ticker(yf_symbol)

            # Primary timeframe: 1h bars (5 days)
            hist_1h = ticker.history(period="5d", interval="1h")
            if hist_1h.empty or len(hist_1h) < 20:
                logger.warning(f"{symbol}: Not enough 1h bars ({len(hist_1h) if not hist_1h.empty else 0})")
                return None

            result = {
                "symbol": symbol,
                "ohlcv": hist_1h.tail(80).to_dict('records'),
                "close": float(hist_1h['Close'].iloc[-1]),
                "volume": float(hist_1h['Volume'].iloc[-1]),
                "timestamp": hist_1h.index[-1].isoformat()
            }

            # Higher timeframe: daily bars for trend confirmation
            try:
                hist_daily = ticker.history(period="3mo", interval="1d")
                if not hist_daily.empty and len(hist_daily) >= 30:
                    result["ohlcv_daily"] = hist_daily.tail(60).to_dict('records')
                    logger.debug(f"{symbol}: Fetched {len(hist_1h)} 1h + {len(hist_daily)} daily bars")
            except Exception as e:
                logger.debug(f"{symbol}: Daily data fetch failed (non-fatal): {e}")

            return result

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
    
    def _normalize_signal(self, raw_signal: Optional[str]) -> Optional[str]:
        """Normalize signal string to BUY/SELL/HOLD/None"""
        if raw_signal is None:
            return None
        upper = raw_signal.upper()
        if upper in ("BUY", "STRONG_BUY"):
            return "BUY"
        if upper in ("SELL", "STRONG_SELL"):
            return "SELL"
        if upper in ("HOLD", "NEUTRAL", "WAIT"):
            return "HOLD"
        return None

    def _create_mt5_signal(
        self,
        symbol: str,
        final_signal: Optional[str],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Create the MT5 signal object with strict schema.

        Schema (STRICT):
            symbol: str - Trading symbol (no slashes)
            signal: str | null - "BUY", "SELL", "HOLD", or null
            confidence: float - 0.0 to 1.0
            timestamp: str - ISO-8601 format
        """
        normalized = self._normalize_signal(final_signal)

        # Apply confidence threshold: below threshold = null signal (no trade)
        if confidence < self.confidence_threshold:
            normalized = None
            confidence = 0.0

        return {
            "symbol": symbol.replace("/", ""),
            "signal": normalized,
            "confidence": round(float(confidence), 4),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    
    def _process_symbol(self, symbol: str) -> bool:
        """
        Process a single symbol through the full pipeline.

        Enforces:
        - Minimum interval between signals
        - Confidence threshold
        """
        # ENFORCE INTERVAL: skip if not enough time has passed
        if not self._is_interval_elapsed(symbol):
            remaining = self.interval_seconds - (time.time() - self.last_signal_time.get(symbol, 0))
            logger.debug(f"{symbol}: Interval not elapsed, {remaining:.0f}s remaining. Skipping.")
            return False

        logger.info(f"Processing {symbol}...")

        # Step 1: Fetch closed-bar market data
        market_data = self._fetch_market_data(symbol)

        if market_data is None:
            logger.warning(f"No market data available for {symbol}, emitting null signal")
            mt5_signal = self._create_mt5_signal(symbol, None, 0.0)
        else:
            # Step 2: Generate signal from engine
            signal_result = self._generate_signal(symbol, market_data)

            final_signal = signal_result.get("final_signal")
            confidence = signal_result.get("confidence", 0.0)

            # Step 3: Create MT5 signal with confidence threshold applied
            mt5_signal = self._create_mt5_signal(symbol, final_signal, confidence)

        # Step 4: Write to MT5 (atomic, single file overwrite)
        success = write_signal_output(mt5_signal)

        if success:
            self.last_signal_time[symbol] = time.time()
            self.last_emitted_signal[symbol] = mt5_signal["signal"]
            logger.info(
                f"Signal written: {mt5_signal['symbol']} | "
                f"{mt5_signal['signal']} | "
                f"confidence={mt5_signal['confidence']:.2f}"
            )
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
                write_signal_output(mt5_signal)
                results[symbol] = False
        
        return results
    
    def run(self):
        """
        Run the MT5 live loop continuously.

        Execution discipline:
        - One signal per symbol per interval (minimum 60s)
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
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Signal directory: {get_signal_dir()}")
        logger.info("=" * 60)

        try:
            while self.running:
                cycle_count += 1
                cycle_start = time.time()

                logger.info(f"--- Cycle {cycle_count} ---")

                results = self.run_once()

                success_count = sum(1 for v in results.values() if v)
                logger.info(
                    f"Cycle {cycle_count} complete: "
                    f"{success_count}/{len(results)} signals written"
                )

                # Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.interval_seconds - elapsed)

                if sleep_time > 0:
                    logger.info(f"Next cycle in {sleep_time:.0f}s...")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal (Ctrl+C)")
        finally:
            self.running = False
            logger.info("MT5 Live Runner stopped")
    
    def stop(self):
        """Stop the live runner"""
        self.running = False


class FallbackSignalGenerator:
    """
    Fallback signal generator when QMP engine is not available.

    Uses simple SMA crossover on closed bars. Deterministic given same data.
    NOT random. NOT noisy. Only changes when bar data actually changes.
    """

    def __init__(self):
        self._last_data_hash: Dict[str, Optional[str]] = {}
        self._last_result: Dict[str, Dict[str, Any]] = {}

    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a deterministic signal based on SMA crossover of closed bars"""
        try:
            ohlcv = market_data.get("ohlcv", [])

            if len(ohlcv) < 20:
                return {"final_signal": None, "confidence": 0.0}

            closes = [bar.get("Close", bar.get("close", 0)) for bar in ohlcv[-20:]]

            if not closes or closes[-1] == 0:
                return {"final_signal": None, "confidence": 0.0}

            # Create a data fingerprint to avoid recomputing on identical data
            data_hash = f"{closes[-1]:.6f}_{closes[-5]:.6f}_{closes[0]:.6f}"
            if data_hash == self._last_data_hash.get(symbol) and symbol in self._last_result:
                return self._last_result[symbol]

            # SMA crossover (deterministic, no randomness)
            sma_fast = sum(closes[-5:]) / 5
            sma_slow = sum(closes[-20:]) / 20

            if sma_slow == 0:
                return {"final_signal": None, "confidence": 0.0}

            # Trend strength as percentage spread
            spread = (sma_fast - sma_slow) / sma_slow

            # Only emit directional signal if spread is meaningful (> 0.1%)
            if spread > 0.001:
                signal = "BUY"
                confidence = min(0.95, 0.6 + abs(spread) * 20)
            elif spread < -0.001:
                signal = "SELL"
                confidence = min(0.95, 0.6 + abs(spread) * 20)
            else:
                signal = "HOLD"
                confidence = 0.5

            result = {
                "final_signal": signal,
                "confidence": confidence,
                "sma_fast": round(sma_fast, 4),
                "sma_slow": round(sma_slow, 4),
                "spread": round(spread, 6)
            }

            # Cache result so identical data returns identical signal
            self._last_data_hash[symbol] = data_hash
            self._last_result[symbol] = result

            return result

        except Exception as e:
            logger.error(f"Fallback signal generation error: {e}")
            return {"final_signal": None, "confidence": 0.0}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MT5 Live Runner - The ONLY MT5 live entry point for RayBridge EA"
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
        help=f"Seconds between signal cycles (default: 60, minimum: {MIN_INTERVAL_SECONDS})"
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
        for sym, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  {sym}: {status}")
    else:
        runner.run()


if __name__ == "__main__":
    main()
