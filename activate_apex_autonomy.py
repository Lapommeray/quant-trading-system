#!/usr/bin/env python3
"""
Apex Autonomy Activation Script

Activates the unsupervised eternal evolution mode for the
perpetual ascension trading system.

This script:
1. Configures environment for autonomous operation
2. Initializes the self-evolution agent
3. Starts the perpetual daemon
4. Enables unsupervised mode after validation

Usage:
    python activate_apex_autonomy.py [--paper|--live] [--interval HOURS]
    
Environment Variables:
    LLM_API_KEY: API key for LLM (OpenAI/Grok/Claude)
    LIVE_TRADING: Set to "TRUE" for live trading (requires HUMAN_OVERRIDE)
    HUMAN_OVERRIDE: Set to "TRUE" to enable live trading
    UNSUPERVISED_MODE: Set to "TRUE" to remove human confirmation after validation
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("apex_autonomy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ApexAutonomy")


def configure_environment(mode: str = "paper", llm_api_key: str = None):
    """Configure environment for autonomous operation"""
    
    if llm_api_key:
        os.environ["LLM_API_KEY"] = llm_api_key
    elif not os.getenv("LLM_API_KEY"):
        logger.warning("LLM_API_KEY not set. Using template-based responses.")
        
    if mode == "paper":
        os.environ["LIVE_TRADING"] = "PAPER"
        os.environ["HUMAN_OVERRIDE"] = ""
        logger.info("Configured for PAPER trading mode")
    elif mode == "live":
        os.environ["LIVE_TRADING"] = "TRUE"
        if not os.getenv("HUMAN_OVERRIDE"):
            logger.error("HUMAN_OVERRIDE required for live trading!")
            logger.error("Set HUMAN_OVERRIDE=TRUE to enable live trading")
            sys.exit(1)
        logger.warning("Configured for LIVE trading mode - USE WITH CAUTION")
        
    os.environ["UNSUPERVISED_MODE"] = "TRUE"
    logger.info("Unsupervised mode enabled")
    
    return {
        "mode": mode,
        "llm_configured": bool(os.getenv("LLM_API_KEY")),
        "live_trading": os.getenv("LIVE_TRADING"),
        "human_override": bool(os.getenv("HUMAN_OVERRIDE")),
        "unsupervised": True
    }


def validate_system():
    """Validate system components before activation"""
    logger.info("Validating system components...")
    
    required_modules = [
        "self_evolution_agent",
        "safety_governance",
        "mt5_live_engine",
        "advanced_modules.feature_evolution",
        "advanced_modules.loss_prevention_core",
        "advanced_modules.regime_detection",
        "risk.institutional_risk_manager"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"  [OK] {module}")
        except ImportError as e:
            logger.error(f"  [FAIL] {module}: {e}")
            missing.append(module)
            
    if missing:
        logger.error(f"Missing modules: {missing}")
        return False
        
    logger.info("All system components validated")
    return True


def run_initial_validation_cycle(agent):
    """Run initial validation cycle before full autonomy"""
    logger.info("Running initial validation cycle...")
    
    try:
        result = agent.run_single_cycle_demo()
        
        logger.info(f"Validation cycle complete:")
        logger.info(f"  Tasks processed: {result['cycle_result']['tasks_processed']}")
        logger.info(f"  Duration: {result['cycle_result']['duration']:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation cycle failed: {e}")
        return False


def activate_apex_autonomy(mode: str = "paper", 
                           interval_hours: float = 24,
                           llm_api_key: str = None,
                           skip_validation: bool = False):
    """
    Activate apex autonomy mode.
    
    Args:
        mode: Trading mode ("paper" or "live")
        interval_hours: Daemon cycle interval
        llm_api_key: Optional LLM API key
        skip_validation: Skip initial validation cycle
    """
    print("=" * 70)
    print("APEX AUTONOMY ACTIVATION")
    print("Eternal Self-Evolving Trading System")
    print("=" * 70)
    print()
    
    config = configure_environment(mode, llm_api_key)
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print()
    
    if not validate_system():
        logger.error("System validation failed. Aborting activation.")
        sys.exit(1)
    print()
    
    from self_evolution_agent import SelfEvolutionAgent
    
    logger.info("Initializing Self-Evolution Agent...")
    agent = SelfEvolutionAgent(auto_apply=True)
    
    if not skip_validation:
        if not run_initial_validation_cycle(agent):
            logger.error("Initial validation failed. Aborting activation.")
            sys.exit(1)
        print()
        
    print("=" * 70)
    print("PERPETUAL ASCENSION ENGINE ACTIVATED")
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {interval_hours} hours")
    print(f"Unsupervised: TRUE")
    print("=" * 70)
    print()
    print("The system will now evolve autonomously.")
    print("Monitor logs at: self_evolution.log, apex_autonomy.log")
    print("Evolution log: evolution_log.json")
    print()
    print("Press Ctrl+C to stop the daemon.")
    print()
    
    log_entry = {
        "event": "apex_autonomy_activated",
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "interval_hours": interval_hours,
        "config": config
    }
    
    log_file = Path("apex_autonomy_history.json")
    history = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            history = json.load(f)
    history.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=2)
        
    try:
        agent.start_perpetual_daemon(interval_hours=interval_hours)
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
        agent.stop_daemon()
        
    print()
    print("=" * 70)
    print("APEX AUTONOMY DEACTIVATED")
    print("=" * 70)
    
    final_status = agent.get_status()
    print(f"Final Status: {json.dumps(final_status, indent=2, default=str)}")


def main():
    parser = argparse.ArgumentParser(
        description="Activate Apex Autonomy - Eternal Self-Evolution Mode"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=24,
        help="Daemon cycle interval in hours (default: 24)"
    )
    
    parser.add_argument(
        "--llm-key",
        type=str,
        default=None,
        help="LLM API key (or set LLM_API_KEY env var)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip initial validation cycle"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current system status and exit"
    )
    
    args = parser.parse_args()
    
    if args.status:
        from self_evolution_agent import SelfEvolutionAgent
        agent = SelfEvolutionAgent()
        status = agent.get_status()
        print("Current System Status:")
        print(json.dumps(status, indent=2, default=str))
        return
        
    activate_apex_autonomy(
        mode=args.mode,
        interval_hours=args.interval,
        llm_api_key=args.llm_key,
        skip_validation=args.skip_validation
    )


if __name__ == "__main__":
    main()
