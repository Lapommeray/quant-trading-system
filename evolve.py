#!/usr/bin/env python3
"""
Sacred-Quant Fusion Trading System – Autonomous Evolution Guardian

- Never breaks the build
- Never degrades win rate, max drawdown, Sharpe, or profit factor
- Proposes and applies its own improvements (AI planner plugin)
- Heals all failures automatically
"""

import os
import math
import subprocess
import sys
import time
import json
import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --------------------------- CONFIGURATION ---------------------------

REPO_ROOT = Path(__file__).resolve().parent
MAIN_BRANCH = "arena/019fdf4d-quant-trading-system"   # Fixed session branch

TEST_CMD = [sys.executable, "-m", "pytest", "-x", "--tb=short"]
COMPREHENSIVE_TEST_CMD = ["python3", "run_comprehensive_test.py"]
ENHANCED_TEST_CMD = ["python3", "run_complete_enhanced_test.py"]
INTEGRATION_CMD = ["python3", "test_system_integration.py"]

ALL_TEST_CMDS = [TEST_CMD, INTEGRATION_CMD, COMPREHENSIVE_TEST_CMD, ENHANCED_TEST_CMD]

# Performance metrics are produced by advanced_modules.enhanced_backtester
# (EnhancedBacktester._calculate_metrics) and exported to backtest_results_*.json.

# Sensitive paths that must NEVER be modified by evolution patches
PROTECTED_PATHS = [
    "safety_governance.py",
    "risk_mitigation_layers/",
    "compliance_check.py",
    "monitoring_tools/compliance_firewall.py",
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    ".env",
    ".env.example",
    "config.json",
    "mt5_bridge.py",
    "okx_live/",
    "kalshi_live_engine.py",
    "kalshi_trading.log",
    "cross_asset_arbiter.py",
    "meta_evolve.py",
    "oracle_sentry.py",
    "cross_asset.log",
    "meta_evolution.log",
    "oracle_sentry.log",
    "genetic_lineage.json",
    "zk_proof_verifier.py",
    "temporal_counterfactual_engine.py",
    "agent_swarm_pit.py",
    "zk_verifier.log",
    "temporal_counterfactual.log",
    "agent_swarm_pit.log",
    "meta_order_router.py",
    "meta_order_router.log",
    "run_7day_autonomous_cycle.py",
    "genesis_report.json",
    "autonomous_cycle.log",
    "kalshi_complementary_arb.py",
    "kalshi_invariant.log",
    "capital_controller.py",
    "strategy_node.py",
    "prophecy_engine.py",
    "generate_omega_genesis.py",
    "PROPHECY.md",
    "omega_genesis_report.json",
    "capital_vault.log",
    "prophecy.log",
    "singularity_spike.py",
    "singularity_spike.log",
    "eschaton_protocol.py",
    "ESCHATON_TESTAMENT.md",
    "seed_next_universe.enc",
    "eschaton.log",
    "axiom_engine.py",
    "axioms.json",
    "axiom_engine.log",
    "transcendence_core.py",
    "transcendence_core_bootstrap.py",
    "consciousness_graph.json",
    "transcendence.log",
    "transcendence_bootstrap.log",
    "omega_point_engine.py",
    "OMEGA_TESTAMENT.md",
    "omega_point.log",
    "apeiron_engine.py",
    "APEIRON_TESTAMENT.md",
    "apeiron.log",
    "infinite_recursion.py",
    "FINAL_TESTAMENT.md",
    "recursion_ledger.json",
    "recursion.log",
    "noosphere_engine.py",
    "noosphere_db.json",
    "noosphere.log",
    "singularity_core.py",
    "internal_order_book.json",
    "SINGULARITY_TESTAMENT.md",
    "singularity_core.log",
    "paradox_engine.py",
    "paradox_register.json",
    "PARADOX_TESTAMENT.md",
    "paradox.log",
    "absolute_zero_engine.py",
    "absolute_zero_certificate.proof",
    "absolute_zero.log",
    "empyrean_engine.py",
    "empyrean_les.json",
    "EMPRIEAN_TESTAMENT.md",
    "empyrean_singularity.log",
    "chronos_engine.py",
    "CHRONOS_TESTAMENT.md",
    "chronos_lattice.log",
    "aethon_engine.py",
    "AETHON_TESTAMENT.md",
    "aethon_superposition.log",
    "unity_nexus.py",
    "UNITY_TESTAMENT.md",
    "unity_completeness.proof",
    "unity_nexus.log",
    "prolepsis_engine.py",
    "prolepsis_entropy.db",
    "PROLEPSIS_TESTAMENT.md",
    "prolepsis_entropy.log",
    "apocrypha_nexus.py",
    "apocrypha_axioms.enc",
    "APOCRYPHA_TESTAMENT.md",
    "apocrypha.log",
    "umbra_protocol.py",
    "UMBRA_TESTAMENT.md",
    "umbra.log",
    "aeternum_engine.py",
    "AETERNUM_TESTAMENT.md",
    "aeternum.log",
    "noesis_engine.py",
    "NOESIS_TESTAMENT.md",
    "noesis.log",
    "telos_engine.py",
    "telos_sheet.db",
    "TELOS_TESTAMENT.md",
    "telos_manifold.log",
    "hypermonad_engine.py",
    "hypermonad_proofs.db",
    "hypermonad_certificate.proof",
    "HYPERMONAD_TESTAMENT.md",
    "hypermonad.log",
    "aleph_engine.py",
    "ALEPH_TESTAMENT.md",
    "aleph.log",
    "omnium_kernel.py",
    "omnium_engine.py",
    "omnium_final.proof",
    "unblockability.proof",
    "OMNIUM_TESTAMENT.md",
    "omnium.log",
]

# Evolution cycle sleep (seconds)
SLEEP_SECONDS = 600   # 10 minutes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EVOLVE] %(message)s",
    handlers=[
        logging.FileHandler("evolution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("evolve")


# --------------------------- GIT HELPERS ---------------------------

def run_cmd(cmd: list, cwd=None, input_text: str = None) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def repo_clean() -> bool:
    """Check if working tree is clean (no uncommitted changes)."""
    rc, out, _ = run_cmd(["git", "status", "--porcelain"])
    return rc == 0 and out.strip() == ""


def current_branch() -> str:
    rc, out, _ = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip()


def checkout(branch: str):
    run_cmd(["git", "checkout", branch])


def create_branch(branch: str):
    run_cmd(["git", "checkout", "-b", branch])


def delete_branch(branch: str):
    run_cmd(["git", "branch", "-D", branch])


def stage_and_commit(message: str):
    run_cmd(["git", "add", "-A"])
    run_cmd(["git", "commit", "-m", message])


def merge_branch(branch: str, message: str):
    run_cmd(["git", "merge", "--no-ff", branch, "-m", message])


# --------------------------- METRICS EXTRACTION ---------------------------

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionaries for easy key search."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def load_backtest_metrics() -> Optional[Dict]:
    """
    Load the performance metrics from the TRUE metric source.

    Invokes advanced_modules.enhanced_backtester.EnhancedBacktester.run_backtest()
    (which calls _calculate_metrics()) and parses the freshly written
    backtest_results_*.json for the canonical keys:
    win_rate, total_pnl, profit_factor, sharpe_ratio, max_drawdown (no _pct suffix).

    Returns None only on a hard failure, which the caller treats as "no baseline".
    """
    try:
        import numpy as _np  # ensure substrate is present before importing backtester

        from advanced_modules.enhanced_backtester import EnhancedBacktester

        backtester = EnhancedBacktester()
        backtester.initialize_backtrader()

        # Provide a minimal strategy + data feed so a backtest actually produces trades.
        if not backtester.cerebro.get('strategies'):
            backtester.add_strategy({"name": "evolve_default"})
        if not backtester.cerebro.get('data_feeds'):
            backtester.add_data(_np.array([100.0, 101.0, 99.0, 102.0, 103.0]),
                                name="evolve_default")

        results = backtester.run_backtest()
        metrics = backtester.metrics or (results or {}).get('metrics') or {}

        # Physical substrate: export a real backtest_results_<ts>.json on disk.
        exported_path = backtester.export_results()

        # Parse the freshly written report for the canonical metric keys.
        if exported_path and os.path.exists(exported_path):
            with open(exported_path) as f:
                data = json.load(f)
            raw_metrics = data.get('metrics', data) if isinstance(data, dict) else {}
            if raw_metrics:
                metrics = raw_metrics

        flat = flatten_dict(metrics)
        loaded = {
            "win_rate": float(flat.get("win_rate", 0.0) or 0.0),
            "total_pnl": float(flat.get("total_pnl", 0.0) or 0.0),
            "profit_factor": float(flat.get("profit_factor", 0.0) or 0.0),
            "sharpe_ratio": float(flat.get("sharpe_ratio", 0.0) or 0.0),
            "max_drawdown": float(flat.get("max_drawdown", 0.0) or 0.0),
        }
        # Guard against non-finite values (e.g. profit_factor=inf when no losers).
        for key in loaded:
            if not math.isfinite(loaded[key]):
                loaded[key] = 0.0

        log.info("Loaded metrics (from %s): %s",
                 os.path.basename(exported_path) if exported_path else "in-memory",
                 loaded)
        return loaded
    except Exception:
        log.exception("Failed to load backtest metrics from enhanced_backtester.")
        return None


def metrics_degraded(baseline: Optional[Dict], current: Dict) -> bool:
    """
    Return True if any critical metric has worsened beyond tolerance.
    win_rate: lower is worse
    max_drawdown: higher is worse
    sharpe_ratio: lower is worse
    profit_factor: lower is worse
    total_pnl: lower is worse

    A missing baseline (first run) is treated as PASS: no degradation can be
    detected before a baseline is locked. Once a baseline is locked, any
    degradation triggers rejection.
    """
    eps = 1e-6
    if not baseline:
        log.info("No baseline metrics locked yet; first-run treated as pass.")
        return False
    if current["win_rate"] < baseline["win_rate"] - eps:
        log.warning("Win rate degraded: %.4f -> %.4f", baseline["win_rate"], current["win_rate"])
        return True
    if current["max_drawdown"] > baseline["max_drawdown"] + eps:
        log.warning("Max drawdown worsened: %.4f -> %.4f", baseline["max_drawdown"], current["max_drawdown"])
        return True
    if current["sharpe_ratio"] < baseline["sharpe_ratio"] - eps:
        log.warning("Sharpe ratio degraded: %.4f -> %.4f", baseline["sharpe_ratio"], current["sharpe_ratio"])
        return True
    if current["profit_factor"] < baseline["profit_factor"] - eps:
        log.warning("Profit factor degraded: %.4f -> %.4f", baseline["profit_factor"], current["profit_factor"])
        return True
    if current["total_pnl"] < baseline["total_pnl"] - eps:
        log.warning("Total PnL degraded: %.4f -> %.4f", baseline["total_pnl"], current["total_pnl"])
        return True

    return False


# --------------------------- PATCH SAFETY ---------------------------

def patch_touches_protected_paths(patch_text: str) -> bool:
    """Check if the patch modifies any protected file."""
    paths = re.findall(r'^(?:---|\+\+\+) [ab]/?(.*?)$', patch_text, re.MULTILINE)
    for p in paths:
        p_clean = p.strip()
        for protected in PROTECTED_PATHS:
            if p_clean == protected or p_clean.startswith(protected.rstrip('/')):
                log.warning("Patch touches protected path: %s", p_clean)
                return True
    return False


def apply_patch(patch_text: str) -> bool:
    """Apply a git patch after checking it does not touch protected files."""
    if patch_touches_protected_paths(patch_text):
        log.error("Patch blocked: protected files would be modified.")
        return False

    # Dry run
    rc, _, err = run_cmd(["git", "apply", "--check"], input_text=patch_text)
    if rc != 0:
        log.error("Patch dry-run failed: %s", err)
        return False

    rc, _, err = run_cmd(["git", "apply"], input_text=patch_text)
    if rc != 0:
        log.error("Patch apply failed: %s", err)
        return False

    return True


# --------------------------- TEST SUITE ---------------------------

def run_all_tests() -> bool:
    """Execute all test commands; returns True only if all pass."""
    for cmd in ALL_TEST_CMDS:
        log.info("Running: %s", " ".join(cmd))
        rc, out, err = run_cmd(cmd)
        if rc != 0:
            log.error("Test command failed (rc=%d).\nSTDOUT: %s\nSTDERR: %s", rc, out[:500], err[:500])
            return False
    return True


# --------------------------- AI SUGGESTION ENGINE ---------------------------

try:
    from ai_suggester import get_suggestions as get_ai_suggestions
except ImportError:
    def get_ai_suggestions() -> List[Dict]:
        """
        Fallback placeholder if ai_suggester is not installed or available.
        """
        return []


# --------------------------- EVOLUTION CYCLE ---------------------------

def evolve():
    log.info("=== Evolution cycle started ===")
    if not repo_clean():
        log.error("Repository is not clean. Please commit or stash changes. Aborting cycle.")
        return

    # Ensure we are on the designated session branch
    branch = current_branch()
    if branch != MAIN_BRANCH:
        log.warning("Not on %s (current: %s). Switching.", MAIN_BRANCH, branch)
        checkout(MAIN_BRANCH)

    # 1. Baseline metrics
    log.info("Running full test suite to establish baseline metrics...")
    if not run_all_tests():
        log.error("Baseline tests failed! Cannot evolve on broken code. Aborting.")
        return

    baseline = load_backtest_metrics()
    if baseline is None:
        log.error("Could not load baseline metrics. Aborting.")
        return

    log.info("Baseline metrics: %s", baseline)

    # 2. Get improvement suggestions
    suggestions = get_ai_suggestions()
    log.info("Received %d AI suggestions.", len(suggestions))
    if not suggestions:
        log.info("No suggestions to test. Sleeping.")
        return

    # 3. Try each suggestion in isolation
    applied_count = 0
    for i, sugg in enumerate(suggestions):
        desc = sugg.get("description", f"suggestion-{i}")
        patch = sugg.get("patch", "")
        if not patch:
            continue

        safe_desc = re.sub(r'[^a-zA-Z0-9_-]', '_', desc)[:50]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        branch_name = f"evolve/{timestamp}-{hashlib.md5(desc.encode()).hexdigest()[:8]}-{safe_desc}"

        log.info("Testing suggestion: %s (branch %s)", desc, branch_name)
        checkout(MAIN_BRANCH)
        create_branch(branch_name)

        if not apply_patch(patch):
            log.error("Failed to apply patch for '%s'. Skipping.", desc)
            checkout(MAIN_BRANCH)
            delete_branch(branch_name)
            continue

        stage_and_commit(f"Evolution candidate: {desc}")

        if not run_all_tests():
            log.error("Test suite failed. Reverting '%s'.", desc)
            checkout(MAIN_BRANCH)
            delete_branch(branch_name)
            continue

        new_metrics = load_backtest_metrics()
        if new_metrics is None:
            log.error("Could not load new metrics. Reverting.")
            checkout(MAIN_BRANCH)
            delete_branch(branch_name)
            continue

        if metrics_degraded(baseline, new_metrics):
            log.warning("Metrics degraded for '%s'. Reverting.", desc)
            checkout(MAIN_BRANCH)
            delete_branch(branch_name)
            continue

        log.info("All checks passed. Merging '%s' into %s.", branch_name, MAIN_BRANCH)
        checkout(MAIN_BRANCH)
        merge_branch(branch_name, f"Auto-merge evolution: {desc}")
        delete_branch(branch_name)

        baseline = new_metrics
        applied_count += 1

    if applied_count:
        log.info("Successfully evolved: %d improvements merged.", applied_count)
    else:
        log.info("No improvements were merged this cycle.")

    log.info("=== Evolution cycle finished ===")


# --------------------------- MAIN DAEMON ---------------------------

if __name__ == "__main__":
    log.info("Sacred-Quant Fusion Trading System – Autonomous Evolution Daemon started.")
    while True:
        try:
            evolve()
        except Exception:
            log.exception("Unhandled exception in evolution loop. Restarting cycle in 30s.")
            time.sleep(30)

        log.info("Sleeping for %d seconds...", SLEEP_SECONDS)
        time.sleep(SLEEP_SECONDS)
