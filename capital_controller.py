#!/usr/bin/env python3
"""
Sacred-Quant Fusion Trading System – Unified Capital Controller
Stage 1: Self-Funding Master Capital Allocator with Kelly Criterion & Reserve Vault

Allocates capital across venues (Kalshi, OKX, MT5) via fractional Kelly criterion,
capped by ZK Invariants (max 2% per trade, max 10% total exposure).
Compounds profits, maintains 20% cold reserve vault, and funds 5% self-evolution AI budget.
"""

import os
import math
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from zk_proof_verifier import ZKTradeInvariantVerifier


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [CapitalController] %(message)s",
        handlers=[logging.FileHandler("capital_vault.log"), logging.StreamHandler()],
    )


class MasterCapitalController:
    """
    Self-Owning Master Capital Allocator.
    Manages compounding account equity, Kelly position sizing, reserve vaulting,
    and self-evolution compute funding.
    """

    def __init__(self, initial_equity: float = 100000.0):
        self.logger = logging.getLogger("CapitalController")
        setup_logging()

        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.realized_pnl = 0.0

        # Allocation Buckets
        self.cold_reserve_vault = 0.0  # 20% of profits
        self.self_evolution_fund = 0.0  # 5% of profits
        self.reinvested_capital = initial_equity  # 75% compounding trading capital

        self.zk_verifier = ZKTradeInvariantVerifier(max_allowed_risk=0.02)
        self.active_allocations: Dict[str, float] = {
            "kalshi_binary": 0.0,
            "okx_crypto": 0.0,
            "mt5_fx_gold": 0.0,
        }

    def calculate_kelly_position_size(
        self, win_probability: float, payoff_ratio: float = 1.0
    ) -> float:
        """
        Fractional Kelly Criterion:
        f* = (p * b - (1 - p)) / b
        Scaled down to Quarter-Kelly (0.25) for smooth exponential compounding.
        """
        p = max(0.0, min(1.0, win_probability))
        b = max(0.1, payoff_ratio)

        kelly_f = (p * b - (1.0 - p)) / b
        fractional_kelly = max(0.0, kelly_f * 0.25)

        # Hard ZK Cap: Maximum 2% per trade
        capped_fraction = min(0.02, fractional_kelly)
        position_size = self.current_equity * capped_fraction
        return position_size

    def record_realized_profit(self, profit_amount: float):
        """Compounding & Profit Distribution Logic."""
        if profit_amount <= 0:
            self.current_equity += profit_amount
            self.realized_pnl += profit_amount
            self.logger.warning(
                "Loss Recorded: $%.2f | New Equity: $%.2f",
                profit_amount,
                self.current_equity,
            )
            return

        # Profit Slicing: 20% Vault, 5% AI Compute Fund, 75% Reinvested Trading Equity
        vault_slice = profit_amount * 0.20
        ai_fund_slice = profit_amount * 0.05
        compounding_slice = profit_amount * 0.75

        self.cold_reserve_vault += vault_slice
        self.self_evolution_fund += ai_fund_slice
        self.current_equity += compounding_slice
        self.realized_pnl += profit_amount

        self.logger.info(
            "PROFIT RECORDED: +$%.2f | Vault (+20%%): $%.2f | AI Compute (+5%%): $%.2f | Trading Equity (+75%%): $%.2f",
            profit_amount,
            self.cold_reserve_vault,
            self.self_evolution_fund,
            self.current_equity,
        )

    def allocate_order_capital(
        self, venue: str, win_prob: float, payoff_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """Authorize & Allocate Capital to Order Payload."""
        # Total Exposure Check (Max 10%)
        current_total_exposure = (
            sum(self.active_allocations.values()) / self.current_equity
        )
        if current_total_exposure >= 0.10:
            self.logger.warning(
                "Capital Allocation Denied: Total Exposure %.2f%% >= 10%% Cap",
                current_total_exposure * 100,
            )
            return {
                "authorized": False,
                "allocated_amount": 0.0,
                "reason": "MAX_EXPOSURE_CAP_REACHED",
            }

        position_size = self.calculate_kelly_position_size(win_prob, payoff_ratio)

        # Verify against ZK Risk Verifier
        signal = {
            "direction": "BUY",
            "confidence": win_prob,
            "never_loss_protected": True,
        }
        zk_valid, zk_proof = self.zk_verifier.generate_proof(
            signal, position_size=position_size, account_balance=self.current_equity
        )

        if not zk_valid:
            self.logger.error(
                "Capital Allocation Denied: ZK Invariant Risk Check Failed!"
            )
            return {
                "authorized": False,
                "allocated_amount": 0.0,
                "reason": "ZK_INVARIANT_VIOLATION",
            }

        # Reserve active allocation
        self.active_allocations[venue] = (
            self.active_allocations.get(venue, 0.0) + position_size
        )

        self.logger.info(
            "CAPITAL ALLOCATED: Venue: %s | Amount: $%.2f (Kelly Fraction: %.2f%%) | ZK-Proof Hash: %s",
            venue,
            position_size,
            (position_size / self.current_equity) * 100,
            zk_proof["commitment_hash"][:16],
        )

        return {
            "authorized": True,
            "allocated_amount": position_size,
            "venue": venue,
            "zk_proof_hash": zk_proof["commitment_hash"],
        }


if __name__ == "__main__":
    controller = MasterCapitalController(initial_equity=100000.0)

    # Test allocation
    alloc = controller.allocate_order_capital(
        "kalshi_binary", win_prob=0.85, payoff_ratio=1.2
    )
    print("Allocation Result:", alloc)

    # Simulate profit
    controller.record_realized_profit(1500.0)
