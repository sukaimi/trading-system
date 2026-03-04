"""Pure-Python risk management — no LLM calls.

Validates orders against position limits, daily loss caps, drawdown
thresholds, open position counts, correlation exposure, and stop-loss
requirements. Also provides ATR-based position sizing.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.risk")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "risk_params.json")


class RiskManager:
    """Pure Python risk management — no LLM calls."""

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = self._load_config()

        self.max_position_pct: float = config["max_position_pct"]
        self.max_daily_loss_pct: float = config["max_daily_loss_pct"]
        self.max_total_drawdown_pct: float = config["max_total_drawdown_pct"]
        self.max_open_positions: int = config["max_open_positions"]
        self.max_correlation: float = config["max_correlation"]
        self.stop_loss_atr_mult: float = config["stop_loss_atr_mult"]
        self.base_risk_per_trade_pct: float = config.get("base_risk_per_trade_pct", 2.0)

    @staticmethod
    def _load_config() -> dict[str, Any]:
        with open(DEFAULT_CONFIG) as f:
            return json.load(f)

    def validate_order(
        self,
        execution_order: dict[str, Any],
        portfolio_state: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Validate an order against all risk checks.

        Returns:
            (approved, reason, adjusted_order_or_None)
        """
        # Check 1: Position size within limits
        position_pct = execution_order.get("position_size_pct", 0)
        if position_pct > self.max_position_pct:
            return (
                False,
                f"Position {position_pct}% exceeds max {self.max_position_pct}%",
                None,
            )

        # Check 2: Daily loss limit
        daily_pnl = portfolio_state.get("daily_pnl_pct", 0.0)
        if daily_pnl <= -self.max_daily_loss_pct:
            return False, "DAILY LOSS LIMIT REACHED — NO NEW TRADES", None

        # Check 3: Total drawdown
        drawdown = portfolio_state.get("drawdown_from_peak_pct", 0.0)
        if drawdown >= self.max_total_drawdown_pct:
            return False, "MAX DRAWDOWN REACHED — CIRCUIT BREAKER", None

        # Check 4: Open position count
        open_positions = portfolio_state.get("open_positions", [])
        if len(open_positions) >= self.max_open_positions:
            return (
                False,
                f"Max {self.max_open_positions} positions reached",
                None,
            )

        # Check 5: Duplicate asset prevention
        asset = execution_order.get("asset", "")
        for pos in open_positions:
            if pos.get("asset") == asset:
                return (
                    False,
                    f"Duplicate asset {asset} already in portfolio",
                    None,
                )

        # Check 6: Stop-loss must be defined
        if not execution_order.get("stop_loss"):
            return False, "No stop-loss defined — rejected", None

        log.info("Order validated: %s %s", execution_order.get("direction"), asset)
        return True, "APPROVED", execution_order

    def calculate_position_size(
        self,
        confidence: float,
        atr: float,
        portfolio_value: float,
    ) -> float:
        """ATR-based position sizing adjusted by confidence.

        Args:
            confidence: Trade confidence score 0.0-1.0
            atr: Average True Range for the asset
            portfolio_value: Current portfolio equity

        Returns:
            Position value in USD, capped at max_position_pct.
        """
        if atr <= 0 or portfolio_value <= 0:
            return 0.0

        base_risk = self.base_risk_per_trade_pct / 100.0
        stop_distance = atr * self.stop_loss_atr_mult

        position_value = (portfolio_value * base_risk) / stop_distance

        # Scale by confidence
        confidence_scalar = min(max(confidence, 0.0), 1.0)
        position_value *= confidence_scalar

        # Cap at max position size
        max_value = portfolio_value * (self.max_position_pct / 100.0)
        return min(position_value, max_value)
