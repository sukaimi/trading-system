"""Adaptive Stop Optimizer — pure Python, no LLM calls (Tier 0).

Analyzes historical MAE/MFE data from closed trades to recommend
optimal stop-loss and take-profit distances per asset.

Reads from data/trade_journal.json, outputs to data/stop_recommendations.json
for the SelfOptimizer to consume during weekly reviews.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.adaptive_stops")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
JOURNAL_FILE = os.path.join(DATA_DIR, "trade_journal.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "stop_recommendations.json")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
RISK_PARAMS_FILE = os.path.join(CONFIG_DIR, "risk_params.json")


class AdaptiveStopOptimizer:
    """Analyze closed trades to recommend stop-loss and take-profit levels."""

    def __init__(self, min_trades: int = 10):
        self._min_trades = min_trades

    def _load_journal(self) -> list[dict[str, Any]]:
        """Load closed trades from trade journal."""
        if not os.path.exists(JOURNAL_FILE):
            return []
        try:
            with open(JOURNAL_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load trade journal: %s", e)
        return []

    def _load_risk_params(self) -> dict[str, Any]:
        """Load current risk params for comparison."""
        if not os.path.exists(RISK_PARAMS_FILE):
            return {}
        try:
            with open(RISK_PARAMS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _is_closed_trade(self, trade: dict[str, Any]) -> bool:
        """Check if a trade entry is a closed trade with MAE/MFE data."""
        return (
            trade.get("pnl_pct") is not None
            and trade.get("mae_pct") is not None
            and trade.get("mfe_pct") is not None
            and trade.get("exit_reason") is not None
        )

    def _is_winner(self, trade: dict[str, Any]) -> bool:
        """Check if trade was profitable."""
        pnl = trade.get("pnl_pct", 0)
        return pnl is not None and pnl > 0

    def analyze(self) -> dict[str, Any]:
        """Analyze closed trades and return stop/TP recommendations.

        Returns a dict with sufficient_data flag, per-asset recommendations,
        and portfolio-level aggregations.
        """
        journal = self._load_journal()
        closed_trades = [t for t in journal if self._is_closed_trade(t)]
        total_closed = len(closed_trades)

        result: dict[str, Any] = {
            "sufficient_data": total_closed >= self._min_trades,
            "total_closed_trades": total_closed,
            "per_asset": {},
            "portfolio_level": {},
        }

        if total_closed == 0:
            return result

        risk_params = self._load_risk_params()
        current_stop = risk_params.get("default_stop_loss_pct", 3.0)
        current_tp = risk_params.get("default_take_profit_pct", 5.0)

        # Group trades by asset
        by_asset: dict[str, list[dict[str, Any]]] = {}
        for trade in closed_trades:
            asset = trade.get("asset", "UNKNOWN")
            by_asset.setdefault(asset, []).append(trade)

        all_recommended_stops: list[float] = []
        all_recommended_tps: list[float] = []

        for asset, trades in by_asset.items():
            asset_result = self._analyze_asset(
                trades, current_stop, current_tp
            )
            result["per_asset"][asset] = asset_result

            if asset_result.get("recommended_stop_pct") is not None:
                all_recommended_stops.append(asset_result["recommended_stop_pct"])
            if asset_result.get("recommended_tp_pct") is not None:
                all_recommended_tps.append(asset_result["recommended_tp_pct"])

        # Portfolio-level aggregation
        if all_recommended_stops:
            result["portfolio_level"]["recommended_stop_pct"] = round(
                statistics.mean(all_recommended_stops), 2
            )
        if all_recommended_tps:
            result["portfolio_level"]["recommended_tp_pct"] = round(
                statistics.mean(all_recommended_tps), 2
            )

        return result

    def _analyze_asset(
        self,
        trades: list[dict[str, Any]],
        current_stop: float,
        current_tp: float,
    ) -> dict[str, Any]:
        """Analyze trades for a single asset."""
        winners = [t for t in trades if self._is_winner(t)]
        losers = [t for t in trades if not self._is_winner(t)]

        asset_result: dict[str, Any] = {
            "trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "current_stop_pct": current_stop,
            "current_tp_pct": current_tp,
        }

        # MAE is typically negative (max adverse excursion = worst drawdown)
        # We work with absolute values for recommendations
        if winners:
            winner_maes = [abs(t["mae_pct"]) for t in winners]
            winner_mfes = [t["mfe_pct"] for t in winners]

            asset_result["median_winner_mae_pct"] = round(
                -statistics.median(winner_maes), 2
            )
            asset_result["p90_winner_mae_pct"] = round(
                -_percentile(winner_maes, 90), 2
            )
            asset_result["avg_winner_mfe_pct"] = round(
                statistics.mean(winner_mfes), 2
            )

            # Recommended stop: accommodate 90th percentile MAE of winners
            # so we don't get stopped out of trades that would have been winners
            recommended_stop = round(_percentile(winner_maes, 90), 2)
            # Floor at 1% to avoid absurdly tight stops
            recommended_stop = max(recommended_stop, 1.0)
            asset_result["recommended_stop_pct"] = recommended_stop

            # Recommended TP: average MFE of winners
            recommended_tp = round(statistics.mean(winner_mfes), 2)
            # Floor at 2% to avoid closing too early
            recommended_tp = max(recommended_tp, 2.0)
            asset_result["recommended_tp_pct"] = recommended_tp
        else:
            # All losers — widen stops slightly from current
            asset_result["median_winner_mae_pct"] = None
            asset_result["p90_winner_mae_pct"] = None
            asset_result["avg_winner_mfe_pct"] = None
            asset_result["recommended_stop_pct"] = round(current_stop * 1.2, 2)
            asset_result["recommended_tp_pct"] = current_tp

        if losers:
            loser_mfes = [t["mfe_pct"] for t in losers]
            asset_result["median_loser_mfe_pct"] = round(
                statistics.median(loser_mfes), 2
            )
        else:
            asset_result["median_loser_mfe_pct"] = None

        return asset_result

    def persist_recommendations(self) -> None:
        """Save recommendations to JSON for SelfOptimizer to consume."""
        recommendations = self.analyze()
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(RECOMMENDATIONS_FILE, "w") as f:
                json.dump(recommendations, f, indent=2)
            log.info(
                "Stop recommendations saved (%d trades analyzed)",
                recommendations["total_closed_trades"],
            )
        except OSError as e:
            log.error("Failed to save stop recommendations: %s", e)


def _percentile(data: list[float], pct: int) -> float:
    """Compute the pct-th percentile of a list of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (pct / 100.0) * (len(sorted_data) - 1)
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(sorted_data) - 1)
    frac = k - floor_k
    return sorted_data[floor_k] + frac * (sorted_data[ceil_k] - sorted_data[floor_k])
