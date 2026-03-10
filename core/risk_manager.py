"""Pure-Python risk management — no LLM calls.

Validates orders against position limits, daily loss caps, drawdown
thresholds, open position counts, correlation exposure, and stop-loss
requirements. Also provides ATR-based position sizing.
"""

from __future__ import annotations

import json
import os
import time
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
        self.max_portfolio_avg_correlation: float = config.get("max_portfolio_avg_correlation", 0.6)

        # Correlation cache: {frozenset(assets): {"matrix": ..., "timestamp": ...}}
        self._correlation_cache: dict[str, Any] = {}
        self._correlation_ttl: int = 3600  # 1 hour

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

        # Check 5: Duplicate asset — log warning but allow through
        # The Devil's Advocate already flags this as a soft challenge.
        # Blocking here was causing 86% kill rate on add-to-position signals.
        asset = execution_order.get("asset", "")
        for pos in open_positions:
            if pos.get("asset") == asset:
                log.warning(
                    "Duplicate asset %s already in portfolio — allowing through (DA handles sizing)",
                    asset,
                )

        # Check 6: Stop-loss must be defined
        if not execution_order.get("stop_loss"):
            return False, "No stop-loss defined — rejected", None

        # Check 7: Sector concentration limit
        sector_groups = {
            "tech": ["AAPL", "NVDA", "TSLA", "AMZN", "META"],
            "indices": ["SPY"],
            "crypto": ["BTC", "ETH"],
            "commodities": ["GLDM", "SLV"],
            "bonds": ["TLT"],
            "energy": ["XLE"],
            "asia": ["FXI", "EWS"],
        }
        asset_sector = None
        for sector, members in sector_groups.items():
            if asset in members:
                asset_sector = sector
                break
        if asset_sector:
            same_sector_count = sum(
                1 for pos in open_positions
                if pos.get("asset") in sector_groups[asset_sector]
            )
            if same_sector_count >= 4:
                return (
                    False,
                    f"Sector concentration limit: {same_sector_count} {asset_sector} positions already held",
                    None,
                )

        # Check 8: Correlation limit — reject if too correlated with existing positions
        corr_ok, corr_reason = self._check_correlation(asset, open_positions)
        if not corr_ok:
            return False, corr_reason, None

        log.info("Order validated: %s %s", execution_order.get("direction"), asset)
        return True, "APPROVED", execution_order

    def _check_correlation(
        self, asset: str, open_positions: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Check if new asset is too correlated with existing portfolio.

        Uses cached correlation data (1-hour TTL) to avoid re-fetching prices
        on every validate_order call. Gracefully approves if price data is
        unavailable.

        Returns:
            (ok, reason) — ok=True means the order passes the correlation check.
        """
        if not open_positions:
            return True, ""

        held_assets = [pos.get("asset", "") for pos in open_positions if pos.get("asset")]
        if not held_assets:
            return True, ""

        try:
            from tools.correlation import CorrelationAnalyzer
            from tools.market_data import MarketDataFetcher

            # Build cache key from sorted asset list (candidate + held)
            all_assets = sorted(set(held_assets + [asset]))
            cache_key = ",".join(all_assets)

            now = time.time()
            cached = self._correlation_cache.get(cache_key)
            price_series: dict[str, list[float]] = {}

            if cached and (now - cached["timestamp"]) < self._correlation_ttl:
                price_series = cached["series"]
            else:
                # Fetch fresh price data
                mdf = MarketDataFetcher()
                for sym in all_assets:
                    try:
                        ohlcv = mdf.get_ohlcv(sym, period="1mo", interval="1d")
                        price_series[sym] = [bar["close"] for bar in ohlcv if "close" in bar]
                    except Exception as e:
                        log.warning("Correlation price fetch failed for %s: %s", sym, e)
                        price_series[sym] = []

                self._correlation_cache[cache_key] = {
                    "series": price_series,
                    "timestamp": now,
                }

            analyzer = CorrelationAnalyzer()
            candidate_series = price_series.get(asset, [])

            if len(candidate_series) < 5:
                # Not enough data — approve by default
                log.info("Correlation check: insufficient data for %s — approved", asset)
                return True, ""

            # Check pairwise correlation with each held asset
            all_correlations: list[float] = []
            for held in held_assets:
                if held == asset:
                    continue  # Skip self-correlation (add-to-position allowed by Check 5)
                held_series = price_series.get(held, [])
                if len(held_series) < 5:
                    continue
                corr = analyzer.pairwise_correlation(candidate_series, held_series)
                abs_corr = abs(corr)
                all_correlations.append(abs_corr)

                if corr > self.max_correlation:
                    return (
                        False,
                        f"Correlation limit: {asset} vs {held} correlation {corr:.2f} "
                        f"exceeds max {self.max_correlation}",
                    )

            # Portfolio-level average correlation warning (don't reject)
            if all_correlations:
                avg_corr = sum(all_correlations) / len(all_correlations)
                if avg_corr > self.max_portfolio_avg_correlation:
                    log.warning(
                        "CORRELATION WARNING: portfolio avg correlation with %s is %.2f "
                        "(threshold %.2f) — order approved but elevated risk",
                        asset, avg_corr, self.max_portfolio_avg_correlation,
                    )

        except ImportError:
            log.warning("Correlation check skipped — tools not available")
        except Exception as e:
            log.warning("Correlation check failed — approving by default: %s", e)

        return True, ""

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
