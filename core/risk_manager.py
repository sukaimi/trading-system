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
        self.max_exposure_ratio: float = config.get("max_exposure_ratio", 0.30)

        # Graduated correlation system
        self.correlation_hard_cap: float = config.get("correlation_hard_cap", 0.85)
        self.correlation_soft_cap: float = config.get("correlation_soft_cap", self.max_correlation)
        self.correlation_high_conviction_threshold: float = config.get("correlation_high_conviction_threshold", 0.75)
        self.correlation_max_highly_correlated: int = config.get("correlation_max_highly_correlated", 3)
        self.correlation_sector_exempt: bool = config.get("correlation_sector_exempt", True)

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

        # Check 7: Sector concentration limit (dynamic — uses asset registry for sector lookup)
        asset_sector = self._get_sector(asset)
        if asset_sector and asset_sector != "unknown":
            same_sector_count = sum(
                1 for pos in open_positions
                if self._get_sector(pos.get("asset", "")) == asset_sector
            )
            if same_sector_count >= 4:
                return (
                    False,
                    f"Sector concentration limit: {same_sector_count} {asset_sector} positions already held",
                    None,
                )

        # Check 8: Correlation limit — graduated 4-layer system
        confidence = execution_order.get("confidence", 0.5)
        corr_ok, corr_reason = self._check_correlation(asset, open_positions, confidence)
        if not corr_ok:
            return False, corr_reason, None

        # Check 9: Exposure ratio gate
        equity = portfolio_state.get("equity", 0.0)
        current_er = self._calculate_exposure_ratio(open_positions, equity)
        if current_er >= self.max_exposure_ratio:
            return (False, f"Exposure ratio {current_er:.1%} exceeds max {self.max_exposure_ratio:.0%} — reduce positions before opening new trades", None)

        log.info("Order validated: %s %s", execution_order.get("direction"), asset)
        return True, "APPROVED", execution_order

    @staticmethod
    def _get_sector(asset: str) -> str:
        """Get sector for an asset using the dynamic registry.

        Falls back to hardcoded mapping for core assets to avoid yfinance calls
        during hot-path risk checks.
        """
        # Hardcoded fallback for core assets (avoids network calls)
        _CORE_SECTORS = {
            "AAPL": "tech", "NVDA": "tech", "TSLA": "tech", "AMZN": "tech", "META": "tech",
            "SPY": "indices", "BTC": "crypto", "ETH": "crypto",
            "GLDM": "commodities", "SLV": "commodities",
            "TLT": "bonds", "XLE": "energy", "FXI": "asia", "EWS": "asia",
        }
        if asset in _CORE_SECTORS:
            return _CORE_SECTORS[asset]

        # Dynamic assets: use registry sector
        try:
            from core.asset_registry import get_registry
            registry = get_registry()
            config = registry.get_config(asset)
            sector = config.get("sector", "unknown")
            # Normalize yfinance sectors to our groupings
            sector_lower = sector.lower()
            if "technology" in sector_lower or "communication" in sector_lower:
                return "tech"
            if "energy" in sector_lower:
                return "energy"
            if "financial" in sector_lower:
                return "financials"
            if "health" in sector_lower:
                return "healthcare"
            if "consumer" in sector_lower:
                return "consumer"
            if "industrial" in sector_lower:
                return "industrials"
            if "real estate" in sector_lower:
                return "real_estate"
            if "utilit" in sector_lower:
                return "utilities"
            if "material" in sector_lower or "basic" in sector_lower:
                return "materials"
            return sector_lower if sector_lower != "unknown" else "unknown"
        except Exception:
            return "unknown"

    def _calculate_exposure_ratio(
        self, open_positions: list[dict[str, Any]], equity: float
    ) -> float:
        """Calculate total capital deployed as a ratio of equity.

        Returns:
            Ratio of total position value to equity (e.g. 0.25 = 25%).
            Returns 999.0 if equity <= 0.
        """
        if equity <= 0:
            return 999.0

        total_value = 0.0
        for pos in open_positions:
            qty = pos.get("quantity", 0)
            if not qty:
                continue
            price = pos.get("current_price", 0)
            if not price:
                price = pos.get("entry_price", 0)
            if not price:
                continue
            total_value += abs(price * qty)

        return total_value / equity

    def _check_correlation(
        self, asset: str, open_positions: list[dict[str, Any]], confidence: float = 0.5
    ) -> tuple[bool, str]:
        """Check if new asset is too correlated with existing portfolio.

        Uses a graduated 4-layer system:
        1. Hard cap (0.85): reject if ANY pairwise correlation exceeds this
        2. Sector exemption: new sector skips soft cap (still subject to hard cap)
        3. Conviction-scaled soft cap: count positions above soft cap, reject if
           count > max_highly_correlated unless confidence >= high_conviction_threshold
        4. Portfolio average warning: logging only (non-blocking)

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

            # Compute pairwise correlations with each held asset
            pairwise: list[tuple[str, float]] = []  # (held_asset, correlation)
            all_abs_correlations: list[float] = []
            for held in held_assets:
                if held == asset:
                    continue  # Skip self-correlation (add-to-position allowed by Check 5)
                held_series = price_series.get(held, [])
                if len(held_series) < 5:
                    continue
                corr = analyzer.pairwise_correlation(candidate_series, held_series)
                pairwise.append((held, corr))
                all_abs_correlations.append(abs(corr))

            # ── Layer 1: Hard cap — reject if ANY pair exceeds hard cap ──
            for held, corr in pairwise:
                if corr > self.correlation_hard_cap:
                    return (
                        False,
                        f"Correlation limit: {asset} vs {held} correlation {corr:.2f} "
                        f"exceeds hard cap {self.correlation_hard_cap}",
                    )

            # ── Layer 2: Sector exemption ────────────────────────────────
            if self.correlation_sector_exempt:
                candidate_sector = self._get_sector(asset)
                if candidate_sector and candidate_sector != "unknown":
                    portfolio_sectors = {
                        self._get_sector(pos.get("asset", ""))
                        for pos in open_positions
                        if pos.get("asset")
                    }
                    portfolio_sectors.discard("unknown")
                    portfolio_sectors.discard("")
                    if candidate_sector not in portfolio_sectors:
                        log.info(
                            "Correlation check: %s sector '%s' not in portfolio — sector exempt, approved",
                            asset, candidate_sector,
                        )
                        return True, ""

            # ── Layer 3: Conviction-scaled soft cap ──────────────────────
            highly_correlated_count = sum(
                1 for _, corr in pairwise if corr > self.correlation_soft_cap
            )
            if highly_correlated_count > self.correlation_max_highly_correlated:
                if confidence >= self.correlation_high_conviction_threshold:
                    log.info(
                        "Correlation check: %s has %d positions above soft cap %.2f, "
                        "but confidence %.2f >= %.2f — high conviction override, approved",
                        asset, highly_correlated_count, self.correlation_soft_cap,
                        confidence, self.correlation_high_conviction_threshold,
                    )
                else:
                    return (
                        False,
                        f"Correlation limit: {asset} has {highly_correlated_count} positions "
                        f"correlated above {self.correlation_soft_cap} "
                        f"(max {self.correlation_max_highly_correlated}) "
                        f"and confidence {confidence:.2f} < {self.correlation_high_conviction_threshold}",
                    )

            # ── Layer 4: Portfolio average warning (non-blocking) ────────
            if all_abs_correlations:
                avg_corr = sum(all_abs_correlations) / len(all_abs_correlations)
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
