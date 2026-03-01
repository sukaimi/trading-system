"""Cross-asset correlation analysis using numpy.

Provides pairwise Pearson correlation, portfolio correlation matrices,
and simple rule-based regime detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.logger import setup_logger

log = setup_logger("trading.correlation")


class CorrelationAnalyzer:
    """Analyze cross-asset correlations for portfolio risk."""

    def __init__(self, market_data_fetcher: Any = None):
        self._mdf = market_data_fetcher

    def pairwise_correlation(
        self, series_a: list[float], series_b: list[float]
    ) -> float:
        """Calculate Pearson correlation between two price series.

        Returns correlation coefficient [-1.0, 1.0].
        Returns 0.0 if series are too short (< 5 data points) or constant.
        """
        min_len = min(len(series_a), len(series_b))
        if min_len < 5:
            return 0.0

        a = np.array(series_a[:min_len], dtype=float)
        b = np.array(series_b[:min_len], dtype=float)

        # Guard against constant series
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        corr_matrix = np.corrcoef(a, b)
        return float(corr_matrix[0, 1])

    def portfolio_correlation_matrix(
        self, positions: list[dict]
    ) -> dict[str, Any]:
        """Calculate correlation matrix for current portfolio positions.

        Uses injected MarketDataFetcher to get 30d daily closes.

        Returns:
            {
                "matrix": {asset_a: {asset_b: corr, ...}, ...},
                "max_correlation": float,
                "high_correlation_pairs": [(asset_a, asset_b, corr), ...]
            }
        """
        assets = list({pos.get("asset", "") for pos in positions if pos.get("asset")})

        if len(assets) < 2:
            return {
                "matrix": {},
                "max_correlation": 0.0,
                "high_correlation_pairs": [],
            }

        # Fetch price series for each asset
        price_series: dict[str, list[float]] = {}
        for asset in assets:
            if self._mdf:
                try:
                    ohlcv = self._mdf.get_ohlcv(asset, period="1mo", interval="1d")
                    price_series[asset] = [bar["close"] for bar in ohlcv if "close" in bar]
                except Exception as e:
                    log.warning("Failed to fetch prices for %s: %s", asset, e)
                    price_series[asset] = []
            else:
                price_series[asset] = []

        # Build correlation matrix
        matrix: dict[str, dict[str, float]] = {}
        high_pairs: list[tuple[str, str, float]] = []
        max_corr = 0.0

        for i, a in enumerate(assets):
            matrix[a] = {}
            for j, b in enumerate(assets):
                if i == j:
                    matrix[a][b] = 1.0
                elif j < i:
                    matrix[a][b] = matrix[b][a]
                else:
                    corr = self.pairwise_correlation(
                        price_series.get(a, []),
                        price_series.get(b, []),
                    )
                    matrix[a][b] = corr
                    abs_corr = abs(corr)
                    if abs_corr > max_corr:
                        max_corr = abs_corr
                    if abs_corr > 0.5:
                        high_pairs.append((a, b, corr))

        return {
            "matrix": matrix,
            "max_correlation": max_corr,
            "high_correlation_pairs": high_pairs,
        }

    def detect_regime(self, market_data: dict[str, Any]) -> str:
        """Detect current market regime using simple rules.

        Args:
            market_data: Dict with optional keys: vix, dxy_change_7d,
                         btc_change_7d, gold_change_7d

        Returns:
            One of: "risk_on", "risk_off", "transitional", "unknown"
        """
        vix = market_data.get("vix", None)
        btc_change = market_data.get("btc_change_7d", None)
        gold_change = market_data.get("gold_change_7d", None)
        dxy_change = market_data.get("dxy_change_7d", None)

        if vix is None:
            return "unknown"

        # Risk-off: VIX high + gold rising + DXY rising (flight to safety)
        if vix > 25:
            if gold_change is not None and gold_change > 0:
                return "risk_off"
            if dxy_change is not None and dxy_change > 0:
                return "risk_off"

        # Risk-on: VIX low + crypto rising
        if vix < 20:
            if btc_change is not None and btc_change > 0:
                return "risk_on"

        # Mixed signals
        if vix is not None:
            return "transitional"

        return "unknown"
