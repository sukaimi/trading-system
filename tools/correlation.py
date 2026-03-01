"""Cross-asset correlation analysis.

Phase 2 implementation. Placeholder stubs for:
- BTC-Gold 30d correlation
- Portfolio correlation matrix
- Regime detection (risk-on vs risk-off)
"""


class CorrelationAnalyzer:
    """Analyze cross-asset correlations for portfolio risk."""

    def pairwise_correlation(self, series_a: list[float], series_b: list[float]) -> float:
        """Calculate Pearson correlation between two price series."""
        raise NotImplementedError("Phase 2 — pairwise correlation")

    def portfolio_correlation_matrix(self, positions: list[dict]) -> dict:
        """Calculate correlation matrix for current portfolio."""
        raise NotImplementedError("Phase 2 — portfolio correlation matrix")

    def detect_regime(self, market_data: dict) -> str:
        """Detect current market regime (risk_on / risk_off)."""
        raise NotImplementedError("Phase 2 — regime detection")
