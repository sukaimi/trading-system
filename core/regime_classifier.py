"""Market regime classifier — pure Python, no LLM calls (Tier 0).

Classifies each asset into one of five regimes using multiple technical signals:
ADX, ATR ratio, Bollinger Bandwidth, SMA slope, and RSI.

Replaces the basic CorrelationAnalyzer.detect_regime() with a proper
multi-signal classifier that drives adaptive stop-loss behavior.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any

from core.logger import setup_logger
from tools.technical_indicators import TechnicalIndicators

log = setup_logger("trading.regime")


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


# Default trailing stop multipliers per regime
DEFAULT_REGIME_TRAILING_MULT: dict[str, float] = {
    "TRENDING_UP": 0.8,
    "TRENDING_DOWN": 0.8,
    "RANGING": 1.5,
    "HIGH_VOLATILITY": 2.0,
    "LOW_VOLATILITY": 1.0,
}

# Default initial stop multipliers per regime
DEFAULT_REGIME_INITIAL_STOP_MULT: dict[str, float] = {
    "TRENDING_UP": 0.9,
    "TRENDING_DOWN": 0.9,
    "RANGING": 1.0,
    "HIGH_VOLATILITY": 1.3,
    "LOW_VOLATILITY": 1.0,
}


class RegimeClassifier:
    """Classify market regime per asset using technical signals."""

    def __init__(self, market_data_fetcher: Any = None):
        self._mdf = market_data_fetcher
        self._ti = TechnicalIndicators()

    def classify(self, asset: str) -> dict[str, Any]:
        """Classify regime for a single asset.

        Fetches 60d daily OHLCV data and runs multiple technical signals
        to determine the current market regime.

        Returns:
            {
                "regime": "TRENDING_UP",
                "confidence": 0.8,
                "adx": 32.5,
                "atr_ratio": 1.1,
                "bollinger_bandwidth": 0.05,
                "sma_slope": 0.002,
                "rsi": 62.0,
            }
        """
        result: dict[str, Any] = {
            "regime": MarketRegime.RANGING.value,
            "confidence": 0.5,
            "adx": 0.0,
            "atr_ratio": 1.0,
            "bollinger_bandwidth": 0.0,
            "sma_slope": 0.0,
            "rsi": 50.0,
        }

        if not self._mdf:
            return result

        # Fetch 60 days of daily OHLCV data
        try:
            ohlcv = self._mdf.get_ohlcv(asset, period="3mo", interval="1d")
        except Exception as e:
            log.warning("Regime classifier: failed to fetch OHLCV for %s: %s", asset, e)
            return result

        if not ohlcv or len(ohlcv) < 30:
            log.debug("Regime classifier: insufficient data for %s (%d bars)", asset, len(ohlcv) if ohlcv else 0)
            return result

        return self.classify_from_ohlcv(ohlcv)

    def classify_from_ohlcv(self, ohlcv: list[dict[str, Any]]) -> dict[str, Any]:
        """Classify regime from pre-fetched OHLCV data.

        Useful for testing or when OHLCV data is already available.

        Args:
            ohlcv: List of {"open", "high", "low", "close", "volume"} dicts, oldest first.

        Returns:
            Same dict format as classify().
        """
        result: dict[str, Any] = {
            "regime": MarketRegime.RANGING.value,
            "confidence": 0.5,
            "adx": 0.0,
            "atr_ratio": 1.0,
            "bollinger_bandwidth": 0.0,
            "sma_slope": 0.0,
            "rsi": 50.0,
        }

        if not ohlcv or len(ohlcv) < 30:
            return result

        highs = [bar["high"] for bar in ohlcv]
        lows = [bar["low"] for bar in ohlcv]
        closes = [bar["close"] for bar in ohlcv]

        # 1. ADX — trend strength
        adx_val = self._ti.adx(highs, lows, closes, period=14)
        result["adx"] = round(adx_val, 2)

        # 2. ATR ratio — current volatility vs 20-day SMA of ATR
        atr_val = self._ti.atr(highs, lows, closes, period=14)
        if len(closes) >= 35:
            # Calculate ATR for rolling windows to get SMA of ATR
            atr_series = self._compute_atr_series(highs, lows, closes, atr_period=14, window=20)
            if atr_series and atr_series > 0:
                atr_ratio = atr_val / atr_series
            else:
                atr_ratio = 1.0
        else:
            atr_ratio = 1.0
        result["atr_ratio"] = round(atr_ratio, 2)

        # 3. Bollinger Bandwidth
        bb = self._ti.bollinger_bands(closes, period=20, std=2.0)
        result["bollinger_bandwidth"] = round(bb["bandwidth"], 4)

        # 4. SMA slope — 20-day SMA direction
        sma_vals = self._ti.sma_series(closes, 20)
        if len(sma_vals) >= 2:
            # Normalized slope: change per bar as fraction of SMA value
            sma_slope = (sma_vals[-1] - sma_vals[-2]) / sma_vals[-1] if sma_vals[-1] != 0 else 0
        else:
            sma_slope = 0.0
        result["sma_slope"] = round(sma_slope, 6)

        # 5. RSI
        rsi_val = self._ti.rsi(closes, period=14)
        result["rsi"] = round(rsi_val, 2)

        # ── Classification logic ──────────────────────────────────────
        regime, confidence = self._determine_regime(
            adx=adx_val,
            atr_ratio=atr_ratio,
            bandwidth=bb["bandwidth"],
            sma_slope=sma_slope,
            rsi=rsi_val,
        )
        result["regime"] = regime.value
        result["confidence"] = round(confidence, 2)

        return result

    def classify_portfolio(self, assets: list[str]) -> dict[str, Any]:
        """Classify regime for multiple assets and compute portfolio-level summary.

        Returns:
            {
                "per_asset": {"BTC": {...}, "ETH": {...}, ...},
                "dominant_regime": "TRENDING_UP",
                "regime_agreement": 0.7,
            }
        """
        per_asset: dict[str, dict[str, Any]] = {}
        for asset in assets:
            per_asset[asset] = self.classify(asset)

        # Determine dominant regime
        if not per_asset:
            return {
                "per_asset": {},
                "dominant_regime": MarketRegime.RANGING.value,
                "regime_agreement": 0.0,
            }

        regimes = [info["regime"] for info in per_asset.values()]
        regime_counts = Counter(regimes)
        dominant_regime = regime_counts.most_common(1)[0][0]
        regime_agreement = regime_counts[dominant_regime] / len(regimes)

        return {
            "per_asset": per_asset,
            "dominant_regime": dominant_regime,
            "regime_agreement": round(regime_agreement, 2),
        }

    def get_trailing_multiplier(
        self, regime: str, config: dict[str, Any] | None = None
    ) -> float:
        """Get trailing stop distance multiplier for a given regime.

        Args:
            regime: MarketRegime value string.
            config: Optional risk_params dict with "regime_trailing_mult" key.

        Returns:
            Multiplier float (e.g., 0.8 for trending, 2.0 for high vol).
        """
        mult_map = DEFAULT_REGIME_TRAILING_MULT
        if config and "regime_trailing_mult" in config:
            mult_map = config["regime_trailing_mult"]
        return mult_map.get(regime, 1.0)

    def get_initial_stop_multiplier(
        self, regime: str, config: dict[str, Any] | None = None
    ) -> float:
        """Get initial stop-loss multiplier for a given regime.

        Args:
            regime: MarketRegime value string.
            config: Optional risk_params dict with "regime_initial_stop_mult" key.

        Returns:
            Multiplier float (e.g., 0.9 for trending, 1.3 for high vol).
        """
        mult_map = DEFAULT_REGIME_INITIAL_STOP_MULT
        if config and "regime_initial_stop_mult" in config:
            mult_map = config["regime_initial_stop_mult"]
        return mult_map.get(regime, 1.0)

    # ── Internal helpers ──────────────────────────────────────────────

    def _compute_atr_series(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        atr_period: int = 14,
        window: int = 20,
    ) -> float:
        """Compute the SMA of recent ATR values to normalize current ATR.

        Instead of computing ATR at many points, we approximate by computing
        the ATR over different sub-windows and averaging.
        """
        n = len(closes)
        if n < atr_period + window + 1:
            return 0.0

        atr_values = []
        for end_idx in range(n - window, n):
            start_idx = max(0, end_idx - atr_period - 1)
            sub_h = highs[start_idx : end_idx + 1]
            sub_l = lows[start_idx : end_idx + 1]
            sub_c = closes[start_idx : end_idx + 1]
            atr_val = self._ti.atr(sub_h, sub_l, sub_c, period=atr_period)
            if atr_val > 0:
                atr_values.append(atr_val)

        if not atr_values:
            return 0.0

        return sum(atr_values) / len(atr_values)

    def _determine_regime(
        self,
        adx: float,
        atr_ratio: float,
        bandwidth: float,
        sma_slope: float,
        rsi: float,
    ) -> tuple[MarketRegime, float]:
        """Determine regime from technical signals with confidence score.

        Signal voting system:
        - Each signal casts a vote for a regime
        - The regime with most votes wins
        - Confidence = proportion of agreeing signals

        Returns:
            (MarketRegime, confidence)
        """
        votes: list[MarketRegime] = []

        # ── ADX vote ──
        if adx > 25:
            # Strong trend — direction from SMA slope
            if sma_slope > 0:
                votes.append(MarketRegime.TRENDING_UP)
            else:
                votes.append(MarketRegime.TRENDING_DOWN)
        elif adx < 20:
            votes.append(MarketRegime.RANGING)
        # ADX 20-25: ambiguous, no vote

        # ── ATR ratio vote ──
        if atr_ratio > 1.5:
            votes.append(MarketRegime.HIGH_VOLATILITY)
        elif atr_ratio < 0.7:
            votes.append(MarketRegime.LOW_VOLATILITY)
        # 0.7-1.5: normal, no vote

        # ── Bollinger Bandwidth vote ──
        if bandwidth > 0.10:
            votes.append(MarketRegime.HIGH_VOLATILITY)
        elif bandwidth < 0.03:
            votes.append(MarketRegime.LOW_VOLATILITY)
        # 0.03-0.10: normal, no vote

        # ── SMA slope vote ──
        if sma_slope > 0.003:
            votes.append(MarketRegime.TRENDING_UP)
        elif sma_slope < -0.003:
            votes.append(MarketRegime.TRENDING_DOWN)
        else:
            votes.append(MarketRegime.RANGING)

        # ── RSI vote ──
        if rsi > 65:
            votes.append(MarketRegime.TRENDING_UP)
        elif rsi < 35:
            votes.append(MarketRegime.TRENDING_DOWN)
        else:
            votes.append(MarketRegime.RANGING)

        # Tally votes
        if not votes:
            return MarketRegime.RANGING, 0.5

        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        confidence = count / len(votes)

        # Minimum confidence floor
        confidence = max(confidence, 0.3)

        return winner, confidence
