"""Chart Scanner — Tier 0 pure Python technical signal generator.

Scans all tradeable assets using technical indicators (RSI, MACD, Bollinger
Bands, EMA crossovers) and generates SignalAlert objects when conditions trigger.
No LLM calls. Runs every 2 hours alongside news scans.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import setup_logger
from core.schemas import SignalAlert, SignalCategory, Sentiment, Urgency
from tools.market_data import MarketDataFetcher
from tools.technical_indicators import TechnicalIndicators

log = setup_logger("trading.chart_scanner")

# Default config — overridden by config/chart_scanner_params.json
_DEFAULT_CONFIG = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_squeeze_threshold": 0.03,
    "macd_histogram_threshold": 0.0,
    "ema_fast": 9,
    "ema_slow": 21,
    "min_signal_strength": 0.4,
    "lookback_period": "3mo",
    "lookback_interval": "1d",
    "weights": {
        "trend": 0.30,
        "momentum": 0.25,
        "volatility": 0.20,
        "volume": 0.15,
        "pattern": 0.10,
    },
}


def _load_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "config" / "chart_scanner_params.json"
    try:
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            merged = {**_DEFAULT_CONFIG, **cfg}
            merged["weights"] = {**_DEFAULT_CONFIG["weights"], **cfg.get("weights", {})}
            return merged
    except Exception as e:
        log.warning("Could not load chart_scanner_params.json: %s", e)
    return _DEFAULT_CONFIG.copy()


class ChartScanner:
    """Tier 0 technical scanner — generates SignalAlerts from price action."""

    def __init__(self) -> None:
        self._ti = TechnicalIndicators()
        self._mdf = MarketDataFetcher()
        self._cfg = _load_config()

    def scan_all(self) -> list[SignalAlert]:
        """Scan all tradeable assets and return triggered signals."""
        from core.asset_registry import get_tradeable_assets

        assets = get_tradeable_assets()
        signals: list[SignalAlert] = []

        for symbol in assets:
            try:
                signal = self._scan_asset(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                log.warning("Chart scan failed for %s: %s", symbol, e)

        log.info("Chart scan complete: %d signals from %d assets", len(signals), len(assets))
        return signals

    def _scan_asset(self, symbol: str) -> SignalAlert | None:
        """Analyze one asset and return a SignalAlert if conditions trigger."""
        ohlcv = self._mdf.get_ohlcv(
            symbol,
            period=self._cfg["lookback_period"],
            interval=self._cfg["lookback_interval"],
        )
        if len(ohlcv) < 35:
            return None

        closes = [bar["close"] for bar in ohlcv]
        volumes = [bar.get("volume", 0) for bar in ohlcv]

        # Calculate indicators
        rsi = self._ti.rsi(closes)
        macd = self._ti.macd(closes)
        bb = self._ti.bollinger_bands(closes)
        ema_fast_series = self._ti._ema(closes, self._cfg["ema_fast"])
        ema_slow_series = self._ti._ema(closes, self._cfg["ema_slow"])
        ema_fast = ema_fast_series[-1] if ema_fast_series else 0.0
        ema_slow = ema_slow_series[-1] if ema_slow_series else 0.0
        current_price = closes[-1]

        # Score components
        scores = self._compute_scores(
            rsi, macd, bb, ema_fast, ema_slow, current_price, closes, volumes
        )

        # Composite signal
        w = self._cfg["weights"]
        composite = (
            scores["trend"] * w["trend"]
            + scores["momentum"] * w["momentum"]
            + scores["volatility"] * w["volatility"]
            + scores["volume"] * w["volume"]
            + scores["pattern"] * w["pattern"]
        )

        # Direction from composite sign
        strength = abs(composite)
        if strength < self._cfg["min_signal_strength"]:
            return None

        # Determine direction and sentiment
        if composite > 0:
            sentiment = Sentiment.BULLISH
        elif composite < 0:
            sentiment = Sentiment.BEARISH
        else:
            return None

        # Build headline
        triggers = self._describe_triggers(rsi, macd, bb, ema_fast, ema_slow, current_price)
        headline = f"{symbol}: {', '.join(triggers[:2])}" if triggers else f"{symbol}: technical signal"
        headline = headline[:100]

        # Urgency based on strength
        if strength >= 0.75:
            urgency = Urgency.HIGH
        elif strength >= 0.55:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        return SignalAlert(
            source="chart_scanner",
            asset=symbol,
            signal_strength=min(strength, 1.0),
            headline=headline,
            sentiment=sentiment,
            category=SignalCategory.TECHNICAL,
            new_information=f"Technical indicators triggered: {', '.join(triggers)}",
            urgency=urgency,
            already_priced_in=True,
            confidence_in_classification=min(strength, 1.0),
        )

    def _compute_scores(
        self,
        rsi: float,
        macd: dict,
        bb: dict,
        ema_fast: float,
        ema_slow: float,
        price: float,
        closes: list[float],
        volumes: list[float],
    ) -> dict[str, float]:
        """Compute component scores in [-1, 1] range."""

        # Trend: EMA crossover direction
        if ema_fast > 0 and ema_slow > 0:
            trend = (ema_fast - ema_slow) / ema_slow
            trend = max(-1.0, min(1.0, trend * 20))  # scale small differences
        else:
            trend = 0.0

        # Momentum: RSI + MACD histogram
        rsi_score = 0.0
        if rsi <= self._cfg["rsi_oversold"]:
            rsi_score = (self._cfg["rsi_oversold"] - rsi) / self._cfg["rsi_oversold"]
        elif rsi >= self._cfg["rsi_overbought"]:
            rsi_score = -(rsi - self._cfg["rsi_overbought"]) / (100 - self._cfg["rsi_overbought"])

        macd_score = 0.0
        if macd["signal_line"] != 0:
            macd_score = macd["histogram"] / abs(macd["signal_line"]) if macd["signal_line"] != 0 else 0
            macd_score = max(-1.0, min(1.0, macd_score))

        momentum = (rsi_score * 0.5 + macd_score * 0.5)

        # Volatility: Bollinger bandwidth squeeze/expansion
        volatility = 0.0
        if bb["bandwidth"] > 0 and bb["middle"] > 0:
            if bb["bandwidth"] < self._cfg["bb_squeeze_threshold"]:
                # Squeeze — use price position relative to middle band
                bb_pos = (price - bb["middle"]) / (bb["upper"] - bb["middle"]) if bb["upper"] != bb["middle"] else 0
                volatility = max(-0.5, min(0.5, bb_pos * 0.5))
            elif price > bb["upper"]:
                volatility = 0.7  # breakout above
            elif price < bb["lower"]:
                volatility = -0.7  # breakout below

        # Volume: compare recent vs average
        volume_score = 0.0
        if len(volumes) >= 20 and sum(volumes[-20:]) > 0:
            avg_vol = sum(volumes[-20:]) / 20
            recent_vol = volumes[-1]
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio > 1.5:
                    volume_score = min(1.0, (vol_ratio - 1.0))
                    # Direction from price movement
                    if len(closes) >= 2 and closes[-1] < closes[-2]:
                        volume_score = -volume_score

        # Pattern: price relative to key moving averages
        pattern = 0.0
        if len(closes) >= 50:
            sma50 = sum(closes[-50:]) / 50
            sma20 = sum(closes[-20:]) / 20
            if sma20 > sma50 and price > sma20:
                pattern = 0.5  # bullish structure
            elif sma20 < sma50 and price < sma20:
                pattern = -0.5  # bearish structure

        return {
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "volume": volume_score,
            "pattern": pattern,
        }

    def _describe_triggers(
        self,
        rsi: float,
        macd: dict,
        bb: dict,
        ema_fast: float,
        ema_slow: float,
        price: float,
    ) -> list[str]:
        """Generate human-readable trigger descriptions."""
        triggers = []

        if rsi <= self._cfg["rsi_oversold"]:
            triggers.append(f"RSI oversold ({rsi:.0f})")
        elif rsi >= self._cfg["rsi_overbought"]:
            triggers.append(f"RSI overbought ({rsi:.0f})")

        if macd["histogram"] > 0 and macd["macd_line"] > macd["signal_line"]:
            triggers.append("MACD bullish crossover")
        elif macd["histogram"] < 0 and macd["macd_line"] < macd["signal_line"]:
            triggers.append("MACD bearish crossover")

        if price > bb["upper"] and bb["upper"] > 0:
            triggers.append("Above upper Bollinger Band")
        elif price < bb["lower"] and bb["lower"] > 0:
            triggers.append("Below lower Bollinger Band")

        if ema_fast > ema_slow > 0:
            triggers.append("EMA bullish alignment")
        elif ema_slow > ema_fast > 0:
            triggers.append("EMA bearish alignment")

        if bb["bandwidth"] > 0 and bb["bandwidth"] < self._cfg["bb_squeeze_threshold"]:
            triggers.append("Bollinger squeeze")

        return triggers
