"""Chart Analyst Agent — Tier 1 (DeepSeek V3.2).

Dedicated price-action and chart pattern analyst. Reads OHLCV data and
identifies support/resistance levels, candlestick patterns, trendlines,
and volume profile signals. Outputs a ChartSignal that feeds into the
pipeline as a 4th confirming signal alongside fundamental, technical
indicators, and cross-asset correlation.

Runs in parallel with MarketAnalyst — adds no latency to the pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from tools.market_data import MarketDataFetcher
from tools.technical_indicators import TechnicalIndicators

log = setup_logger("trading.chart_analyst")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
PARAMS_FILE = os.path.join(CONFIG_DIR, "chart_analyst_params.json")

SYSTEM_PROMPT = """You are a Price Action Specialist who spent 15 years reading charts at a proprietary trading desk. You don't care about news, fundamentals, or opinions — you only care about what the chart is telling you.

Your edge: pattern recognition across timeframes, volume confirmation, and precise support/resistance identification. You've seen thousands of chart setups and know which ones actually have a statistical edge."""

CHART_ANALYSIS_PROMPT = """Analyze this price data for {asset} and identify any actionable chart patterns.

OHLCV Data (last 60 days, daily candles):
{ohlcv_data}

Technical Indicators:
{indicators}

Analyze the following dimensions:

1. SUPPORT & RESISTANCE
   - Key horizontal levels from recent price action
   - How many times each level has been tested
   - Distance from current price to nearest S/R

2. CANDLESTICK PATTERNS (last 5 candles)
   - Any reversal patterns (hammer, engulfing, doji at extremes)?
   - Any continuation patterns (three white soldiers, rising three)?
   - Volume confirmation on patterns?

3. TREND STRUCTURE
   - Higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)?
   - Is price making a breakout or breakdown?
   - Is price in a channel, triangle, or wedge?

4. VOLUME ANALYSIS
   - Is volume confirming the move?
   - Any volume divergence (price up but volume dropping)?
   - Unusual volume spikes?

5. OVERALL CHART VERDICT
   - "bullish" / "bearish" / "neutral"
   - Confidence 0.0-1.0 in the pattern
   - Suggested entry zone and invalidation level

Return JSON:
{{
  "pattern_found": true/false,
  "pattern_name": "name of dominant pattern or 'none'",
  "direction": "long" / "short" / "neutral",
  "confidence": 0.0-1.0,
  "support_levels": [list of key support prices],
  "resistance_levels": [list of key resistance prices],
  "candlestick_pattern": "name or null",
  "volume_confirms": true/false,
  "trend": "uptrend" / "downtrend" / "sideways",
  "description": "1-2 sentence chart read",
  "entry_zone": "price or range",
  "invalidation": "price level that disproves the pattern"
}}

If no clear pattern exists, return {{"pattern_found": false, "direction": "neutral", "confidence": 0.0, "description": "No actionable pattern"}}"""


class ChartAnalyst:
    """Price action and chart pattern analyst (Tier 1 — DeepSeek)."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        market_data: MarketDataFetcher | None = None,
        indicators: TechnicalIndicators | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._mdf = market_data or MarketDataFetcher()
        self._ti = indicators or TechnicalIndicators()
        self._params = self._load_params()

    def analyze(self, asset: str) -> dict[str, Any]:
        """Analyze chart patterns for an asset.

        Returns a dict with pattern_found, direction, confidence, description,
        and supporting data. Returns a neutral/empty result on any error.
        """
        try:
            ohlcv = self._mdf.get_ohlcv(asset, period="3mo", interval="1d")
            if not ohlcv or len(ohlcv) < 20:
                log.info("Insufficient OHLCV data for %s chart analysis", asset)
                return self._empty_result()

            # Prepare data for the LLM
            ohlcv_summary = self._format_ohlcv(ohlcv)
            indicators = self._compute_indicators(ohlcv)

            prompt = CHART_ANALYSIS_PROMPT.format(
                asset=asset,
                ohlcv_data=json.dumps(ohlcv_summary, default=str),
                indicators=json.dumps(indicators, default=str),
            )

            result = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)

            if result.get("error"):
                log.warning("Chart analysis LLM error for %s: %s", asset, result["error"])
                return self._empty_result()

            # Validate and normalize
            return self._normalize_result(result, asset)

        except Exception as e:
            log.error("Chart analysis failed for %s: %s", asset, e)
            return self._empty_result()

    def _format_ohlcv(self, ohlcv: list[dict]) -> list[dict]:
        """Format OHLCV for the LLM prompt — last 60 bars with key stats."""
        bars = ohlcv[-60:]
        formatted = []
        for bar in bars:
            formatted.append({
                "date": bar.get("date", ""),
                "open": round(bar.get("open", 0), 2),
                "high": round(bar.get("high", 0), 2),
                "low": round(bar.get("low", 0), 2),
                "close": round(bar.get("close", 0), 2),
                "volume": bar.get("volume", 0),
            })
        return formatted

    def _compute_indicators(self, ohlcv: list[dict]) -> dict[str, Any]:
        """Compute indicators for chart context."""
        closes = [bar["close"] for bar in ohlcv]
        highs = [bar["high"] for bar in ohlcv]
        lows = [bar["low"] for bar in ohlcv]
        volumes = [bar["volume"] for bar in ohlcv]

        current = closes[-1] if closes else 0
        sma_20 = self._ti.sma(closes, 20)
        sma_50 = self._ti.sma(closes, 50)

        # Recent candle body sizes (last 5)
        recent_bodies = []
        for bar in ohlcv[-5:]:
            body = abs(bar["close"] - bar["open"])
            full_range = bar["high"] - bar["low"]
            recent_bodies.append({
                "body_pct": round(body / full_range * 100, 1) if full_range > 0 else 0,
                "bullish": bar["close"] > bar["open"],
            })

        # Volume trend (last 5 vs prior 20)
        vol_recent_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
        vol_20_avg = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0
        vol_trend = round(vol_recent_avg / vol_20_avg, 2) if vol_20_avg > 0 else 0

        # Price relative to SMAs
        price_vs_sma20 = round((current / sma_20 - 1) * 100, 2) if sma_20 > 0 else 0
        price_vs_sma50 = round((current / sma_50 - 1) * 100, 2) if sma_50 > 0 else 0

        # 52-week high/low approximation
        high_90d = max(highs[-90:]) if len(highs) >= 90 else max(highs) if highs else 0
        low_90d = min(lows[-90:]) if len(lows) >= 90 else min(lows) if lows else 0

        return {
            "current_price": current,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi_14": self._ti.rsi(closes),
            "bollinger": self._ti.bollinger_bands(closes),
            "atr_14": self._ti.atr(highs, lows, closes),
            "price_vs_sma20_pct": price_vs_sma20,
            "price_vs_sma50_pct": price_vs_sma50,
            "volume_trend_ratio": vol_trend,
            "recent_candles": recent_bodies,
            "90d_high": high_90d,
            "90d_low": low_90d,
        }

    def _normalize_result(self, result: dict[str, Any], asset: str) -> dict[str, Any]:
        """Normalize and validate LLM response."""
        return {
            "asset": asset,
            "pattern_found": bool(result.get("pattern_found", False)),
            "pattern_name": str(result.get("pattern_name", "none")),
            "direction": result.get("direction", "neutral"),
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
            "support_levels": result.get("support_levels", []),
            "resistance_levels": result.get("resistance_levels", []),
            "candlestick_pattern": result.get("candlestick_pattern"),
            "volume_confirms": bool(result.get("volume_confirms", False)),
            "trend": result.get("trend", "sideways"),
            "description": str(result.get("description", ""))[:300],
            "entry_zone": str(result.get("entry_zone", "")),
            "invalidation": str(result.get("invalidation", "")),
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty/neutral chart analysis result."""
        return {
            "pattern_found": False,
            "pattern_name": "none",
            "direction": "neutral",
            "confidence": 0.0,
            "support_levels": [],
            "resistance_levels": [],
            "candlestick_pattern": None,
            "volume_confirms": False,
            "trend": "sideways",
            "description": "No chart data available",
            "entry_zone": "",
            "invalidation": "",
        }

    def _load_params(self) -> dict[str, Any]:
        try:
            with open(PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
