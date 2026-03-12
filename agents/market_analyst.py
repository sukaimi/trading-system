"""Market Analyst Agent — Tier 1 (DeepSeek, with escalation second pass).

Transforms signals into trade theses with entry/exit criteria,
confidence scores, and risk quantification.
See TRADING_AGENT_PRD.md Section 3.2.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.asset_registry import get_registry, get_tradeable_assets
from core.schemas import (
    ConfirmingSignal,
    ConfirmingSignals,
    Direction,
    SignalAlert,
    TimeHorizon,
    TradeThesis,
)
from tools.alpha_vantage import AlphaVantageClient, get_av_client
from tools.correlation import CorrelationAnalyzer
from tools.market_data import MarketDataFetcher
from tools.technical_indicators import TechnicalIndicators

log = setup_logger("trading.market_analyst")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
PARAMS_FILE = os.path.join(CONFIG_DIR, "market_analyst_params.json")

SYSTEM_PROMPT = """You are a Quantitative Market Analyst at Bridgewater Associates. You specialize in cross-asset correlation analysis (crypto, gold, USD, bonds), regime detection, and multi-timeframe technical analysis.

You have a strict "no thesis, no trade" policy. You require minimum 2 independent confirming signals. You explicitly quantify what could go wrong. Silence is a valid output — only propose trades when genuine edge exists."""

ANALYSIS_PROMPT = """You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no text before or after the JSON.

Analyze this trading signal and generate a trade thesis if warranted.

Signal: {signal}

Current Technical Data for {asset}:
{tech_data}

Current Market Context:
{market_context}

Requirements:
- There are 4 confirming signal types: fundamental, technical indicators, cross-asset, chart pattern
- Chart pattern analysis is done separately — focus on fundamental, technical indicators, and cross-asset here
- Minimum 1 confirming signal to propose a micro-position, 2+ for standard position
- Confidence 0.8-1.0 = 3+ confirm, 0.5-0.79 = 2 of 4, 0.4-0.49 = 1 signal (micro-position)
- Position sizing: 2% for 0.4-0.5 confidence, 4% for 0.5-0.65, 7% for 0.65+
- During paper trading: we want DATA — propose trades even with modest confidence

Respond with ONLY this JSON structure (no other text):
{{"asset": "{asset}", "direction": "long|short|neutral", "confidence": 0.0-1.0, "thesis": "2-3 sentence rationale", "confirming_signals": {{"fundamental": {{"present": true/false, "description": "..."}}, "technical": {{"present": true/false, "description": "..."}}, "cross_asset": {{"present": true/false, "description": "..."}}}}, "entry_trigger": "...", "invalidation_level": "...", "time_horizon": "1-3 days|1 week|swing", "suggested_position_pct": 0.0, "risk_reward_ratio": "...", "supporting_data": {{}}, "what_could_go_wrong": ["..."]}}

If no trade warranted (confidence < 0.4), respond with ONLY: {{"no_trade": true, "reason": "..."}}"""


class MarketAnalyst:
    """Quantitative market analyst & trade thesis generator."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        market_data: MarketDataFetcher | None = None,
        indicators: TechnicalIndicators | None = None,
        correlation: CorrelationAnalyzer | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._mdf = market_data or MarketDataFetcher()
        self._ti = indicators or TechnicalIndicators()
        self._corr = correlation or CorrelationAnalyzer(self._mdf)
        self._params = self._load_params()

    def analyze_signal(self, signal_alert: SignalAlert) -> TradeThesis | None:
        """Analyze a signal and generate a trade thesis or None."""
        asset = signal_alert.asset
        if asset == "MACRO":
            # Macro signals analyzed across all tradeable assets
            return None

        tech_data = self._gather_technical_data(asset)
        market_ctx = self._mdf.get_market_context()
        regime = self._corr.detect_regime({
            "vix": market_ctx.get("vix", 20),
            "btc_change_7d": tech_data.get("price_change_7d", 0),
        })

        prompt = ANALYSIS_PROMPT.format(
            signal=json.dumps(signal_alert.model_dump(), default=str),
            asset=asset,
            tech_data=json.dumps(tech_data, default=str),
            market_context=json.dumps({**market_ctx, "regime": regime}, default=str),
        )

        # First pass with DeepSeek
        result = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)

        # DeepSeek sometimes returns a list or string instead of dict
        if isinstance(result, list):
            result = result[0] if result else {"error": "empty list response"}
        if not isinstance(result, dict):
            result = {"error": f"Unexpected response type: {type(result).__name__}"}

        if result.get("no_trade") or result.get("error"):
            log.info("No trade for %s: %s", asset, result.get("reason", result.get("error")))
            return None

        # Build thesis
        thesis = self._build_thesis(result, signal_alert, tech_data)
        if thesis is None:
            return None

        # Check escalation (was Kimi, now DeepSeek for cost savings)
        if self.should_escalate(thesis):
            log.info("Escalating %s thesis to DeepSeek (second pass)", asset)
            escalated = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)
            if isinstance(escalated, list):
                escalated = escalated[0] if escalated else {"error": "empty list response"}
            if not isinstance(escalated, dict):
                escalated = {"error": f"Unexpected response type: {type(escalated).__name__}"}
            if not escalated.get("no_trade") and not escalated.get("error"):
                thesis = self._build_thesis(escalated, signal_alert, tech_data)
                if thesis:
                    thesis.model_used = "deepseek"
                    thesis.escalated = True

        # Check minimum confidence
        min_conf = self._params.get("min_confidence_for_trade", 0.6)
        if thesis and thesis.confidence < min_conf:
            log.info("Thesis confidence %.2f below threshold %.2f", thesis.confidence, min_conf)
            return None

        return thesis

    def scheduled_analysis(self, session: str) -> list[TradeThesis]:
        """Run scheduled market analysis for all assets."""
        log.info("Running scheduled analysis: %s", session)
        assets = get_tradeable_assets()
        theses = []

        for asset in assets:
            # Create a synthetic signal for scheduled analysis
            signal = SignalAlert(
                asset=asset,  # already a string from registry
                signal_strength=0.5,
                headline=f"Scheduled {session} analysis for {asset}",
                sentiment="neutral",
                category="macro",
                new_information=f"Scheduled {session} scan",
                urgency="medium",
                confidence_in_classification=0.5,
            )
            thesis = self.analyze_signal(signal)
            if thesis:
                theses.append(thesis)

        log.info("Scheduled analysis produced %d theses", len(theses))
        return theses

    def should_escalate(self, thesis: TradeThesis) -> bool:
        """Determine if thesis should be escalated to DeepSeek second pass."""
        if thesis.confidence >= 0.6 and thesis.suggested_position_pct > 3.0:
            return True
        asset_config = get_registry().get_config(thesis.asset)
        if asset_config.get("escalate_swing") and thesis.time_horizon == TimeHorizon.SWING:
            return True
        if len(thesis.what_could_go_wrong) >= 3:
            return True
        return False

    def _gather_technical_data(self, asset: str) -> dict[str, Any]:
        """Fetch OHLCV and compute all technical indicators."""
        ohlcv = self._mdf.get_ohlcv(asset, period="3mo", interval="1d")
        if not ohlcv:
            return {"error": "No OHLCV data available"}

        closes = [bar["close"] for bar in ohlcv]
        highs = [bar["high"] for bar in ohlcv]
        lows = [bar["low"] for bar in ohlcv]
        volumes = [bar["volume"] for bar in ohlcv]

        current_price = closes[-1] if closes else 0
        price_7d_ago = closes[-7] if len(closes) >= 7 else current_price

        # Volume vs 20d average
        vol_avg_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0
        vol_ratio = volumes[-1] / vol_avg_20 if vol_avg_20 > 0 else 0

        result = {
            "current_price": current_price,
            "price_change_7d": ((current_price - price_7d_ago) / price_7d_ago * 100)
            if price_7d_ago
            else 0,
            "rsi_14": self._ti.rsi(closes),
            "macd": self._ti.macd(closes),
            "bollinger": self._ti.bollinger_bands(closes),
            "atr_14": self._ti.atr(highs, lows, closes),
            "sma_50": self._ti.sma(closes, 50),
            "sma_200": self._ti.sma(closes, 200),
            "volume_vs_avg": round(vol_ratio, 2),
            "vwap": self._ti.vwap(highs, lows, closes, volumes),
            "stochastic": self._ti.stochastic(highs, lows, closes),
            "obv": self._ti.obv(closes, volumes),
        }

        # Alpha Vantage sentiment (P0: zero extra API calls)
        try:
            av = get_av_client()
            sentiment = av.ticker_sentiment(asset)
            if sentiment:
                result["av_sentiment"] = sentiment["score"]
                result["av_sentiment_label"] = sentiment["label"]
        except Exception:
            pass  # Graceful fallback — omit if unavailable

        # Alpha Vantage earnings context (P1)
        try:
            from core.earnings_calendar import EarningsCalendar
            cal = EarningsCalendar()
            days = cal.days_until_earnings(asset)
            if days is not None:
                result["days_until_earnings"] = days
            eps_history = cal.get_eps_history(asset)
            if eps_history:
                result["eps_history"] = eps_history
        except Exception:
            pass

        # Alpha Vantage sector performance (P2)
        try:
            av = get_av_client()
            rankings = av.get_sector_rankings("1mo")
            if rankings:
                result["sector_rankings"] = rankings
        except Exception:
            pass

        return result

    def _build_thesis(
        self,
        llm_result: dict[str, Any],
        signal: SignalAlert,
        tech_data: dict[str, Any],
    ) -> TradeThesis | None:
        """Build a TradeThesis from LLM result."""
        try:
            # Parse confirming signals (LLM may return list instead of dict)
            cs_data = llm_result.get("confirming_signals", {})
            if not isinstance(cs_data, dict):
                cs_data = {}
            def _as_dict(val):
                """Ensure confirming signal value is a dict (LLM may return list)."""
                if isinstance(val, list):
                    return val[0] if val and isinstance(val[0], dict) else {}
                return val if isinstance(val, dict) else {}

            confirming = ConfirmingSignals(
                fundamental=ConfirmingSignal(
                    present=_as_dict(cs_data.get("fundamental", {})).get("present", False),
                    description=_as_dict(cs_data.get("fundamental", {})).get("description", ""),
                ),
                technical=ConfirmingSignal(
                    present=_as_dict(cs_data.get("technical", {})).get("present", False),
                    description=_as_dict(cs_data.get("technical", {})).get("description", ""),
                ),
                cross_asset=ConfirmingSignal(
                    present=_as_dict(cs_data.get("cross_asset", {})).get("present", False),
                    description=_as_dict(cs_data.get("cross_asset", {})).get("description", ""),
                ),
            )

            # Map confidence to tiered position size (Kelly-inspired)
            confidence = float(llm_result.get("confidence", 0.0))
            if confidence >= 0.65:
                default_size = 7.0
            elif confidence >= 0.50:
                default_size = 4.0
            else:
                default_size = 2.0  # Micro position — generate data

            thesis = TradeThesis(
                asset=llm_result.get("asset", signal.asset),
                direction=Direction(llm_result.get("direction", "neutral")),
                confidence=confidence,
                thesis=llm_result.get("thesis", ""),
                confirming_signals=confirming,
                entry_trigger=llm_result.get("entry_trigger", ""),
                invalidation_level=llm_result.get("invalidation_level", ""),
                time_horizon=self._parse_time_horizon(llm_result.get("time_horizon", "1-3 days")),
                suggested_position_pct=float(
                    (lambda v: default_size if str(v).strip().rstrip('%').upper() in ('N/A', 'NONE', '') else str(v).strip().rstrip('%'))(
                        llm_result.get("suggested_position_pct", default_size)
                    )
                ),
                risk_reward_ratio=str(llm_result.get("risk_reward_ratio", "")),
                supporting_data=llm_result.get("supporting_data", tech_data)
                if isinstance(llm_result.get("supporting_data"), dict)
                else tech_data,
                what_could_go_wrong=llm_result.get("what_could_go_wrong", []),
                triggering_alert_id=str(id(signal)),
            )
            return thesis
        except (ValueError, KeyError, AttributeError) as e:
            log.warning("Failed to build thesis: %s", e)
            return None

    @staticmethod
    def _parse_time_horizon(value: str) -> TimeHorizon:
        """Parse time horizon, falling back to SHORT for invalid values."""
        try:
            return TimeHorizon(value)
        except ValueError:
            # Map common LLM variations
            v = value.lower().strip()
            if "swing" in v:
                return TimeHorizon.SWING
            if "week" in v:
                return TimeHorizon.MEDIUM
            return TimeHorizon.SHORT

    def _load_params(self) -> dict[str, Any]:
        try:
            with open(PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
