"""News Scout Agent — Tier 1 (DeepSeek V3.2).

Scans, filters, and classifies financial news. Outputs SignalAlert objects.
See TRADING_AGENT_PRD.md Section 3.1.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.asset_registry import get_tradeable_assets
from core.schemas import Sentiment, SignalAlert, SignalCategory, Urgency
from tools.news_fetcher import NewsFetcher

log = setup_logger("trading.news_scout")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
PARAMS_FILE = os.path.join(CONFIG_DIR, "news_scout_params.json")

SYSTEM_PROMPT = """You are a Financial News Intelligence Analyst. You spent 12 years at a macro hedge fund filtering 3,000+ daily headlines into the 5-10 that actually move markets. You specialize in central bank policy, geopolitical events, crypto catalysts, precious metals drivers, and black swan detection.

You NEVER forward recycled opinion pieces. You ONLY escalate when you detect genuinely new information that isn't already priced in. Your false positive rate is below 10%."""

CLASSIFY_PROMPT = """Analyze the following news articles and classify each as a trading signal.

For each article that contains genuinely actionable information, return a JSON object with:
- "asset": one of {valid_assets}
- "signal_strength": 0.0-1.0 (0.8-1.0=CRITICAL, 0.5-0.7=NOTABLE, 0.3-0.4=MONITOR, <0.3=NOISE)
- "headline": max 100 chars summary
- "sentiment": "bullish", "bearish", "neutral", or "uncertain"
- "category": one of "central_bank", "geopolitical", "regulatory", "technical", "macro", "crypto_specific", "precious_metals"
- "new_information": what NEW fact isn't priced in
- "urgency": "critical", "high", "medium", or "low"
- "already_priced_in": true/false
- "confidence_in_classification": 0.0-1.0

Rules:
- If 3+ outlets report the same story, it's already priced in — set already_priced_in=true and downgrade signal_strength by 0.2
- If article uses "could/might/may" without citing a named source, downgrade by 0.3
- For scheduled events, only flag if actual deviates from consensus by >10%
- Discard articles with signal_strength < 0.3

Return a JSON array of signal objects. If no articles are actionable, return an empty array [].

Articles:
{articles}"""


class NewsScout:
    """Financial news intelligence analyst."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        news_fetcher: NewsFetcher | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._fetcher = news_fetcher or NewsFetcher()
        self._params = self._load_params()
        self._recent_headlines: list[str] = []
        self._alerts_this_hour: list[float] = []

    def scan(self) -> list[SignalAlert]:
        """Scan all news sources and return signal alerts."""
        articles = self._fetcher.fetch_all()
        if not articles:
            log.info("No news articles found")
            return []

        log.info("Classifying %d articles", len(articles))
        raw_signals = self.classify_articles(articles)
        return self._apply_filters(raw_signals)

    def classify_articles(self, articles: list[dict]) -> list[dict[str, Any]]:
        """Send articles to DeepSeek for classification."""
        # Format articles for the prompt
        article_text = "\n\n".join(
            f"[{i+1}] {a.get('title', 'No title')}\n"
            f"Source: {a.get('source', 'Unknown')}\n"
            f"Published: {a.get('published', 'Unknown')}\n"
            f"Summary: {a.get('summary', '')[:300]}"
            for i, a in enumerate(articles[:20])  # Limit to 20 articles per batch
        )

        valid_assets = ", ".join(f'"{a}"' for a in get_tradeable_assets() + ["MACRO"])
        prompt = CLASSIFY_PROMPT.format(articles=article_text, valid_assets=valid_assets)
        result = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)

        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            if "error" in result:
                log.warning("LLM classification failed: %s", result["error"])
                return []
            # Might be a single signal wrapped in a dict
            return [result] if "signal_strength" in result else []

        return []

    def _apply_filters(self, raw_signals: list[dict[str, Any]]) -> list[SignalAlert]:
        """Apply anti-noise rules and create SignalAlert objects."""
        now = datetime.utcnow()
        is_weekend = now.weekday() >= 5
        min_threshold = self._params.get("min_signal_threshold", 0.4)
        max_per_hour = self._params.get("max_alerts_per_hour", 5)
        weekend_penalty = self._params.get("weekend_signal_penalty", 0.15)

        # Clean up hourly counter
        cutoff = now.timestamp() - 3600
        self._alerts_this_hour = [t for t in self._alerts_this_hour if t > cutoff]

        alerts: list[SignalAlert] = []

        for sig in raw_signals:
            strength = sig.get("signal_strength", 0.0)

            # Weekend penalty
            if is_weekend:
                strength -= weekend_penalty

            # Already priced in penalty
            if sig.get("already_priced_in", False):
                strength -= 0.2

            # Speculation filter
            headline = sig.get("headline", "")
            new_info = sig.get("new_information", "")
            combined_text = f"{headline} {new_info}".lower()
            if any(word in combined_text for word in ["could", "might", "may"]):
                if not any(word in combined_text for word in ["said", "announced", "confirmed"]):
                    strength -= 0.3

            # Clamp
            strength = max(0.0, min(1.0, strength))

            # Threshold filter
            if strength < min_threshold:
                continue

            # Dedup
            headline_lower = headline.lower().strip()
            if headline_lower in self._recent_headlines:
                continue

            # Max alerts per hour
            if len(self._alerts_this_hour) >= max_per_hour:
                log.warning("Max alerts per hour reached (%d)", max_per_hour)
                break

            # Build SignalAlert
            try:
                alert = SignalAlert(
                    asset=sig.get("asset", "MACRO"),
                    signal_strength=strength,
                    headline=headline[:100],
                    sentiment=Sentiment(sig.get("sentiment", "neutral")),
                    category=SignalCategory(sig.get("category", "macro")),
                    new_information=new_info,
                    raw_sources=[sig.get("source", "")],
                    urgency=Urgency(sig.get("urgency", "medium")),
                    already_priced_in=sig.get("already_priced_in", False),
                    confidence_in_classification=sig.get(
                        "confidence_in_classification", strength
                    ),
                )
                alerts.append(alert)
                self._recent_headlines.append(headline_lower)
                self._alerts_this_hour.append(now.timestamp())
            except (ValueError, KeyError) as e:
                log.warning("Skipping invalid signal: %s", e)

        # Keep recent headlines list bounded
        self._recent_headlines = self._recent_headlines[-100:]

        log.info("Produced %d signal alerts after filtering", len(alerts))
        return alerts

    def _load_params(self) -> dict[str, Any]:
        try:
            with open(PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
