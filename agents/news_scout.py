"""News Scout Agent — Tier 1 (DeepSeek V3.2).

Scans, filters, and classifies financial news. Outputs SignalAlert objects.
See TRADING_AGENT_PRD.md Section 3.1.
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.asset_registry import get_registry, get_tradeable_assets
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
- "category": one of "central_bank", "geopolitical", "regulatory", "technical", "macro", "crypto_specific", "precious_metals", "equity"
- "new_information": what NEW fact isn't priced in
- "urgency": "critical", "high", "medium", or "low"
- "already_priced_in": true/false
- "confidence_in_classification": 0.0-1.0

Ticker mapping rules:
- Company-specific news (earnings, guidance, analyst upgrades/downgrades) → map to the specific ticker: AAPL, NVDA, TSLA, AMZN, or META
- Fed, interest rate, broad market, index, or S&P 500 news → SPY
- Gold, precious metals, gold ETF news → GLDM
- Silver news → SLV
- Bitcoin, BTC news → BTC
- Ethereum, ETH news → ETH
- Cross-market or unclear → MACRO

Reflexivity detection — for EACH article, also return:
- "reflexivity_flag": true/false — is this article part of a self-reinforcing narrative loop?
- "reflexivity_stage": one of "none", "forming", "peak", "reversal"

Detect reflexivity when:
- The article cites recent price action as evidence for the thesis ("BTC surged 10%, indicating strong momentum")
- Multiple sources are repeating the same narrative verbatim (echo chamber)
- Language has shifted from cautious ("may", "could") to extrapolative ("will continue", "inevitable")
- Price predictions are circular — the price is high because it will go higher

Stage definitions:
- "forming": narrative is building but not yet consensus (early believers)
- "peak": everyone agrees, price cited as proof, extrapolation language dominant
- "reversal": cracks appearing — the narrative is being questioned after price moved

Rules:
- IMPORTANT: We trade ALL asset classes — crypto, stocks, ETFs, gold, silver. Classify articles for ALL relevant assets, not just crypto. Earnings, guidance, analyst upgrades/downgrades, sector rotation, and company-specific news are actionable for equities.
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
        # Shuffle to avoid crypto-first bias from feed ordering
        shuffled = list(articles)
        random.shuffle(shuffled)

        # Format articles for the prompt
        article_text = "\n\n".join(
            f"[{i+1}] {a.get('title', 'No title')}\n"
            f"Source: {a.get('source', 'Unknown')}\n"
            f"Published: {a.get('published', 'Unknown')}\n"
            f"Summary: {a.get('summary', '')[:300]}"
            for i, a in enumerate(shuffled[:30])  # Limit to 30 articles per batch
        )

        valid_assets = ", ".join(f'"{a}"' for a in get_tradeable_assets() + ["MACRO"])
        prompt = CLASSIFY_PROMPT.format(articles=article_text, valid_assets=valid_assets)

        # Append historical signal accuracy context
        accuracy_context = self._load_signal_accuracy_context()
        if accuracy_context:
            prompt += f"\n\n{accuracy_context}"

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

    def _get_asset_class(self, asset: str) -> str:
        """Look up asset class from the asset registry. Returns 'MACRO' for unknown."""
        if asset == "MACRO":
            return "MACRO"
        registry = get_registry()
        config = registry.get_config(asset)
        return config.get("type", "MACRO")

    def _apply_filters(self, raw_signals: list[dict[str, Any]]) -> list[SignalAlert]:
        """Apply anti-noise rules and create SignalAlert objects."""
        now = datetime.utcnow()
        is_weekend = now.weekday() >= 5
        min_threshold = self._params.get("min_signal_threshold", 0.4)
        max_per_hour = self._params.get("max_alerts_per_hour", 7)
        weekend_penalty = self._params.get("weekend_signal_penalty", 0.15)

        # Per-class quotas
        per_class_quotas = self._params.get("per_class_quotas", {
            "crypto": 2, "stock": 3, "etf": 2, "MACRO": 1,
        })
        class_counters: dict[str, int] = {}

        # Clean up hourly counter
        cutoff = now.timestamp() - 3600
        self._alerts_this_hour = [t for t in self._alerts_this_hour if t > cutoff]

        alerts: list[SignalAlert] = []

        for sig in raw_signals:
            strength = sig.get("signal_strength", 0.0)

            # Weekend penalty (crypto trades 24/7, exempt from weekend penalty)
            if is_weekend and sig.get("asset", "MACRO") not in {"BTC", "ETH"}:
                strength -= weekend_penalty

            # Already priced in penalty
            if sig.get("already_priced_in", False):
                strength -= 0.35

            # Speculation filter
            headline = sig.get("headline", "")
            new_info = sig.get("new_information", "")
            combined_text = f"{headline} {new_info}".lower()
            if any(word in combined_text for word in ["could", "might", "may"]):
                if not any(word in combined_text for word in ["said", "announced", "confirmed"]):
                    strength -= 0.3

            # Reflexivity adjustment
            reflexivity_flag = sig.get("reflexivity_flag", False)
            reflexivity_stage = sig.get("reflexivity_stage", "none")
            if reflexivity_flag and reflexivity_stage != "none":
                if reflexivity_stage == "forming":
                    strength += 0.10
                    log.info("Reflexivity FORMING detected for %s — boosting signal +0.10", asset)
                elif reflexivity_stage == "peak":
                    strength -= 0.15
                    log.info("Reflexivity PEAK detected for %s — crowded trade warning, reducing signal -0.15", asset)
                elif reflexivity_stage == "reversal":
                    # Flip direction
                    current_sentiment = sig.get("sentiment", "neutral")
                    if current_sentiment == "bullish":
                        sig["sentiment"] = "bearish"
                    elif current_sentiment == "bearish":
                        sig["sentiment"] = "bullish"
                    strength += 0.10
                    log.info("Reflexivity REVERSAL detected for %s — flipping signal direction, boosting +0.10", asset)

            # Clamp
            strength = max(0.0, min(1.0, strength))

            # Threshold filter
            if strength < min_threshold:
                continue

            # Dedup
            headline_lower = headline.lower().strip()
            if headline_lower in self._recent_headlines:
                continue

            # Total alerts per hour cap
            if len(self._alerts_this_hour) >= max_per_hour:
                log.warning("Max alerts per hour reached (%d)", max_per_hour)
                break

            # Per-class quota check — skip signal if class quota full, continue to next
            asset = sig.get("asset", "MACRO")
            asset_class = self._get_asset_class(asset)
            class_limit = per_class_quotas.get(asset_class, max_per_hour)
            current_count = class_counters.get(asset_class, 0)
            if current_count >= class_limit:
                log.info("Per-class quota full for %s (%d/%d) — skipping %s",
                         asset_class, current_count, class_limit, headline[:60])
                continue

            # Build SignalAlert
            try:
                alert = SignalAlert(
                    asset=asset,
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
                    reflexivity_flag=reflexivity_flag,
                    reflexivity_stage=reflexivity_stage if reflexivity_flag else "none",
                )
                alerts.append(alert)
                self._recent_headlines.append(headline_lower)
                self._alerts_this_hour.append(now.timestamp())
                class_counters[asset_class] = current_count + 1
            except (ValueError, KeyError) as e:
                log.warning("Skipping invalid signal: %s", e)

        # Keep recent headlines list bounded
        self._recent_headlines = self._recent_headlines[-100:]

        log.info("Produced %d signal alerts after filtering", len(alerts))
        return alerts

    def _load_signal_accuracy_context(self) -> str:
        """Load signal accuracy data and return calibration context for the prompt."""
        data_file = Path(__file__).parent.parent / "data" / "signal_accuracy.json"
        try:
            if not data_file.exists():
                return ""
            with open(data_file) as f:
                signals = json.load(f)

            # Compute per-category win rates from closed trades
            by_category: dict[str, dict[str, int]] = {}
            total_wins = 0
            total_closed = 0
            for s in signals:
                if s.get("exit_price") is None:
                    continue
                cat = s.get("category", "unknown")
                if cat not in by_category:
                    by_category[cat] = {"wins": 0, "total": 0}
                by_category[cat]["total"] += 1
                total_closed += 1
                if s.get("signal_correct"):
                    by_category[cat]["wins"] += 1
                    total_wins += 1

            if total_closed < 5:
                return ""

            lines = ["Historical signal accuracy:"]
            for cat, stats in sorted(by_category.items()):
                if stats["total"] >= 5:
                    wr = round(stats["wins"] / stats["total"] * 100)
                    lines.append(f"  {cat} signals have {wr}% win rate ({stats['total']} trades).")
            overall_wr = round(total_wins / total_closed * 100)
            lines.append(f"  Overall signal precision: {overall_wr}% ({total_closed} closed trades).")
            lines.append("Calibrate your confidence accordingly.")
            return "\n".join(lines)
        except Exception:
            return ""

    def _load_params(self) -> dict[str, Any]:
        try:
            with open(PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
