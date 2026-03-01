"""News Scout Agent — Tier 1 (DeepSeek V3.2).

Scans, filters, and classifies financial news that could impact
portfolio assets (BTC, ETH, GLDM, SLV). Outputs signal_alert JSON.

Schedule:
- Every 30 min during market hours
- Every 60 min off-hours
- Immediate on CRITICAL keyword triggers

See TRADING_AGENT_PRD.md Section 3.1 for full specification.

Phase 2 implementation — CrewAI Agent with DeepSeek LLM.
"""


class NewsScout:
    """Financial news intelligence analyst (Phase 2 stub)."""

    def scan(self) -> list[dict]:
        """Scan all news sources and return signal alerts."""
        raise NotImplementedError("Phase 2 — CrewAI + DeepSeek news scanning")

    def classify_signal(self, article: dict) -> dict:
        """Classify a news article into a signal_alert."""
        raise NotImplementedError("Phase 2 — signal classification")
