"""Weekly Strategist Agent — Tier 3 (Claude Opus 4.6).

Conducts comprehensive weekly review of all trading activity.
Identifies patterns, assesses strategy performance, and produces
specific actionable directives with parameter changes.

Runs every Sunday. ~$1.00/month.

See TRADING_AGENT_PRD.md Section 3.7 for full specification.

Phase 3 implementation — CrewAI Agent with Opus 4.6 LLM.
"""


class WeeklyStrategist:
    """Chief investment strategist (Phase 3 stub)."""

    def review_week(self, weekly_package: dict) -> dict:
        """Conduct weekly strategy review and produce directives."""
        raise NotImplementedError("Phase 3 — Opus weekly review")

    def assess_regime(self, market_data: dict) -> str:
        """Assess current market regime."""
        raise NotImplementedError("Phase 3 — regime assessment")
