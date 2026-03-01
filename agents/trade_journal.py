"""Trade Journal Agent — Tier 1 (DeepSeek V3.2).

Creates meticulous, structured records of every trade decision.
Records entries, exits, theses, outcomes, and lessons. The journal
feeds the weekly self-optimization loop.

See TRADING_AGENT_PRD.md Section 3.6 for full specification.

Phase 2 implementation — CrewAI Agent with DeepSeek LLM.
"""


class TradeJournal:
    """Quantitative trade documentarian (Phase 2 stub)."""

    def record_entry(self, trade: dict) -> dict:
        """Record a trade entry in the journal."""
        raise NotImplementedError("Phase 2 — trade entry journaling")

    def record_exit(self, trade_id: str, exit_data: dict) -> dict:
        """Record a trade exit with outcome and lessons."""
        raise NotImplementedError("Phase 2 — trade exit journaling")

    def record_no_trade(self, signal: dict, reasoning: str) -> dict:
        """Record a no-trade decision."""
        raise NotImplementedError("Phase 2 — no-trade journaling")

    def assemble_weekly_package(self, week_ending: str) -> dict:
        """Assemble the weekly input package for the Strategist."""
        raise NotImplementedError("Phase 2 — weekly package assembly")
