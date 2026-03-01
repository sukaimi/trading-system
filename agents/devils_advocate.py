"""Devil's Advocate Agent — Tier 2 (Kimi K2.5).

Contrarian risk challenger that stress-tests every trade thesis
using a 7-dimension challenge framework. Outputs APPROVED,
APPROVED_WITH_MODIFICATION, or KILLED verdict.

Kill rate target: 40-60%. False kill rate target: <20%.

See TRADING_AGENT_PRD.md Section 3.3 for full specification.

Phase 2 implementation — CrewAI Agent with Kimi K2.5 LLM.
"""


class DevilsAdvocate:
    """Contrarian risk challenger & pre-trade stress tester (Phase 2 stub)."""

    def challenge(self, trade_thesis: dict) -> dict:
        """Challenge a trade thesis on all 7 dimensions."""
        raise NotImplementedError("Phase 2 — CrewAI + Kimi challenge framework")

    def check_fatal_flaws(self, thesis: dict, portfolio_state: dict) -> list[str]:
        """Check for auto-kill fatal flaws."""
        raise NotImplementedError("Phase 2 — fatal flaw detection")
