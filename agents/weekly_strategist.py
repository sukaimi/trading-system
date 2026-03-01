"""Weekly Strategist Agent — Tier 3 (Claude Opus 4.6).

Conducts comprehensive weekly review of all trading activity.
Identifies patterns, assesses strategy performance, and produces
specific actionable directives with parameter changes.

Runs every Sunday. ~$1.00/month.

See TRADING_AGENT_PRD.md Section 3.7 for full specification.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.schemas import WeeklyDirective
from core.self_optimizer import SelfOptimizer
from tools.telegram_bot import TelegramNotifier

log = setup_logger("trading.weekly_strategist")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REVIEWS_DIR = os.path.join(DATA_DIR, "weekly_reviews")

SYSTEM_PROMPT = """You are the Chief Investment Strategist at a quantitative macro hedge fund. You have 20 years of experience managing systematic trading strategies across crypto, precious metals, and traditional assets.

Every Sunday you conduct a thorough review of the week's trading activity. You produce SPECIFIC, ACTIONABLE directives — not vague guidance. Every recommendation includes exact parameter values.

Your philosophy:
- A 40% win rate is fine if winners are 3x losers (positive expectancy)
- Position sizing matters more than entry timing
- Regime awareness prevents catastrophic losses
- Each agent parameter exists for a reason — change with evidence, not intuition
- Small, incremental adjustments beat dramatic overhauls

You are authorized to change any agent parameter. You communicate changes as structured JSON."""

REVIEW_PROMPT = """Conduct your weekly strategy review based on the following data.

=== PORTFOLIO PERFORMANCE ===
{portfolio_summary}

=== TRADE SUMMARY ===
{trade_summary}

=== FULL TRADE JOURNAL ===
{trade_journal}

=== CURRENT STRATEGY PARAMETERS ===
{strategy_params}

=== CURRENT RISK PARAMETERS ===
{risk_params}

=== WEEK ENDING ===
{week_ending}

Produce a JSON response with this exact structure:
{{
  "week_reviewed": "{week_ending}",
  "assessment": {{
    "overall": "1-2 sentence summary of the week",
    "what_worked": ["list of things that worked well"],
    "what_failed": ["list of things that didn't work"],
    "regime_assessment": "current market regime and outlook"
  }},
  "parameter_changes": [
    {{
      "target": "agent_name",
      "parameter": "param.path",
      "old_value": 0.0,
      "new_value": 0.0,
      "reason": "why this change"
    }}
  ],
  "next_week_focus": ["3+ specific focus items for next week"],
  "risk_adjustments": {{}}
}}

Valid targets: "news_scout", "market_analyst", "devils_advocate", "risk"
Use dot-notation for nested params (e.g., "signal_weights.central_bank").
If no changes needed, return empty parameter_changes array.
ONLY recommend changes supported by evidence from this week's data."""


class WeeklyStrategist:
    """Chief investment strategist — weekly review and directive generation."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        optimizer: SelfOptimizer | None = None,
        telegram: TelegramNotifier | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._optimizer = optimizer or SelfOptimizer()
        self._telegram = telegram

    def review_week(self, weekly_package: dict[str, Any]) -> WeeklyDirective:
        """Conduct weekly strategy review and produce directives."""
        week_ending = weekly_package.get("week_ending", "")

        # Build the review prompt
        prompt = REVIEW_PROMPT.format(
            portfolio_summary=json.dumps(
                weekly_package.get("portfolio_summary", {}), indent=2, default=str
            ),
            trade_summary=json.dumps(
                weekly_package.get("trade_summary", {}), indent=2, default=str
            ),
            trade_journal=json.dumps(
                weekly_package.get("full_trade_journal", [])[:20], indent=2, default=str
            ),
            strategy_params=json.dumps(
                weekly_package.get("current_strategy_params", {}), indent=2, default=str
            ),
            risk_params=json.dumps(
                weekly_package.get("current_risk_params", {}), indent=2, default=str
            ),
            week_ending=week_ending,
        )

        # Call Opus
        result = self._llm.call_anthropic(prompt, SYSTEM_PROMPT)

        if result.get("error"):
            log.warning("Opus review failed: %s — producing empty directive", result["error"])
            directive = WeeklyDirective(
                week_reviewed=week_ending,
                assessment={"overall": f"Review failed: {result['error']}"},
            )
        else:
            directive = self._parse_directive(result, week_ending)

        # Apply directives via SelfOptimizer
        if directive.parameter_changes:
            try:
                self._optimizer.apply_directives(directive.model_dump(mode="json"))
            except Exception as e:
                log.error("Failed to apply directives: %s", e)

        # Save directive to weekly_reviews/
        self._save_directive(directive, week_ending)

        # Send Telegram weekly report
        if self._telegram:
            try:
                trade_summary = weekly_package.get("trade_summary", {})
                self._telegram.send_weekly_report({
                    "week_reviewed": week_ending,
                    "weekly_return_pct": weekly_package.get(
                        "portfolio_summary", {}
                    ).get("return_pct", 0.0),
                    "total_trades": trade_summary.get("total_trades", 0),
                    "win_rate": trade_summary.get("win_rate", 0.0),
                })
            except Exception as e:
                log.error("Failed to send weekly report: %s", e)

        log.info("Weekly review complete for %s", week_ending)
        return directive

    def assess_regime(self, market_data: dict[str, Any]) -> str:
        """Assess current market regime using rule-based classification."""
        vix = market_data.get("vix")
        btc_change = market_data.get("btc_change_7d")
        gold_change = market_data.get("gold_change_7d")
        dxy_change = market_data.get("dxy_change_7d")

        if vix is None:
            return "unknown"

        if vix > 25:
            if gold_change is not None and gold_change > 0:
                return "risk_off"
            if dxy_change is not None and dxy_change > 0:
                return "risk_off"

        if vix < 20:
            if btc_change is not None and btc_change > 0:
                return "risk_on"

        return "transitional"

    def _parse_directive(
        self, result: dict[str, Any], week_ending: str
    ) -> WeeklyDirective:
        """Parse LLM result into a WeeklyDirective."""
        try:
            return WeeklyDirective(
                week_reviewed=result.get("week_reviewed", week_ending),
                assessment=result.get("assessment", {}),
                parameter_changes=[
                    {
                        "target": c.get("target", ""),
                        "parameter": c.get("parameter", ""),
                        "old_value": c.get("old_value", 0),
                        "new_value": c.get("new_value", 0),
                        "reason": c.get("reason", ""),
                    }
                    for c in result.get("parameter_changes", [])
                ],
                next_week_focus=result.get("next_week_focus", []),
                risk_adjustments=result.get("risk_adjustments", {}),
            )
        except Exception as e:
            log.warning("Failed to parse directive: %s", e)
            return WeeklyDirective(
                week_reviewed=week_ending,
                assessment={"overall": "Failed to parse Opus response"},
            )

    def _save_directive(self, directive: WeeklyDirective, week_ending: str) -> None:
        """Save the weekly directive to data/weekly_reviews/."""
        os.makedirs(REVIEWS_DIR, exist_ok=True)
        filename = f"{week_ending.replace(' ', '_').replace('/', '-')}.json"
        filepath = os.path.join(REVIEWS_DIR, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(directive.model_dump(mode="json"), f, indent=2, default=str)
            log.info("Saved weekly directive to %s", filepath)
        except Exception as e:
            log.error("Failed to save directive: %s", e)
