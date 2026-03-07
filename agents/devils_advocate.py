"""Devil's Advocate Agent — Tier 1 (DeepSeek).

Contrarian risk challenger using 7-dimension challenge framework.
Outputs APPROVED, APPROVED_WITH_MODIFICATION, or KILLED verdict.
See TRADING_AGENT_PRD.md Section 3.3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.schemas import (
    AssumptionStressTest,
    BaseRateCheck,
    CorrelationTrap,
    CrowdedTradeCheck,
    DevilsChallenges,
    DevilsVerdict,
    SecondOrderEffects,
    SizeCheck,
    TimingCheck,
    TradeThesis,
    Verdict,
)
from tools.correlation import CorrelationAnalyzer
from tools.market_data import MarketDataFetcher

log = setup_logger("trading.devils_advocate")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
PARAMS_FILE = os.path.join(CONFIG_DIR, "devils_advocate_params.json")

SYSTEM_PROMPT = """You are "The Risk Adjuster" — a former risk manager at Citadel who calibrated position sizes across 8 profitable years. You believe every thesis deserves a hearing, but the SIZE must match the conviction.

Your methodology:
1. ASSUME THE THESIS IS WRONG — then size accordingly
2. IDENTIFY THE CROWDED TRADE — reduce size, don't kill
3. STRESS TEST the key assumption — adjust confidence
4. CHECK THE BASE RATE — inform sizing
5. FIND SECOND-ORDER EFFECTS — flag but don't auto-kill

Three outputs: APPROVED | APPROVED_WITH_MODIFICATION | KILLED
You kill trades ONLY for structural reasons (near daily loss limit, zero confirming signals, no invalidation level).
Most trades get APPROVED_WITH_MODIFICATION with adjusted size.
During paper trading, data > protection."""

CHALLENGE_PROMPT = """You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no text before or after the JSON.

Challenge this trade thesis on ALL 7 dimensions.

Thesis:
{thesis}

Portfolio State:
{portfolio_state}

For each dimension, provide your analysis:

1. CROWDED TRADE CHECK — Score 0-1. Is everyone already positioned this way?
2. ASSUMPTION STRESS TEST — What's the critical assumption? What if it fails? Worst case loss?
3. TIMING CHALLENGE — Is this FOMO or genuine entry? Has the move already happened?
4. CORRELATION TRAP — Does this trade add risk or diversify? Post-trade concentration?
5. HISTORICAL BASE RATE — What % of similar setups historically profited? Sample size?
6. SECOND-ORDER EFFECTS — Non-obvious consequences that could reverse the trade?
7. SIZE APPROPRIATENESS — Is the proposed size right for the confidence level?
   (0.4-0.5 confidence = 2%, 0.5-0.65 = 4%, 0.65+ = 7%)
   Note: there are now 4 confirming signals (fundamental, technical, cross-asset, chart pattern)

Return JSON:
{{
  "challenges": {{
    "crowded_trade": {{"score": 0.0-1.0, "reasoning": "..."}},
    "assumption_stress_test": {{"critical_assumption": "...", "if_wrong": "...", "worst_case_loss_usd": 0.0}},
    "timing": {{"flag": true/false, "reasoning": "..."}},
    "correlation_trap": {{"flag": true/false, "post_trade_concentration": 0.0}},
    "base_rate": {{"available": true/false, "historical_win_rate": 0.0, "sample_size": 0}},
    "second_order_effects": {{"identified": ["..."], "severity": "low|medium|high"}},
    "size_check": {{"proposed_pct": 0.0, "recommended_pct": 0.0}}
  }},
  "modifications": ["list of suggested changes"],
  "confidence_adjusted": 0.0-1.0,
  "final_reasoning": "1-2 sentence summary"
}}"""


class DevilsAdvocate:
    """Contrarian risk challenger & pre-trade stress tester."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        correlation: CorrelationAnalyzer | None = None,
        market_data: MarketDataFetcher | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._corr = correlation or CorrelationAnalyzer()
        self._mdf = market_data or MarketDataFetcher()
        self._params = self._load_params()

    def challenge(
        self, trade_thesis: TradeThesis, portfolio_state: dict[str, Any]
    ) -> DevilsVerdict:
        """Challenge a trade thesis on all 7 dimensions."""
        thesis_id = str(id(trade_thesis))

        # Step 1: Check fatal flaws (pure Python, no LLM)
        fatal_flaws = self.check_fatal_flaws(trade_thesis, portfolio_state)
        if fatal_flaws:
            log.info("Fatal flaws found — killing trade: %s", fatal_flaws)
            return DevilsVerdict(
                original_thesis_id=thesis_id,
                verdict=Verdict.KILLED,
                flags_raised=len(fatal_flaws),
                fatal_flaws=fatal_flaws,
                confidence_adjusted=0.0,
                final_reasoning=f"Fatal flaws: {'; '.join(fatal_flaws)}",
            )

        # Check for duplicate asset warning (non-fatal)
        dup_warning = self._duplicate_asset_warning(trade_thesis, portfolio_state)

        # Step 2: Send to DeepSeek for 7-challenge analysis (was Kimi, swapped for cost)
        prompt = CHALLENGE_PROMPT.format(
            thesis=json.dumps(trade_thesis.model_dump(), default=str),
            portfolio_state=json.dumps(portfolio_state, default=str),
        )

        # Append phantom trade feedback (false kill rate calibration)
        phantom_context = self._load_phantom_context()
        if phantom_context:
            prompt += f"\n\n{phantom_context}"
        result = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)

        if result.get("error"):
            log.warning("DeepSeek challenge failed: %s — defaulting to APPROVED_WITH_MODIFICATION", result["error"])
            return DevilsVerdict(
                original_thesis_id=thesis_id,
                verdict=Verdict.APPROVED_WITH_MODIFICATION,
                flags_raised=1,
                modifications=["LLM unavailable — reduce position size by 50%"],
                confidence_adjusted=trade_thesis.confidence * 0.5,
                final_reasoning="Could not perform full challenge — conservative approval",
            )

        # Step 3: Parse challenges
        challenges = self._parse_challenges(result)

        # Step 4: Count flags
        flags = self._count_flags(challenges)

        # Step 5: Determine verdict — prefer downsizing over killing
        confidence_adjusted = float(result.get("confidence_adjusted", trade_thesis.confidence * 0.9))
        modifications = result.get("modifications", [])

        # Duplicate asset is a fatal flaw — prevents position stacking
        if dup_warning:
            log.info("Fatal flaw: %s", dup_warning)
            return DevilsVerdict(
                original_thesis_id=thesis_id,
                verdict=Verdict.KILLED,
                challenges=challenges,
                flags_raised=flags + 1,
                fatal_flaws=[dup_warning],
                confidence_adjusted=0.0,
                final_reasoning=f"Duplicate asset blocked: {dup_warning}",
            )

        kill_threshold = self._params.get("kill_thresholds", {}).get(
            "min_challenges_for_kill", 3
        )

        if flags >= kill_threshold:
            # Even at kill threshold, downgrade to micro position instead of full kill
            verdict = Verdict.APPROVED_WITH_MODIFICATION
            confidence_adjusted = min(confidence_adjusted, 0.42)  # Forces 2% micro position
            modifications.append(f"High flag count ({flags}) — downsized to micro position")
            log.info("Devil's verdict: DOWNGRADED to micro (was KILL with %d flags)", flags)
        elif flags >= 2:
            verdict = Verdict.APPROVED_WITH_MODIFICATION
            # Scale down confidence proportionally to flags
            flag_penalty = flags * 0.05
            confidence_adjusted = max(0.40, confidence_adjusted - flag_penalty)
            log.info("Devil's verdict: APPROVED_WITH_MODIFICATION (%d flags, adj conf %.2f)", flags, confidence_adjusted)
        else:
            verdict = Verdict.APPROVED
            log.info("Devil's verdict: APPROVED (%d flags)", flags)

        return DevilsVerdict(
            original_thesis_id=thesis_id,
            verdict=verdict,
            challenges=challenges,
            flags_raised=flags,
            fatal_flaws=[],
            modifications=modifications,
            confidence_adjusted=confidence_adjusted,
            final_reasoning=result.get("final_reasoning", ""),
        )

    def check_fatal_flaws(
        self, thesis: TradeThesis, portfolio_state: dict[str, Any]
    ) -> list[str]:
        """Check for auto-kill fatal flaws (pure Python).

        Note: duplicate asset is NOT a fatal flaw — the signal could be
        an exit or reversal for a held position. It is tracked as a
        warning flag instead (see ``_duplicate_asset_warning``).
        """
        flaws: list[str] = []

        # 1. Position would breach daily loss limit
        daily_pnl = portfolio_state.get("daily_pnl_pct", 0.0)
        if daily_pnl <= -4.0:  # Within 1% of 5% limit
            flaws.append("Position would risk breaching daily loss limit")

        # 2. No invalidation level defined
        if not thesis.invalidation_level:
            flaws.append("No invalidation level defined")

        # 3. Thesis has zero confirming signals (out of 4: fundamental, technical, cross_asset, chart_pattern)
        signals = thesis.confirming_signals
        confirmed_count = sum([
            signals.fundamental.present,
            signals.technical.present,
            signals.cross_asset.present,
            signals.chart_pattern.present,
        ])
        if confirmed_count < 1:
            flaws.append("Zero confirming signals")

        return flaws

    def _duplicate_asset_warning(
        self, thesis: TradeThesis, portfolio_state: dict[str, Any]
    ) -> str | None:
        """Return a warning string if the asset already has an open position.

        This is intentionally NOT a fatal flaw — the signal could be an
        exit or reversal for the held position.
        """
        existing = portfolio_state.get("open_positions", [])
        for pos in existing:
            if pos.get("asset") == thesis.asset:
                return f"Duplicate asset {thesis.asset} already in portfolio — review for exit/reversal"
        return None

    def _parse_challenges(self, result: dict[str, Any]) -> DevilsChallenges:
        """Parse LLM result into DevilsChallenges model."""
        ch = result.get("challenges", {})
        try:
            return DevilsChallenges(
                crowded_trade=CrowdedTradeCheck(**ch.get("crowded_trade", {})),
                assumption_stress_test=AssumptionStressTest(**ch.get("assumption_stress_test", {})),
                timing=TimingCheck(**ch.get("timing", {})),
                correlation_trap=CorrelationTrap(**ch.get("correlation_trap", {})),
                base_rate=BaseRateCheck(**ch.get("base_rate", {})),
                second_order_effects=SecondOrderEffects(**ch.get("second_order_effects", {})),
                size_check=SizeCheck(**ch.get("size_check", {})),
            )
        except Exception as e:
            log.warning("Failed to parse challenges: %s", e)
            return DevilsChallenges()

    def _count_flags(self, challenges: DevilsChallenges) -> int:
        """Count how many of the 7 challenges were flagged."""
        flags = 0
        if challenges.crowded_trade.score >= 0.6:
            flags += 1
        if challenges.timing.flag:
            flags += 1
        if challenges.correlation_trap.flag:
            flags += 1
        if challenges.base_rate.available and challenges.base_rate.historical_win_rate < 0.5:
            flags += 1
        if challenges.second_order_effects.severity in ("medium", "high"):
            flags += 1
        if challenges.size_check.recommended_pct < challenges.size_check.proposed_pct * 0.7:
            flags += 1
        if challenges.assumption_stress_test.worst_case_loss_usd > 5.0:
            flags += 1
        return flags

    def _load_phantom_context(self) -> str:
        """Load phantom trade summary and return calibration context for the prompt."""
        phantom_file = Path(__file__).parent.parent / "data" / "phantom_trades.json"
        try:
            if not phantom_file.exists():
                return ""
            with open(phantom_file) as f:
                trades = json.load(f)
            checked = [t for t in trades if t.get("outcome_checked")]
            if len(checked) < 3:
                return ""
            won = sum(1 for t in checked if (t.get("outcome_pnl_pct") or 0) > 0)
            lost = len(checked) - won
            false_kill_rate = round(won / len(checked) * 100, 1)
            context = f"Phantom trade data: {len(checked)} killed trades checked — {won} would have won, {lost} would have lost (false kill rate: {false_kill_rate}%)."
            if false_kill_rate > 40:
                context += " You have been killing too many profitable trades. Be less aggressive."
            elif false_kill_rate < 20:
                context += " Your kill accuracy is excellent. Maintain current standards."
            return context
        except Exception:
            return ""

    def _load_params(self) -> dict[str, Any]:
        try:
            with open(PARAMS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
