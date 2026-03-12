"""Per-Loss Post-Mortem Engine — Tier 1 (DeepSeek V3.2).

Triggers on every losing trade. Runs 5 parallel analysis dimensions
(data, sentiment, timing, model, risk), extracts prevention rules,
and feeds them into future Devil's Advocate challenges.

Prevention rules are persisted to data/prevention_rules.json and
injected into DA prompts via get_relevant_rules().
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from core.event_bus import event_bus
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.schemas import (
    DimensionFinding,
    PostMortemDimension,
    PostMortemFinding,
    PostMortemSeverity,
    PreventionRule,
)

log = setup_logger("trading.postmortem")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FINDINGS_FILE = os.path.join(DATA_DIR, "postmortem_findings.json")
RULES_FILE = os.path.join(DATA_DIR, "prevention_rules.json")

MAX_FINDINGS = 500
MAX_POSTMORTEMS_PER_DAY = 10
DIMENSIONS = ["data", "sentiment", "timing", "model", "risk"]

SYSTEM_PROMPT = """You are a post-mortem analyst for an algorithmic trading system. Your job is to analyze losing trades and extract actionable prevention rules to avoid similar losses in the future.

Be specific and actionable. Focus on what was KNOWABLE at the time of the trade, not hindsight bias. Prevention rules should be concrete enough to be programmatically checked or flagged by a devil's advocate agent."""

DIMENSION_QUESTIONS = {
    "data": (
        "DATA QUALITY ANALYSIS: Was the market data accurate at the time of entry? "
        "Were there data gaps, stale prices, or price lag that could have misled the analysis? "
        "Did the data sources agree, or were there conflicting signals that were ignored?"
    ),
    "sentiment": (
        "SENTIMENT ANALYSIS: Was the sentiment signal correct at the time of entry? "
        "Did sentiment flip after entry? Was the sentiment reading based on stale news? "
        "Was the market already pricing in the sentiment before the trade was placed?"
    ),
    "timing": (
        "TIMING ANALYSIS: Was the entry/exit timing optimal? Could we have entered later "
        "at a better price? Should we have exited earlier to reduce losses? "
        "Was this a FOMO entry after the move had already happened?"
    ),
    "model": (
        "MODEL/CONFIDENCE ANALYSIS: Was the confidence level well-calibrated for this trade? "
        "Was the thesis sound, or was it based on weak assumptions? "
        "Did the confirming signals actually support the thesis, or were they cherry-picked?"
    ),
    "risk": (
        "RISK MANAGEMENT ANALYSIS: Was the position size appropriate for the risk? "
        "Was the stop-loss too tight (stopped out prematurely) or too loose (let losses run)? "
        "Was the risk-reward ratio adequate? Did correlation with other positions amplify losses?"
    ),
}


class PostMortemEngine:
    """Runs 5-dimension post-mortem on losing trades and extracts prevention rules."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        telegram: Any = None,
    ):
        self._llm = llm_client or LLMClient()
        self._telegram = telegram
        self._daily_count = 0
        self._daily_reset_date: str = ""

    def run_postmortem(self, trade_record: dict) -> list[dict]:
        """Run 5-dimension post-mortem on a losing trade.

        Returns list of dimension findings. Respects daily limit.
        Never raises — all errors are caught and logged.
        """
        # Reset daily counter if new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_count = 0
            self._daily_reset_date = today

        # Guard: daily limit
        if self._daily_count >= MAX_POSTMORTEMS_PER_DAY:
            log.info("Post-mortem daily limit reached (%d/%d) — skipping",
                     self._daily_count, MAX_POSTMORTEMS_PER_DAY)
            return []

        self._daily_count += 1

        # Build trade context
        trade_id = trade_record.get("trade_id", "unknown")
        outcome = trade_record.get("outcome", {})
        context = {
            "trade_id": trade_id,
            "asset": trade_record.get("asset", ""),
            "direction": trade_record.get("direction", ""),
            "entry_price": trade_record.get("entry_price", 0),
            "exit_price": trade_record.get("exit_price", 0),
            "pnl_usd": outcome.get("pnl_usd", 0),
            "pnl_pct": outcome.get("pnl_pct", 0),
            "exit_reason": outcome.get("exit_reason", ""),
            "hold_duration_hours": outcome.get("hold_duration_hours", 0),
            "mae_pct": outcome.get("mae_pct", 0),
            "mfe_pct": outcome.get("mfe_pct", 0),
            "thesis_summary": trade_record.get("thesis_summary", ""),
            "thesis_confidence_original": trade_record.get("thesis_confidence_original", 0),
            "thesis_confidence_after_devil": trade_record.get("thesis_confidence_after_devil", 0),
            "stop_loss_price": trade_record.get("stop_loss_price"),
            "take_profit_price": trade_record.get("take_profit_price"),
        }

        # Run 5 dimensions in parallel
        dimension_findings: list[dict] = []
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {
                    pool.submit(self._analyze_dimension, dim, context): dim
                    for dim in DIMENSIONS
                }
                for future in as_completed(futures):
                    dim = futures[future]
                    try:
                        finding = future.result()
                        dimension_findings.append(finding)
                    except Exception as e:
                        log.error("Post-mortem dimension %s failed: %s", dim, e)
                        dimension_findings.append({
                            "dimension": dim,
                            "finding": f"Analysis failed: {e}",
                            "severity": "low",
                            "prevention_rule": "",
                            "confidence": 0.0,
                        })
        except Exception as e:
            log.error("Post-mortem parallel execution failed: %s", e)
            return []

        # Build full finding
        finding = PostMortemFinding(
            trade_id=trade_id,
            asset=context["asset"],
            exit_reason=context["exit_reason"],
            pnl_pct=context["pnl_pct"],
            dimensions=[
                DimensionFinding(
                    dimension=PostMortemDimension(d["dimension"]),
                    finding=d["finding"],
                    severity=PostMortemSeverity(d.get("severity", "low")),
                    prevention_rule=d.get("prevention_rule", ""),
                    confidence=max(0.0, min(1.0, float(d.get("confidence", 0.0)))),
                )
                for d in dimension_findings
                if d.get("dimension") in [e.value for e in PostMortemDimension]
            ],
        )

        finding_dict = finding.model_dump(mode="json")

        # Persist and extract rules
        try:
            self._persist_findings(finding_dict)
        except Exception as e:
            log.error("Post-mortem persist failed: %s", e)

        try:
            self._extract_prevention_rules(finding_dict)
        except Exception as e:
            log.error("Prevention rule extraction failed: %s", e)

        # Send Telegram summary
        try:
            self._send_telegram_summary(finding_dict)
        except Exception as e:
            log.error("Post-mortem Telegram alert failed: %s", e)

        # Emit event
        try:
            event_bus.emit("postmortem", "analysis_complete", {
                "trade_id": trade_id,
                "asset": context["asset"],
                "pnl_pct": context["pnl_pct"],
                "dimensions_analyzed": len(dimension_findings),
            })
        except Exception:
            pass

        log.info("Post-mortem complete for %s (%s): %d dimensions analyzed",
                 trade_id, context["asset"], len(dimension_findings))

        return dimension_findings

    def _analyze_dimension(self, dimension: str, context: dict) -> dict:
        """Analyze a single dimension via DeepSeek LLM call."""
        prompt = self._build_prompt(dimension, context)

        try:
            result = self._llm.call_deepseek(prompt, SYSTEM_PROMPT)
        except Exception as e:
            log.error("LLM call failed for dimension %s: %s", dimension, e)
            return {
                "dimension": dimension,
                "finding": f"LLM call failed: {e}",
                "severity": "low",
                "prevention_rule": "",
                "confidence": 0.0,
            }

        if isinstance(result, dict) and "error" in result:
            return {
                "dimension": dimension,
                "finding": f"LLM returned error: {result['error']}",
                "severity": "low",
                "prevention_rule": "",
                "confidence": 0.0,
            }

        # Parse and validate response
        finding_text = result.get("finding", "")
        severity = result.get("severity", "low")
        if severity not in ("low", "medium", "high"):
            severity = "low"
        prevention_rule = result.get("prevention_rule", "")
        confidence = result.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (ValueError, TypeError):
            confidence = 0.0

        return {
            "dimension": dimension,
            "finding": finding_text if finding_text else "No specific finding",
            "severity": severity,
            "prevention_rule": prevention_rule,
            "confidence": confidence,
        }

    def _build_prompt(self, dimension: str, context: dict) -> str:
        """Build dimension-specific post-mortem prompt."""
        header = (
            f"Analyze this LOSING trade:\n"
            f"Asset: {context['asset']}\n"
            f"Direction: {context['direction']}\n"
            f"Entry Price: ${context['entry_price']}\n"
            f"Exit Price: ${context['exit_price']}\n"
            f"P&L: ${context['pnl_usd']} ({context['pnl_pct']}%)\n"
            f"Exit Reason: {context['exit_reason']}\n"
            f"Hold Duration: {context['hold_duration_hours']}h\n"
            f"MAE (max adverse excursion): {context['mae_pct']}%\n"
            f"MFE (max favorable excursion): {context['mfe_pct']}%\n"
            f"Thesis: {context['thesis_summary']}\n"
            f"Original Confidence: {context['thesis_confidence_original']}\n"
            f"Post-DA Confidence: {context['thesis_confidence_after_devil']}\n"
            f"Stop Loss: {context.get('stop_loss_price', 'N/A')}\n"
            f"Take Profit: {context.get('take_profit_price', 'N/A')}\n"
        )

        dimension_question = DIMENSION_QUESTIONS.get(dimension, "Analyze this dimension.")

        return (
            f"You MUST respond with ONLY a valid JSON object. No explanation, no markdown, "
            f"no text before or after the JSON.\n\n"
            f"{header}\n"
            f"{dimension_question}\n\n"
            f"Return JSON:\n"
            f'{{"finding": "1-2 sentence analysis of what went wrong in this dimension", '
            f'"severity": "low|medium|high", '
            f'"prevention_rule": "A specific, actionable rule to prevent this type of loss in future trades", '
            f'"confidence": 0.0-1.0}}'
        )

    def _persist_findings(self, finding: dict) -> None:
        """Append finding to postmortem_findings.json with ring buffer."""
        os.makedirs(DATA_DIR, exist_ok=True)
        findings = self._load_findings()
        findings.append(finding)

        # Ring buffer: keep last MAX_FINDINGS
        if len(findings) > MAX_FINDINGS:
            findings = findings[-MAX_FINDINGS:]

        # Atomic write
        tmp_file = FINDINGS_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(findings, f, indent=2, default=str)
        os.replace(tmp_file, FINDINGS_FILE)

    def _extract_prevention_rules(self, finding: dict) -> None:
        """Extract prevention rules from high-confidence dimension findings."""
        rules = self._load_rules()
        trade_id = finding.get("trade_id", "")
        asset = finding.get("asset", "ALL")

        for dim_finding in finding.get("dimensions", []):
            confidence = dim_finding.get("confidence", 0.0)
            if confidence < 0.6:
                continue

            prevention_rule = dim_finding.get("prevention_rule", "")
            if not prevention_rule:
                continue

            dimension = dim_finding.get("dimension", "")

            # Check for duplicate: same dimension + asset + >70% word overlap
            matched = False
            for existing in rules:
                if (existing.get("dimension") == dimension
                        and existing.get("asset_pattern") in (asset, "ALL")
                        and self._word_overlap(existing.get("rule", ""), prevention_rule) > 0.7):
                    # Merge: add trade_id to source list
                    if trade_id not in existing.get("source_trade_ids", []):
                        existing.setdefault("source_trade_ids", []).append(trade_id)
                    existing["times_matched"] = existing.get("times_matched", 0) + 1
                    matched = True
                    break

            if not matched:
                new_rule = PreventionRule(
                    rule_id=f"rule_{uuid.uuid4().hex[:8]}",
                    rule=prevention_rule,
                    dimension=PostMortemDimension(dimension),
                    source_trade_ids=[trade_id],
                    asset_pattern=asset,
                )
                rules.append(new_rule.model_dump(mode="json"))

        # Persist rules
        tmp_file = RULES_FILE + ".tmp"
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(tmp_file, "w") as f:
            json.dump(rules, f, indent=2, default=str)
        os.replace(tmp_file, RULES_FILE)

    def _word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word-level Jaccard overlap between two strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get_relevant_rules(self, asset: str, limit: int = 5) -> list[str]:
        """Get prevention rules relevant to the given asset.

        Returns rule text strings, sorted by times_matched (most validated first).
        """
        rules = self._load_rules()
        relevant = [
            r for r in rules
            if r.get("active", True)
            and (r.get("asset_pattern") == asset or r.get("asset_pattern") == "ALL")
        ]
        # Sort by times_matched descending
        relevant.sort(key=lambda r: r.get("times_matched", 0), reverse=True)
        return [r.get("rule", "") for r in relevant[:limit] if r.get("rule")]

    def get_recent_findings(self, limit: int = 20) -> list[dict]:
        """Return the most recent post-mortem findings."""
        findings = self._load_findings()
        return findings[-limit:] if findings else []

    def _send_telegram_summary(self, finding: dict) -> None:
        """Send a Telegram summary of the post-mortem."""
        if not self._telegram:
            return

        trade_id = finding.get("trade_id", "?")
        asset = finding.get("asset", "?")
        pnl_pct = finding.get("pnl_pct", 0)
        dims = finding.get("dimensions", [])

        high_severity = [d for d in dims if d.get("severity") == "high"]
        medium_severity = [d for d in dims if d.get("severity") == "medium"]

        msg = (
            f"POST-MORTEM: {asset} (trade {trade_id})\n"
            f"P&L: {pnl_pct:.1f}%\n"
            f"Findings: {len(high_severity)} high, {len(medium_severity)} medium severity\n"
        )

        for d in high_severity[:3]:
            msg += f"  [{d['dimension'].upper()}] {d.get('finding', '')[:80]}\n"

        self._telegram.send_alert(msg)

    def _load_findings(self) -> list[dict]:
        """Load post-mortem findings from disk."""
        if not os.path.exists(FINDINGS_FILE):
            return []
        try:
            with open(FINDINGS_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _load_rules(self) -> list[dict]:
        """Load prevention rules from disk."""
        if not os.path.exists(RULES_FILE):
            return []
        try:
            with open(RULES_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
