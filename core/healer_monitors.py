"""Self-healing agent monitors — 6 monitors for automated system health.

Each monitor inherits from BaseMonitor ABC and implements:
  - check(context) -> MonitorResult: detect issues from context data
  - auto_respond(result, context) -> str: generate human-readable response
  - investigate(result, context, llm_client) -> list[dict]: deep-dive analysis

All monitors are Tier 0 (pure Python) except ThesisFailureMonitor and
PositionSaturationMonitor which use LLM only in investigate() when explicitly called.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any

from core.logger import setup_logger
from core.schemas import MonitorSeverity, MonitorResult, InvestigationFinding

log = setup_logger("trading.healer.monitors")


# ── Base class ────────────────────────────────────────────────────────


class BaseMonitor(ABC):
    """Abstract base for all self-healing monitors."""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self._last_investigation: float = 0  # monotonic timestamp

    @property
    def enabled(self) -> bool:
        return self.params.get("enabled", True)

    def can_investigate(self) -> bool:
        cooldown_h = self.params.get("investigation_cooldown_hours", 6)
        return (time.monotonic() - self._last_investigation) > cooldown_h * 3600

    def mark_investigated(self):
        self._last_investigation = time.monotonic()

    @abstractmethod
    def check(self, context: dict) -> MonitorResult: ...

    @abstractmethod
    def auto_respond(self, result: MonitorResult, context: dict) -> str: ...

    def investigate(
        self, result: MonitorResult, context: dict, llm_client=None
    ) -> list:
        """Override in monitors that need deep investigation."""
        return []


# ── Helpers ───────────────────────────────────────────────────────────


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp string to a timezone-aware datetime, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hours_since(ts: str | None) -> float | None:
    """Return hours elapsed since an ISO timestamp, or None if unparseable."""
    dt = _parse_iso(ts)
    if dt is None:
        return None
    delta = _now_utc() - dt
    return delta.total_seconds() / 3600


def _safe_result(
    monitor_name: str,
    triggered: bool,
    severity: MonitorSeverity,
    details: dict,
    auto_response: str = "",
) -> MonitorResult:
    """Build a MonitorResult safely."""
    return MonitorResult(
        monitor_name=monitor_name,
        triggered=triggered,
        severity=severity,
        details=details,
        auto_response=auto_response,
    )


# ── 1. FeedHealthMonitor ─────────────────────────────────────────────


class FeedHealthMonitor(BaseMonitor):
    """Detects dead or stale news feeds.

    Context keys:
        feed_health: dict of {feed_name: {"consecutive_failures": int,
                     "last_new_article": iso_str|None, "url": str|None}}
        feed_response_times: optional list of {feed_name, response_ms, timestamp}
    """

    def __init__(self, params: dict | None = None):
        super().__init__("feed_health", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            feed_health = context.get("feed_health", {})
            failure_threshold = self.params.get("failure_threshold", 3)
            stale_hours = self.params.get("stale_hours", 24)

            dead_feeds: list[str] = []
            stale_feeds: list[str] = []
            details: dict[str, Any] = {}

            for feed_name, info in feed_health.items():
                if not isinstance(info, dict):
                    continue
                failures = info.get("consecutive_failures", 0)
                last_article = info.get("last_new_article")

                if failures >= failure_threshold:
                    dead_feeds.append(feed_name)

                hours = _hours_since(last_article)
                if hours is not None and hours > stale_hours:
                    stale_feeds.append(feed_name)
                elif last_article is None and failures < failure_threshold:
                    # Never produced an article but not technically dead
                    stale_feeds.append(feed_name)

            # Trend detection: check if response times are degrading
            degrading_feeds: list[str] = []
            response_times = context.get("feed_response_times", [])
            if response_times:
                degrading_feeds = self._detect_response_degradation(response_times)

            triggered = bool(dead_feeds or stale_feeds)
            severity = MonitorSeverity.CRITICAL if dead_feeds else (
                MonitorSeverity.WARNING if stale_feeds else MonitorSeverity.INFO
            )

            details = {
                "dead_feeds": dead_feeds,
                "stale_feeds": stale_feeds,
                "degrading_feeds": degrading_feeds,
                "total_feeds": len(feed_health),
                "healthy_feeds": len(feed_health) - len(dead_feeds) - len(stale_feeds),
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("FeedHealthMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            dead = result.details.get("dead_feeds", [])
            stale = result.details.get("stale_feeds", [])
            degrading = result.details.get("degrading_feeds", [])
            parts: list[str] = []

            if dead:
                parts.append(
                    f"DEAD feeds ({len(dead)}): {', '.join(dead)}. "
                    "These have exceeded the consecutive failure threshold."
                )
            if stale:
                stale_hours = self.params.get("stale_hours", 24)
                parts.append(
                    f"STALE feeds ({len(stale)}): {', '.join(stale)}. "
                    f"No new articles in >{stale_hours}h."
                )
            if degrading:
                parts.append(
                    f"DEGRADING response times: {', '.join(degrading)}. "
                    "Latency trending upward."
                )

            if not parts:
                return "All feeds healthy."

            healthy = result.details.get("healthy_feeds", 0)
            total = result.details.get("total_feeds", 0)
            parts.append(f"Healthy: {healthy}/{total} feeds.")
            return " ".join(parts)

        except Exception as exc:
            log.error("FeedHealthMonitor.auto_respond() error: %s", exc)
            return f"Feed health check error: {exc}"

    def investigate(
        self, result: MonitorResult, context: dict, llm_client=None
    ) -> list:
        """Pure Python: HTTP HEAD check on dead feed URLs."""
        findings: list[dict] = []
        try:
            import requests  # noqa: delayed import — only when investigating

            feed_health = context.get("feed_health", {})
            dead_feeds = result.details.get("dead_feeds", [])

            for feed_name in dead_feeds:
                info = feed_health.get(feed_name, {})
                url = info.get("url")
                if not url:
                    findings.append(InvestigationFinding(
                        dimension="tech",
                        finding=f"Feed '{feed_name}' has no URL configured",
                        root_cause="Missing feed URL in configuration",
                        recommendation=f"Add URL for feed '{feed_name}' or remove it from feed list",
                        confidence=0.9,
                    ).model_dump())
                    continue

                try:
                    resp = requests.head(url, timeout=10, allow_redirects=True)
                    status = resp.status_code
                    if status >= 400:
                        findings.append(InvestigationFinding(
                            dimension="tech",
                            finding=f"Feed '{feed_name}' returned HTTP {status}",
                            root_cause=f"Server at {url} returning {status}",
                            recommendation=(
                                f"HTTP {status}: "
                                + ("URL not found — check if feed moved"
                                   if status == 404
                                   else "Rate limited — add delay or rotate IP"
                                   if status == 429
                                   else "Server error — may be temporary, retry later"
                                   if status >= 500
                                   else f"Client error {status} — check URL and auth")
                            ),
                            confidence=0.85,
                        ).model_dump())
                    else:
                        findings.append(InvestigationFinding(
                            dimension="tech",
                            finding=(
                                f"Feed '{feed_name}' responds HTTP {status} "
                                "but parser may be failing"
                            ),
                            root_cause="Feed is reachable but content parsing may have changed",
                            recommendation="Check feed content format — RSS/HTML structure may have changed",
                            confidence=0.6,
                        ).model_dump())
                except requests.RequestException as req_exc:
                    findings.append(InvestigationFinding(
                        dimension="tech",
                        finding=f"Feed '{feed_name}' unreachable: {req_exc}",
                        root_cause=f"Network error connecting to {url}",
                        recommendation="Check DNS resolution, network connectivity, or if site is blocked",
                        confidence=0.8,
                    ).model_dump())

        except ImportError:
            log.warning("requests library not available for feed investigation")
        except Exception as exc:
            log.error("FeedHealthMonitor.investigate() error: %s", exc, exc_info=True)

        return findings

    def _detect_response_degradation(
        self, response_times: list[dict]
    ) -> list[str]:
        """Detect feeds whose response times are trending upward."""
        degrading: list[str] = []
        try:
            # Group by feed
            by_feed: dict[str, list[float]] = {}
            for entry in response_times:
                name = entry.get("feed_name", "")
                ms = entry.get("response_ms", 0)
                if name and ms > 0:
                    by_feed.setdefault(name, []).append(ms)

            for feed_name, times in by_feed.items():
                if len(times) < 4:
                    continue
                # Compare first half average to second half average
                mid = len(times) // 2
                first_avg = sum(times[:mid]) / mid
                second_avg = sum(times[mid:]) / (len(times) - mid)
                if first_avg > 0 and second_avg / first_avg > 1.5:
                    degrading.append(feed_name)
        except Exception as exc:
            log.error("Response degradation detection error: %s", exc)

        return degrading


# ── 2. ThesisFailureMonitor ──────────────────────────────────────────


class ThesisFailureMonitor(BaseMonitor):
    """Detects high thesis-build failure rates in the signal pipeline.

    Context keys:
        funnel_stats: {signals_generated, analyst_errors, analyst_no_trade, ...}
        thesis_errors: list of recent error strings
        thesis_failure_trend: optional list of {failure_rate, timestamp}
    """

    def __init__(self, params: dict | None = None):
        super().__init__("thesis_failure", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            stats = context.get("funnel_stats", {})
            signals = stats.get("signals_generated", 0)
            errors = stats.get("analyst_errors", 0)
            min_sample = self.params.get("min_sample", 4)
            threshold = self.params.get("failure_rate_threshold", 0.50)

            failure_rate = errors / max(signals, 1)
            triggered = failure_rate > threshold and signals >= min_sample

            # Trend detection
            trend_direction = "stable"
            trend_data = context.get("thesis_failure_trend", [])
            if len(trend_data) >= 3:
                rates = [t.get("failure_rate", 0) for t in trend_data]
                recent = rates[-3:]
                if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                    trend_direction = "worsening"
                elif all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                    trend_direction = "improving"

            severity = MonitorSeverity.INFO
            if triggered:
                severity = (
                    MonitorSeverity.CRITICAL if failure_rate > 0.75
                    else MonitorSeverity.WARNING
                )
            elif trend_direction == "worsening":
                severity = MonitorSeverity.WARNING

            details = {
                "failure_rate": round(failure_rate, 3),
                "signals_generated": signals,
                "analyst_errors": errors,
                "threshold": threshold,
                "min_sample": min_sample,
                "trend": trend_direction,
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("ThesisFailureMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            rate = result.details.get("failure_rate", 0)
            signals = result.details.get("signals_generated", 0)
            errors = result.details.get("analyst_errors", 0)
            trend = result.details.get("trend", "stable")

            recent_errors = context.get("thesis_errors", [])
            error_summary = ""
            if recent_errors:
                # Show last 3 unique errors
                unique = list(dict.fromkeys(recent_errors))[-3:]
                error_summary = " Recent errors: " + "; ".join(unique)

            msg = (
                f"Thesis failure rate: {rate:.0%} ({errors}/{signals} signals).{error_summary}"
            )
            if trend == "worsening":
                msg += " Trend: WORSENING — failure rate increasing over recent periods."
            elif trend == "improving":
                msg += " Trend: improving."

            return msg

        except Exception as exc:
            log.error("ThesisFailureMonitor.auto_respond() error: %s", exc)
            return f"Thesis failure check error: {exc}"

    def investigate(
        self, result: MonitorResult, context: dict, llm_client=None
    ) -> list:
        """Uses LLM to analyze thesis error patterns."""
        findings: list[dict] = []
        try:
            errors = context.get("thesis_errors", [])
            if not errors:
                return findings

            # Deduplicate and take last 10
            unique_errors = list(dict.fromkeys(errors))[-10:]

            # Technical finding from pattern analysis (pure Python)
            error_types: dict[str, int] = {}
            for err in errors:
                err_lower = err.lower()
                if "json" in err_lower or "parse" in err_lower:
                    error_types["json_parse"] = error_types.get("json_parse", 0) + 1
                elif "timeout" in err_lower:
                    error_types["timeout"] = error_types.get("timeout", 0) + 1
                elif "key" in err_lower or "field" in err_lower:
                    error_types["missing_field"] = error_types.get("missing_field", 0) + 1
                else:
                    error_types["other"] = error_types.get("other", 0) + 1

            dominant_type = max(error_types, key=error_types.get) if error_types else "unknown"
            recommendations = {
                "json_parse": "LLM is returning malformed JSON — tighten the prompt schema or add fallback parsing",
                "timeout": "LLM calls are timing out — check provider status or increase timeout",
                "missing_field": "LLM output missing required fields — add field-level defaults in parser",
                "other": "Mixed error types — review recent prompt changes",
                "unknown": "Unable to categorize errors — manual review needed",
            }

            findings.append(InvestigationFinding(
                dimension="tech",
                finding=f"Dominant error type: {dominant_type} ({error_types.get(dominant_type, 0)} occurrences)",
                root_cause=f"Pattern analysis shows {dominant_type} errors are most common",
                recommendation=recommendations.get(dominant_type, "Manual review needed"),
                confidence=0.7,
            ).model_dump())

            # LLM-powered deep analysis (only if client provided)
            if llm_client is not None:
                try:
                    error_text = "\n".join(f"- {e}" for e in unique_errors)
                    prompt = (
                        "These thesis build errors occurred in a trading signal pipeline:\n"
                        f"{error_text}\n\n"
                        "Identify the pattern and suggest a specific parsing fix. "
                        "Also estimate how many valid trading opportunities were likely missed. "
                        "Reply in JSON: {\"pattern\": str, \"fix\": str, \"missed_opportunities_estimate\": int, \"confidence\": float}"
                    )
                    resp = llm_client.call_deepseek(prompt, max_tokens=300)

                    # Try to parse LLM response as JSON
                    llm_data = {}
                    if resp:
                        try:
                            # Strip markdown code fences if present
                            clean = resp.strip()
                            if clean.startswith("```"):
                                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                            llm_data = json.loads(clean)
                        except (json.JSONDecodeError, IndexError):
                            llm_data = {"pattern": resp[:200], "fix": "See raw response"}

                    if llm_data.get("fix"):
                        findings.append(InvestigationFinding(
                            dimension="tech",
                            finding=f"LLM analysis: {llm_data.get('pattern', 'unknown pattern')}",
                            root_cause=llm_data.get("pattern", "Unknown"),
                            recommendation=llm_data.get("fix", "No fix suggested"),
                            confidence=float(llm_data.get("confidence", 0.5)),
                        ).model_dump())

                    missed = llm_data.get("missed_opportunities_estimate", 0)
                    if missed > 0:
                        findings.append(InvestigationFinding(
                            dimension="finance",
                            finding=f"Estimated {missed} trading opportunities missed due to thesis failures",
                            root_cause="Pipeline errors preventing valid signals from reaching execution",
                            recommendation="Fix thesis parsing to recover missed signal flow",
                            confidence=float(llm_data.get("confidence", 0.4)),
                        ).model_dump())

                except Exception as llm_exc:
                    log.warning("ThesisFailureMonitor LLM investigation failed: %s", llm_exc)

        except Exception as exc:
            log.error("ThesisFailureMonitor.investigate() error: %s", exc, exc_info=True)

        return findings


# ── 3. PositionSaturationMonitor ─────────────────────────────────────


class PositionSaturationMonitor(BaseMonitor):
    """Detects when the pipeline is generating signals but none convert to trades.

    Context keys:
        last_trade_timestamp: ISO string or None
        funnel_stats: {signals_generated, analyst_errors, analyst_no_trade,
                       devil_killed, risk_rejected, rr_rejected}
        saturation_trend: optional list of {hours_since_trade, timestamp}
    """

    # Funnel stage names in pipeline order
    FUNNEL_STAGES = [
        "analyst_errors",
        "analyst_no_trade",
        "devil_killed",
        "risk_rejected",
        "rr_rejected",
    ]

    def __init__(self, params: dict | None = None):
        super().__init__("position_saturation", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            last_trade_ts = context.get("last_trade_timestamp")
            stats = context.get("funnel_stats", {})
            signals = stats.get("signals_generated", 0)
            threshold_hours = self.params.get("no_trade_hours", 6)

            hours_since_trade = _hours_since(last_trade_ts)
            no_trade_too_long = (
                hours_since_trade is not None and hours_since_trade > threshold_hours
            ) or (last_trade_ts is None)

            triggered = no_trade_too_long and signals > 0

            # Identify bottleneck
            bottleneck = self._find_bottleneck(stats)

            # Trend detection
            trend_direction = "stable"
            trend_data = context.get("saturation_trend", [])
            if len(trend_data) >= 3:
                gaps = [t.get("hours_since_trade", 0) for t in trend_data]
                recent = gaps[-3:]
                if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                    trend_direction = "worsening"
                elif all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                    trend_direction = "improving"

            severity = MonitorSeverity.INFO
            if triggered:
                severity = (
                    MonitorSeverity.CRITICAL
                    if hours_since_trade is not None and hours_since_trade > threshold_hours * 3
                    else MonitorSeverity.WARNING
                )

            details = {
                "hours_since_trade": round(hours_since_trade, 1) if hours_since_trade is not None else None,
                "threshold_hours": threshold_hours,
                "signals_generated": signals,
                "bottleneck": bottleneck,
                "funnel_stats": {k: stats.get(k, 0) for k in self.FUNNEL_STAGES},
                "trend": trend_direction,
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("PositionSaturationMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            hours = result.details.get("hours_since_trade")
            bottleneck = result.details.get("bottleneck", "unknown")
            signals = result.details.get("signals_generated", 0)
            funnel = result.details.get("funnel_stats", {})
            trend = result.details.get("trend", "stable")

            hours_str = f"{hours:.1f}h" if hours is not None else "unknown"

            stage_labels = {
                "analyst_errors": "MarketAnalyst errors",
                "analyst_no_trade": "MarketAnalyst no-trade verdicts",
                "devil_killed": "DevilsAdvocate kills",
                "risk_rejected": "RiskManager rejections",
                "rr_rejected": "R:R ratio rejections",
            }

            bottleneck_label = stage_labels.get(bottleneck, bottleneck)
            bottleneck_count = funnel.get(bottleneck, 0)

            msg = (
                f"No trades for {hours_str} despite {signals} signals. "
                f"Bottleneck: {bottleneck_label} ({bottleneck_count} rejections)."
            )

            if trend == "worsening":
                msg += " Trend: WORSENING — trade gap increasing."

            # Add funnel breakdown
            active_stages = {k: v for k, v in funnel.items() if v > 0}
            if active_stages:
                breakdown = ", ".join(f"{k}={v}" for k, v in active_stages.items())
                msg += f" Funnel: {breakdown}."

            return msg

        except Exception as exc:
            log.error("PositionSaturationMonitor.auto_respond() error: %s", exc)
            return f"Position saturation check error: {exc}"

    def investigate(
        self, result: MonitorResult, context: dict, llm_client=None
    ) -> list:
        """Uses LLM to analyze why signals are not converting to trades."""
        findings: list[dict] = []
        try:
            funnel = result.details.get("funnel_stats", {})
            bottleneck = result.details.get("bottleneck", "unknown")
            signals = result.details.get("signals_generated", 0)

            # Pure Python: technical analysis of bottleneck
            if bottleneck == "risk_rejected":
                findings.append(InvestigationFinding(
                    dimension="tech",
                    finding="RiskManager is the primary bottleneck — rejecting most signals",
                    root_cause="Position limits, correlation caps, or daily loss limits may be too tight",
                    recommendation="Review risk_params.json: consider relaxing max_position_pct, correlation_threshold, or sector_concentration_limit",
                    confidence=0.7,
                ).model_dump())
            elif bottleneck == "devil_killed":
                findings.append(InvestigationFinding(
                    dimension="tech",
                    finding="DevilsAdvocate is killing most signals",
                    root_cause="Contrarian analysis may be too aggressive or confidence thresholds too high",
                    recommendation="Review DA kill rate in agent_params.json — consider lowering conviction_threshold",
                    confidence=0.65,
                ).model_dump())
            elif bottleneck == "analyst_errors":
                findings.append(InvestigationFinding(
                    dimension="tech",
                    finding="MarketAnalyst is erroring on most signals",
                    root_cause="LLM response parsing failures or market data API issues",
                    recommendation="Check MarketAnalyst error logs for JSON parse or API timeout patterns",
                    confidence=0.75,
                ).model_dump())
            elif bottleneck == "rr_rejected":
                findings.append(InvestigationFinding(
                    dimension="tech",
                    finding="R:R ratio enforcement is rejecting most signals",
                    root_cause="Current market conditions may not support 2:1 R:R",
                    recommendation="Consider temporarily lowering min_rr_ratio in risk_params.json or widening stop distances",
                    confidence=0.6,
                ).model_dump())

            # Finance finding: opportunity cost
            if signals > 0:
                conversion_rate = 1 - (sum(funnel.values()) / max(signals, 1))
                findings.append(InvestigationFinding(
                    dimension="finance",
                    finding=f"Signal conversion rate: {conversion_rate:.0%} ({signals} signals, 0 trades)",
                    root_cause=f"Pipeline bottleneck at {bottleneck} stage",
                    recommendation="Capital is idle — opportunity cost accruing while market moves",
                    confidence=0.8,
                ).model_dump())

            # LLM deep analysis
            if llm_client is not None:
                try:
                    funnel_text = json.dumps(funnel, indent=2)
                    prompt = (
                        "A trading system has generated signals but none converted to trades.\n"
                        f"Funnel stats: {funnel_text}\n"
                        f"Bottleneck stage: {bottleneck}\n"
                        f"Hours since last trade: {result.details.get('hours_since_trade', 'unknown')}\n\n"
                        "Analyze why signals are not converting. Consider: "
                        "1) Are risk limits too tight for current market? "
                        "2) Is the contrarian filter too aggressive? "
                        "3) Are there structural issues in the pipeline?\n"
                        "Reply in JSON: {\"diagnosis\": str, \"adjustment\": str, \"urgency\": str}"
                    )
                    resp = llm_client.call_deepseek(prompt, max_tokens=300)

                    if resp:
                        try:
                            clean = resp.strip()
                            if clean.startswith("```"):
                                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                            llm_data = json.loads(clean)
                        except (json.JSONDecodeError, IndexError):
                            llm_data = {"diagnosis": resp[:200]}

                        findings.append(InvestigationFinding(
                            dimension="finance",
                            finding=f"LLM diagnosis: {llm_data.get('diagnosis', 'unknown')}",
                            root_cause=llm_data.get("diagnosis", "Unknown"),
                            recommendation=llm_data.get("adjustment", "No adjustment suggested"),
                            confidence=0.55,
                        ).model_dump())

                except Exception as llm_exc:
                    log.warning("PositionSaturationMonitor LLM investigation failed: %s", llm_exc)

        except Exception as exc:
            log.error("PositionSaturationMonitor.investigate() error: %s", exc, exc_info=True)

        return findings

    def _find_bottleneck(self, stats: dict) -> str:
        """Identify which funnel stage has the highest rejection count."""
        max_stage = "unknown"
        max_count = 0
        for stage in self.FUNNEL_STAGES:
            count = stats.get(stage, 0)
            if count > max_count:
                max_count = count
                max_stage = stage
        return max_stage


# ── 4. SchedulerHealthMonitor ────────────────────────────────────────


class SchedulerHealthMonitor(BaseMonitor):
    """Detects overdue scheduled tasks.

    Context keys:
        task_last_runs: {task_name: iso_timestamp}
        task_durations: optional list of {task_name, duration_s, timestamp}
    """

    # Expected intervals in minutes
    EXPECTED_INTERVALS: dict[str, int] = {
        "heartbeat": 5,
        "news_scan": 15,
        "chart_scan": 360,
        "proactive_scan": 480,
    }

    def __init__(self, params: dict | None = None):
        super().__init__("scheduler_health", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            task_last_runs = context.get("task_last_runs", {})
            overdue_tasks: list[dict] = []
            healthy_tasks: list[str] = []

            intervals = {**self.EXPECTED_INTERVALS}
            # Allow overrides from params
            custom_intervals = self.params.get("expected_intervals", {})
            intervals.update(custom_intervals)

            overdue_multiplier = self.params.get("overdue_multiplier", 2.0)

            for task_name, expected_min in intervals.items():
                last_run = task_last_runs.get(task_name)
                if last_run is None:
                    overdue_tasks.append({
                        "task": task_name,
                        "expected_interval_min": expected_min,
                        "last_run": None,
                        "overdue_by_min": None,
                    })
                    continue

                hours = _hours_since(last_run)
                if hours is None:
                    continue

                minutes_since = hours * 60
                threshold_min = expected_min * overdue_multiplier
                if minutes_since > threshold_min:
                    overdue_tasks.append({
                        "task": task_name,
                        "expected_interval_min": expected_min,
                        "last_run": last_run,
                        "overdue_by_min": round(minutes_since - threshold_min, 1),
                    })
                else:
                    healthy_tasks.append(task_name)

            # Trend: detect tasks whose execution time is growing
            slow_tasks: list[str] = []
            task_durations = context.get("task_durations", [])
            if task_durations:
                slow_tasks = self._detect_slowing_tasks(task_durations)

            triggered = bool(overdue_tasks)
            severity = MonitorSeverity.INFO
            if triggered:
                # Critical if heartbeat is overdue (system may be hung)
                heartbeat_overdue = any(
                    t["task"] == "heartbeat" for t in overdue_tasks
                )
                severity = (
                    MonitorSeverity.CRITICAL if heartbeat_overdue
                    else MonitorSeverity.WARNING
                )

            details = {
                "overdue_tasks": overdue_tasks,
                "healthy_tasks": healthy_tasks,
                "slow_tasks": slow_tasks,
                "total_expected": len(intervals),
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("SchedulerHealthMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            overdue = result.details.get("overdue_tasks", [])
            slow = result.details.get("slow_tasks", [])
            healthy = result.details.get("healthy_tasks", [])
            total = result.details.get("total_expected", 0)

            if not overdue and not slow:
                return f"All {total} scheduled tasks running on time."

            parts: list[str] = []
            if overdue:
                task_msgs = []
                for t in overdue:
                    if t["last_run"] is None:
                        task_msgs.append(f"{t['task']} (never ran)")
                    else:
                        task_msgs.append(
                            f"{t['task']} (overdue by {t['overdue_by_min']:.0f}min)"
                        )
                parts.append(f"OVERDUE tasks: {', '.join(task_msgs)}.")

            if slow:
                parts.append(f"Slowing tasks: {', '.join(slow)}.")

            parts.append(f"Healthy: {len(healthy)}/{total}.")
            return " ".join(parts)

        except Exception as exc:
            log.error("SchedulerHealthMonitor.auto_respond() error: %s", exc)
            return f"Scheduler health check error: {exc}"

    def _detect_slowing_tasks(self, task_durations: list[dict]) -> list[str]:
        """Detect tasks whose duration is trending upward."""
        slow: list[str] = []
        try:
            by_task: dict[str, list[float]] = {}
            for entry in task_durations:
                name = entry.get("task_name", "")
                dur = entry.get("duration_s", 0)
                if name and dur > 0:
                    by_task.setdefault(name, []).append(dur)

            for task_name, durations in by_task.items():
                if len(durations) < 4:
                    continue
                mid = len(durations) // 2
                first_avg = sum(durations[:mid]) / mid
                second_avg = sum(durations[mid:]) / (len(durations) - mid)
                if first_avg > 0 and second_avg / first_avg > 2.0:
                    slow.append(task_name)
        except Exception as exc:
            log.error("Slowing task detection error: %s", exc)

        return slow


# ── 5. CostAnomalyMonitor ───────────────────────────────────────────


class CostAnomalyMonitor(BaseMonitor):
    """Detects abnormal LLM API spending patterns.

    Context keys:
        cost_summary: {total_today, daily_budget, calls_today, by_provider: {name: usd}}
        recent_calls: list of {provider, agent, cost_usd, timestamp}
        cost_trend: optional list of {daily_total, date}
    """

    def __init__(self, params: dict | None = None):
        super().__init__("cost_anomaly", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            summary = context.get("cost_summary", {})
            total_today = summary.get("total_today", 0.0)
            daily_budget = summary.get("daily_budget", 0.15)
            calls_today = summary.get("calls_today", 0)

            # Project daily spend based on time elapsed
            now = _now_utc()
            hours_elapsed = now.hour + now.minute / 60
            hours_elapsed = max(hours_elapsed, 0.5)  # avoid division by zero
            projected_daily = (total_today / hours_elapsed) * 24

            projection_multiplier = self.params.get("projection_multiplier", 2.0)
            budget_exceeded = projected_daily > daily_budget * projection_multiplier

            # Check for duplicate calls within window
            recent_calls = context.get("recent_calls", [])
            dedup_window_min = self.params.get("dedup_window_min", 5)
            duplicates = self._find_duplicates(recent_calls, dedup_window_min)

            # Trend: spending increasing day over day
            trend_direction = "stable"
            cost_trend = context.get("cost_trend", [])
            if len(cost_trend) >= 3:
                totals = [t.get("daily_total", 0) for t in cost_trend]
                recent = totals[-3:]
                if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                    trend_direction = "increasing"

            triggered = budget_exceeded or bool(duplicates)
            severity = MonitorSeverity.INFO
            if triggered:
                severity = (
                    MonitorSeverity.CRITICAL if total_today > daily_budget
                    else MonitorSeverity.WARNING
                )

            details = {
                "total_today": round(total_today, 4),
                "daily_budget": daily_budget,
                "projected_daily": round(projected_daily, 4),
                "calls_today": calls_today,
                "by_provider": summary.get("by_provider", {}),
                "duplicate_calls": duplicates,
                "trend": trend_direction,
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("CostAnomalyMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            total = result.details.get("total_today", 0)
            budget = result.details.get("daily_budget", 0)
            projected = result.details.get("projected_daily", 0)
            calls = result.details.get("calls_today", 0)
            dupes = result.details.get("duplicate_calls", [])
            by_provider = result.details.get("by_provider", {})
            trend = result.details.get("trend", "stable")

            parts: list[str] = []
            parts.append(
                f"Spend: ${total:.4f} today ({calls} calls). "
                f"Projected: ${projected:.4f}/day vs ${budget:.2f} budget."
            )

            if by_provider:
                breakdown = ", ".join(
                    f"{p}=${v:.4f}" for p, v in by_provider.items() if v > 0
                )
                if breakdown:
                    parts.append(f"By provider: {breakdown}.")

            if dupes:
                dupe_summary = ", ".join(
                    f"{d['agent']}@{d['provider']}" for d in dupes[:3]
                )
                parts.append(f"Duplicate calls detected: {dupe_summary}.")

            if trend == "increasing":
                parts.append("Trend: daily spend INCREASING over recent days.")

            return " ".join(parts)

        except Exception as exc:
            log.error("CostAnomalyMonitor.auto_respond() error: %s", exc)
            return f"Cost anomaly check error: {exc}"

    def _find_duplicates(
        self, recent_calls: list[dict], window_min: int
    ) -> list[dict]:
        """Find calls from same agent+provider within dedup window."""
        duplicates: list[dict] = []
        try:
            if len(recent_calls) < 2:
                return duplicates

            # Group by agent+provider
            groups: dict[str, list[dict]] = {}
            for call in recent_calls:
                key = f"{call.get('agent', '')}:{call.get('provider', '')}"
                groups.setdefault(key, []).append(call)

            for key, calls in groups.items():
                if len(calls) < 2:
                    continue

                # Sort by timestamp
                sorted_calls = sorted(
                    calls,
                    key=lambda c: c.get("timestamp", ""),
                )

                for i in range(1, len(sorted_calls)):
                    ts_prev = _parse_iso(sorted_calls[i - 1].get("timestamp"))
                    ts_curr = _parse_iso(sorted_calls[i].get("timestamp"))
                    if ts_prev and ts_curr:
                        delta_min = (ts_curr - ts_prev).total_seconds() / 60
                        if delta_min < window_min:
                            duplicates.append({
                                "agent": sorted_calls[i].get("agent", ""),
                                "provider": sorted_calls[i].get("provider", ""),
                                "gap_min": round(delta_min, 1),
                            })

        except Exception as exc:
            log.error("Duplicate call detection error: %s", exc)

        return duplicates


# ── 6. ConfigIntegrityMonitor ────────────────────────────────────────


class ConfigIntegrityMonitor(BaseMonitor):
    """Validates configuration file integrity and parameter bounds.

    Context keys:
        config_files: {filename: content_string_or_error_string}
        parameter_bounds: {param_dotted_key: {"min": float, "max": float, "type": str}}
    """

    def __init__(self, params: dict | None = None):
        super().__init__("config_integrity", params or {})

    def check(self, context: dict) -> MonitorResult:
        try:
            config_files = context.get("config_files", {})
            parameter_bounds = context.get("parameter_bounds", {})

            corrupt_files: list[dict] = []
            out_of_bounds: list[dict] = []
            parsed_configs: dict[str, dict] = {}

            # Validate JSON parse-ability
            for filename, content in config_files.items():
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        parsed_configs[filename] = parsed
                    except json.JSONDecodeError as je:
                        corrupt_files.append({
                            "file": filename,
                            "error": str(je),
                            "position": je.pos if hasattr(je, "pos") else None,
                        })
                elif isinstance(content, dict):
                    # Already parsed
                    parsed_configs[filename] = content
                else:
                    corrupt_files.append({
                        "file": filename,
                        "error": f"Unexpected content type: {type(content).__name__}",
                    })

            # Validate parameter bounds
            if parameter_bounds and parsed_configs:
                out_of_bounds = self._check_bounds(parsed_configs, parameter_bounds)

            # Check for empty configs (possible truncation)
            empty_configs: list[str] = []
            for filename, parsed in parsed_configs.items():
                if isinstance(parsed, dict) and len(parsed) == 0:
                    empty_configs.append(filename)

            triggered = bool(corrupt_files or out_of_bounds)
            severity = MonitorSeverity.INFO
            if corrupt_files:
                severity = MonitorSeverity.CRITICAL
            elif out_of_bounds:
                severity = MonitorSeverity.WARNING

            details = {
                "corrupt_files": corrupt_files,
                "out_of_bounds": out_of_bounds,
                "empty_configs": empty_configs,
                "files_checked": len(config_files),
                "params_checked": len(parameter_bounds),
            }

            return _safe_result(self.name, triggered, severity, details)

        except Exception as exc:
            log.error("ConfigIntegrityMonitor.check() error: %s", exc, exc_info=True)
            return _safe_result(
                self.name, False, MonitorSeverity.INFO,
                {"error": str(exc)},
            )

    def auto_respond(self, result: MonitorResult, context: dict) -> str:
        try:
            corrupt = result.details.get("corrupt_files", [])
            oob = result.details.get("out_of_bounds", [])
            empty = result.details.get("empty_configs", [])
            files_checked = result.details.get("files_checked", 0)

            parts: list[str] = []

            if corrupt:
                for cf in corrupt:
                    parts.append(
                        f"CORRUPT: {cf['file']} — {cf['error']}"
                    )

            if oob:
                for param in oob:
                    parts.append(
                        f"OUT OF BOUNDS: {param['key']} = {param['value']} "
                        f"(expected {param.get('min', '?')}–{param.get('max', '?')})"
                    )

            if empty:
                parts.append(f"EMPTY configs (possible truncation): {', '.join(empty)}")

            if not parts:
                return f"All {files_checked} config files valid."

            return " ".join(parts)

        except Exception as exc:
            log.error("ConfigIntegrityMonitor.auto_respond() error: %s", exc)
            return f"Config integrity check error: {exc}"

    def _check_bounds(
        self,
        parsed_configs: dict[str, dict],
        parameter_bounds: dict[str, dict],
    ) -> list[dict]:
        """Validate parameter values against defined bounds.

        parameter_bounds keys use dotted notation: "filename.key.subkey"
        e.g. "risk_params.json.max_position_pct" -> risk_params.json -> max_position_pct
        """
        violations: list[dict] = []
        try:
            for dotted_key, bounds in parameter_bounds.items():
                parts = dotted_key.split(".", 1)
                if len(parts) < 2:
                    continue

                filename = parts[0]
                # Handle filenames with dots (e.g., risk_params.json)
                # Try to find the file by matching known filenames
                param_path = parts[1]
                matched_config = None
                matched_param = None

                for config_name, config_data in parsed_configs.items():
                    if dotted_key.startswith(config_name):
                        matched_config = config_data
                        matched_param = dotted_key[len(config_name) + 1:]
                        break

                if matched_config is None:
                    continue

                # Navigate nested keys
                value = matched_config
                path_parts = matched_param.split(".")
                for p in path_parts:
                    if isinstance(value, dict):
                        value = value.get(p)
                    else:
                        value = None
                        break

                if value is None:
                    continue

                # Type coercion for comparison
                try:
                    num_value = float(value)
                except (ValueError, TypeError):
                    continue

                min_val = bounds.get("min")
                max_val = bounds.get("max")
                violation = False

                if min_val is not None and num_value < float(min_val):
                    violation = True
                if max_val is not None and num_value > float(max_val):
                    violation = True

                if violation:
                    violations.append({
                        "key": dotted_key,
                        "value": value,
                        "min": min_val,
                        "max": max_val,
                    })

        except Exception as exc:
            log.error("Parameter bounds check error: %s", exc)

        return violations


# ── Registry ─────────────────────────────────────────────────────────


ALL_MONITORS: dict[str, type[BaseMonitor]] = {
    "feed_health": FeedHealthMonitor,
    "thesis_failure": ThesisFailureMonitor,
    "position_saturation": PositionSaturationMonitor,
    "scheduler_health": SchedulerHealthMonitor,
    "cost_anomaly": CostAnomalyMonitor,
    "config_integrity": ConfigIntegrityMonitor,
}


def create_monitors(monitor_params: dict[str, dict] | None = None) -> list[BaseMonitor]:
    """Instantiate all monitors with optional per-monitor params.

    Args:
        monitor_params: {monitor_name: {param_key: value}}. If None, defaults used.

    Returns:
        List of enabled BaseMonitor instances.
    """
    params = monitor_params or {}
    monitors: list[BaseMonitor] = []
    for name, cls in ALL_MONITORS.items():
        mp = params.get(name, {})
        monitor = cls(mp)
        if monitor.enabled:
            monitors.append(monitor)
        else:
            log.info("Monitor '%s' is disabled, skipping", name)
    return monitors
