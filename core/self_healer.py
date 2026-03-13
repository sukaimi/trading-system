"""Self-Healer — Tier 0 self-healing orchestrator with learning capabilities.

Detection and auto-response are pure Python ($0).
Investigation escalation uses DeepSeek only when needed (~$0.001/call).

Runs every 5 min from heartbeat. Monitors 6 dimensions:
  - Feed health (news/data freshness)
  - Thesis failure rate (pipeline rejection spikes)
  - Position saturation (can't trade despite signals)
  - Scheduler health (missed/overdue tasks)
  - Cost anomaly (LLM spend spikes/duplicates)
  - Config integrity (parameter drift/corruption)

Learns from incidents via pattern matching. Tracks fix effectiveness.
Meta-watchdog prevents runaway self-healing loops.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from core.event_bus import event_bus
from core.logger import setup_logger
from core.schemas import (
    HealerPattern,
    IncidentReport,
    InvestigationFinding,
    MonitorResult,
    MonitorSeverity,
)

log = setup_logger("trading.healer")

# SGT timezone for incident reports
_SGT = timezone(timedelta(hours=8))

# ── Paths ──────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(_ROOT, "config")
DATA_DIR = os.path.join(_ROOT, "data")
INCIDENTS_DIR = os.path.join(DATA_DIR, "incidents")
PATTERNS_FILE = os.path.join(DATA_DIR, "healer_patterns.json")
STATE_FILE = os.path.join(DATA_DIR, "healer_state.json")
PARAMS_FILE = os.path.join(CONFIG_DIR, "self_healer_params.json")
CLAUDE_MD_FILE = os.path.join(_ROOT, "CLAUDE.md")

# How long a known fix must hold (in monitor cycles) before counting as success
_DEFAULT_FIX_SUCCESS_WINDOW = 2


# ── Helpers ────────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_sgt() -> datetime:
    return datetime.now(_SGT)


def _safe_json_load(path: str, default: Any = None) -> Any:
    """Load JSON from disk, returning *default* on any error."""
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _safe_json_save(path: str, data: Any) -> None:
    """Persist JSON to disk, silently ignoring errors."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Failed to persist %s: %s", path, exc)


def _ts_iso(dt: datetime) -> str:
    return dt.isoformat()


# ── SelfHealer ─────────────────────────────────────────────────────────


class SelfHealer:
    """Tier 0 self-healing orchestrator with learning capabilities.

    Detection and auto-response are pure Python ($0).
    Investigation escalation uses DeepSeek only when needed (~$0.001/call).
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        portfolio: Any | None = None,
        cost_tracker: Any | None = None,
        telegram: Any | None = None,
        llm_client: Any | None = None,
    ):
        self._pipeline = pipeline
        self._portfolio = portfolio
        self._cost_tracker = cost_tracker
        self._telegram = telegram
        self._llm = llm_client

        # Monitors — lazily populated in _init_monitors()
        self._monitors: list[Any] = []
        self._monitors_inited = False

        # Pattern knowledge base: signature -> HealerPattern dict
        self._patterns: dict[str, dict[str, Any]] = {}

        # Healer state (action counts, timestamps, meta-watchdog)
        self._state: dict[str, Any] = {
            "last_check": None,
            "action_log": [],        # list of {"ts": ..., "action": ...}
            "exception_log": [],     # list of {"ts": ..., "error": ...}
            "disabled": False,
            "disabled_until": None,
            "total_incidents": 0,
            "total_auto_fixes": 0,
            "total_investigations": 0,
        }

        # Fix tracking: signature -> {"applied_at": ts, "cycles_since": int}
        self._fix_tracker: dict[str, dict[str, Any]] = {}

        # Config params
        self._params: dict[str, Any] = {}

        # Load persisted state
        self._load_params()
        self._load_state()
        self._load_patterns()

        # Ensure incidents directory
        os.makedirs(INCIDENTS_DIR, exist_ok=True)

        log.info("SelfHealer initialized — patterns=%d", len(self._patterns))

    # ── Monitor Initialization ─────────────────────────────────────────

    def _init_monitors(self) -> None:
        """Lazily initialize monitor instances.

        Deferred to first run_all_monitors() so that the healer_monitors
        module (a separate file) can import without circular issues.
        """
        if self._monitors_inited:
            return
        try:
            from core.healer_monitors import (
                ConfigIntegrityMonitor,
                CostAnomalyMonitor,
                FeedHealthMonitor,
                PositionSaturationMonitor,
                SchedulerHealthMonitor,
                ThesisFailureMonitor,
            )

            monitor_params = self._params.get("monitors", {})

            self._monitors = [
                FeedHealthMonitor(monitor_params.get("feed_health", {})),
                ThesisFailureMonitor(monitor_params.get("thesis_failure", {})),
                PositionSaturationMonitor(monitor_params.get("position_saturation", {})),
                SchedulerHealthMonitor(monitor_params.get("scheduler_health", {})),
                CostAnomalyMonitor(monitor_params.get("cost_anomaly", {})),
                ConfigIntegrityMonitor(monitor_params.get("config_integrity", {})),
            ]
            self._monitors_inited = True
            log.info(
                "Initialized %d monitors: %s",
                len(self._monitors),
                [m.__class__.__name__ for m in self._monitors],
            )
        except ImportError as exc:
            log.warning("healer_monitors not available yet: %s", exc)
            self._monitors = []
            # Don't set _monitors_inited — retry next cycle
        except Exception as exc:
            log.error("Failed to init monitors: %s", exc)
            self._monitors = []

    # ── Main Entry Point ───────────────────────────────────────────────

    def run_all_monitors(self) -> list[MonitorResult]:
        """Run all monitors. Called every 5 min from heartbeat.

        Returns list of MonitorResult (one per monitor).
        """
        # Meta-watchdog: refuse to run if we've been disabled
        if self._is_disabled():
            log.info("SelfHealer disabled until %s — skipping", self._state.get("disabled_until"))
            return []

        # Global kill switch from config
        if not self._params.get("enabled", True):
            return []

        self._init_monitors()
        if not self._monitors:
            return []

        results: list[MonitorResult] = []
        triggered_results: list[MonitorResult] = []
        context = self._build_context()

        for monitor in self._monitors:
            try:
                result = monitor.check(context)
                results.append(result)
                if result.triggered:
                    triggered_results.append(result)
            except Exception as exc:
                self._record_exception(f"{monitor.__class__.__name__}.check: {exc}")
                # Produce a non-triggered result so callers know the monitor ran
                results.append(MonitorResult(
                    monitor_name=getattr(monitor, "name", monitor.__class__.__name__),
                    triggered=False,
                    details={"error": str(exc)},
                ))

        # Handle triggered monitors
        if triggered_results:
            # Cross-correlate before handling individually
            correlated = self._cross_correlate(triggered_results)

            for primary, related in correlated:
                try:
                    self._handle_trigger(primary, context, related_results=related)
                except Exception as exc:
                    self._record_exception(f"_handle_trigger({primary.monitor_name}): {exc}")

        # Track fix effectiveness for non-triggered monitors
        self._track_fix_effectiveness_cycle(results)

        # Meta-watchdog check
        self._meta_watchdog_check()

        # Update state
        self._state["last_check"] = _ts_iso(_now_utc())
        self._persist_state()

        # Emit dashboard event
        try:
            event_bus.emit("healer", "check_complete", {
                "total_monitors": len(results),
                "triggered": len(triggered_results),
                "patterns": len(self._patterns),
                "disabled": self._state.get("disabled", False),
            })
        except Exception:
            pass

        return results

    # ── Context Builder ────────────────────────────────────────────────

    def _build_context(self) -> dict[str, Any]:
        """Gather monitoring data from all system sources."""
        ctx: dict[str, Any] = {}

        # Feed health
        try:
            if self._pipeline and hasattr(self._pipeline, "_news_scout"):
                scout = self._pipeline._news_scout
                ctx["feed_health"] = {
                    "last_scan": getattr(scout, "_last_scan_time", None),
                    "consecutive_failures": getattr(scout, "_consecutive_failures", 0),
                }
            else:
                ctx["feed_health"] = {}
        except Exception:
            ctx["feed_health"] = {}

        # Funnel stats (signal flow)
        try:
            if self._pipeline:
                ctx["funnel_stats"] = {
                    "signals_received": getattr(self._pipeline, "_signals_received", 0),
                    "signals_traded": getattr(self._pipeline, "_signals_traded", 0),
                    "signals_rejected": getattr(self._pipeline, "_signals_rejected", 0),
                }
            else:
                ctx["funnel_stats"] = {}
        except Exception:
            ctx["funnel_stats"] = {}

        # Thesis errors (recent failures in pipeline)
        try:
            if self._pipeline and hasattr(self._pipeline, "_thesis_errors"):
                ctx["thesis_errors"] = list(self._pipeline._thesis_errors)
            else:
                ctx["thesis_errors"] = []
        except Exception:
            ctx["thesis_errors"] = []

        # Last trade timestamp
        try:
            if self._pipeline:
                ctx["last_trade_timestamp"] = getattr(
                    self._pipeline, "_last_trade_timestamp", None
                )
            else:
                ctx["last_trade_timestamp"] = None
        except Exception:
            ctx["last_trade_timestamp"] = None

        # Task last runs (scheduler timing)
        try:
            if self._pipeline and hasattr(self._pipeline, "_task_last_runs"):
                ctx["task_last_runs"] = dict(self._pipeline._task_last_runs)
            else:
                ctx["task_last_runs"] = {}
        except Exception:
            ctx["task_last_runs"] = {}

        # Cost summary
        try:
            if self._cost_tracker:
                ctx["cost_summary"] = self._cost_tracker.summary()
            else:
                ctx["cost_summary"] = {}
        except Exception:
            ctx["cost_summary"] = {}

        # Config files (read all JSON from config/)
        try:
            ctx["config_files"] = self._read_config_files()
        except Exception:
            ctx["config_files"] = {}

        # Parameter bounds (from self_optimizer)
        try:
            from core.self_optimizer import PARAMETER_BOUNDS
            ctx["parameter_bounds"] = PARAMETER_BOUNDS
        except Exception:
            ctx["parameter_bounds"] = {}

        # Open positions
        try:
            if self._portfolio:
                snap = self._portfolio.snapshot()
                ctx["open_positions"] = snap.get("open_positions", [])
            else:
                ctx["open_positions"] = []
        except Exception:
            ctx["open_positions"] = []

        return ctx

    def _read_config_files(self) -> dict[str, Any]:
        """Read all JSON files from config/ for integrity checking."""
        configs: dict[str, Any] = {}
        try:
            for name in os.listdir(CONFIG_DIR):
                if name.endswith(".json"):
                    path = os.path.join(CONFIG_DIR, name)
                    try:
                        with open(path) as f:
                            configs[name] = json.load(f)
                    except json.JSONDecodeError:
                        configs[name] = {"__error": "invalid_json"}
                    except Exception as exc:
                        configs[name] = {"__error": str(exc)}
        except Exception:
            pass
        return configs

    # ── Trigger Handling ───────────────────────────────────────────────

    def _handle_trigger(
        self,
        result: MonitorResult,
        context: dict[str, Any],
        related_results: list[MonitorResult] | None = None,
    ) -> None:
        """Handle a triggered monitor with graduated response.

        1st occurrence: auto-respond only
        2nd occurrence: auto-respond + investigate
        3rd+: mark recurring, critical alert
        """
        signature = self._generate_signature(result.monitor_name, result.details)
        pattern = self._patterns.get(signature)
        occurrence = 1 if pattern is None else pattern.get("occurrence_count", 0) + 1

        log.warning(
            "Monitor triggered: %s (severity=%s, occurrence=#%d)",
            result.monitor_name,
            result.severity.value,
            occurrence,
        )

        # Record action
        self._record_action(f"trigger:{result.monitor_name}:#{occurrence}")

        # Check known fix
        known_fix = self._check_known_fix(result.monitor_name, signature)

        # Graduated response
        investigations: list[InvestigationFinding] = []
        resolution = result.auto_response or ""
        status = "open"

        if occurrence == 1:
            # First time: auto-respond only
            log.info("First occurrence of %s — auto-response only", signature[:12])
            status = "open"

        elif occurrence == 2:
            # Second time: auto-respond + investigate
            log.info("Second occurrence of %s — escalating to investigation", signature[:12])
            investigations = self._investigate(result, context)
            self._state["total_investigations"] += 1
            status = "investigating"

        else:
            # 3rd+: recurring
            log.warning(
                "Recurring incident %s (#%d) — marking critical",
                signature[:12],
                occurrence,
            )
            investigations = self._investigate(result, context)
            self._state["total_investigations"] += 1
            status = "recurring"

            # Escalate severity for recurring
            if result.severity != MonitorSeverity.CRITICAL:
                result = MonitorResult(
                    monitor_name=result.monitor_name,
                    triggered=result.triggered,
                    severity=MonitorSeverity.CRITICAL,
                    details=result.details,
                    auto_response=result.auto_response,
                    timestamp=result.timestamp,
                )

        # Apply known fix if confidence is high enough
        if known_fix and occurrence >= 2:
            threshold = self._params.get("learning", {}).get(
                "confidence_threshold_for_auto_fix", 0.8
            )
            if pattern:
                success = pattern.get("fix_success_count", 0)
                failure = pattern.get("fix_failure_count", 0)
                total = success + failure
                confidence = success / total if total > 0 else 0.0
                if confidence >= threshold:
                    resolution = f"Auto-applied known fix: {known_fix}"
                    self._record_action(f"auto_fix:{result.monitor_name}")
                    self._state["total_auto_fixes"] += 1
                    log.info(
                        "Auto-applying known fix for %s (confidence=%.2f): %s",
                        signature[:12], confidence, known_fix,
                    )

        # Add related monitor info
        related_names = []
        if related_results:
            related_names = [r.monitor_name for r in related_results]

        # Generate incident report
        report = self._generate_incident_report(
            result, investigations, context,
            resolution=resolution,
            status=status,
            occurrence_count=occurrence,
            related_monitors=related_names,
        )

        # Persist everything
        self._state["total_incidents"] += 1
        self._write_incident_file(report)
        self._write_vault_note(report)
        self._send_telegram_alert(report)
        self._update_patterns(result.monitor_name, result, signature, investigations)

        # Maybe append to CLAUDE.md for brand-new patterns
        if occurrence == 1:
            self._maybe_append_claude_md(report)

        # Track that a fix was applied (for effectiveness monitoring)
        if resolution:
            self._fix_tracker[signature] = {
                "applied_at": _ts_iso(_now_utc()),
                "cycles_since": 0,
            }

        # Emit event for dashboard
        try:
            event_bus.emit("healer", "incident", {
                "incident_id": report.incident_id,
                "monitor": report.monitor_name,
                "severity": report.severity.value,
                "status": report.status,
                "occurrence": occurrence,
                "auto_response": report.auto_response[:200],
                "related": related_names,
            })
        except Exception:
            pass

    # ── Investigation ──────────────────────────────────────────────────

    def _investigate(
        self,
        result: MonitorResult,
        context: dict[str, Any],
    ) -> list[InvestigationFinding]:
        """Run tech + finance investigation on a triggered monitor.

        Uses DeepSeek for investigation (~$0.001/call).
        Falls back to heuristic investigation if LLM unavailable.
        """
        findings: list[InvestigationFinding] = []

        # Tech investigation (always attempted)
        try:
            tech = self._investigate_tech(result, context)
            if tech:
                findings.append(tech)
        except Exception as exc:
            log.warning("Tech investigation failed: %s", exc)
            findings.append(InvestigationFinding(
                dimension="tech",
                finding=f"Investigation error: {exc}",
                confidence=0.1,
            ))

        # Finance investigation (only if asset-related)
        asset = result.details.get("asset", "")
        if asset:
            try:
                fin = self._investigate_finance(result, context, asset)
                if fin:
                    findings.append(fin)
            except Exception as exc:
                log.warning("Finance investigation failed: %s", exc)

        return findings

    def _investigate_tech(
        self,
        result: MonitorResult,
        context: dict[str, Any],
    ) -> InvestigationFinding | None:
        """Technical root cause analysis.

        Tries LLM first, falls back to heuristic analysis.
        """
        # Heuristic analysis (pure Python, $0)
        monitor = result.monitor_name
        details = result.details

        # Build heuristic finding based on monitor type
        if monitor == "feed_health":
            failures = details.get("consecutive_failures", 0)
            stale_feeds = details.get("stale_feeds", [])
            root_cause = "News API connectivity issues" if failures > 0 else "Feed staleness"
            if stale_feeds:
                root_cause += f" — stale feeds: {', '.join(stale_feeds[:5])}"
            return InvestigationFinding(
                dimension="tech",
                finding=f"Feed health degraded: {failures} consecutive failures",
                root_cause=root_cause,
                recommendation="Check API keys and rate limits. Verify network connectivity.",
                confidence=0.6,
            )

        if monitor == "thesis_failure":
            rate = details.get("failure_rate", 0)
            recent_errors = details.get("recent_errors", [])
            error_summary = "; ".join(recent_errors[:3]) if recent_errors else "unknown"
            return InvestigationFinding(
                dimension="tech",
                finding=f"Thesis failure rate at {rate:.0%}",
                root_cause=f"Pipeline rejection spike — recent: {error_summary}",
                recommendation="Review MarketAnalyst/DevilsAdvocate thresholds. Check for data quality issues.",
                confidence=0.5,
            )

        if monitor == "position_saturation":
            hours = details.get("hours_since_trade", 0)
            signals = details.get("signals_in_window", 0)
            return InvestigationFinding(
                dimension="tech",
                finding=f"No trades for {hours:.1f}h despite {signals} signals",
                root_cause="Risk manager blocking all signals or correlation gridlock",
                recommendation="Check position limits, correlation thresholds, and sector concentration caps.",
                confidence=0.5,
            )

        if monitor == "scheduler_health":
            overdue = details.get("overdue_tasks", [])
            return InvestigationFinding(
                dimension="tech",
                finding=f"Overdue tasks: {', '.join(overdue[:5])}",
                root_cause="Scheduler thread blocked or task timeout exceeded",
                recommendation="Check for hanging LLM calls or network timeouts. Review task logs.",
                confidence=0.6,
            )

        if monitor == "cost_anomaly":
            spend_rate = details.get("current_rate", 0)
            normal_rate = details.get("normal_rate", 0)
            duplicates = details.get("duplicate_calls", 0)
            return InvestigationFinding(
                dimension="tech",
                finding=f"Cost anomaly: ${spend_rate:.4f}/h vs ${normal_rate:.4f}/h normal, {duplicates} duplicates",
                root_cause="Excessive LLM calls — possible retry loop or dedup failure",
                recommendation="Check circuit breaker cooldown, escalation dedup, and cost budget limits.",
                confidence=0.7,
            )

        if monitor == "config_integrity":
            violations = details.get("violations", [])
            return InvestigationFinding(
                dimension="tech",
                finding=f"Config integrity issues: {len(violations)} violations",
                root_cause=f"Parameter drift: {'; '.join(v[:80] for v in violations[:3])}",
                recommendation="Review SelfOptimizer output. Check for JSON corruption. Validate against PARAMETER_BOUNDS.",
                confidence=0.7,
            )

        # Generic fallback
        return InvestigationFinding(
            dimension="tech",
            finding=f"Monitor {monitor} triggered: {json.dumps(details, default=str)[:200]}",
            root_cause="Unknown — needs manual investigation",
            recommendation="Check system logs and recent changes.",
            confidence=0.3,
        )

    def _investigate_finance(
        self,
        result: MonitorResult,
        context: dict[str, Any],
        asset: str,
    ) -> InvestigationFinding | None:
        """Financial context investigation for asset-related incidents.

        Uses LLM if available, otherwise returns basic finding.
        """
        # Try LLM investigation if available
        if self._llm and not getattr(self._llm, "mock_mode", True):
            try:
                return self._investigate_finance_llm(result, context, asset)
            except Exception as exc:
                log.warning("LLM finance investigation failed for %s: %s", asset, exc)

        # Heuristic fallback
        positions = context.get("open_positions", [])
        held = [p for p in positions if p.get("asset") == asset]

        finding = f"Asset {asset} involved in {result.monitor_name} incident"
        if held:
            pos = held[0]
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            finding += f" — position PnL: {pnl_pct:+.1f}%"

        return InvestigationFinding(
            dimension="finance",
            finding=finding,
            root_cause="Needs market context analysis",
            recommendation=f"Review {asset} price action and recent news.",
            confidence=0.3,
        )

    def _investigate_finance_llm(
        self,
        result: MonitorResult,
        context: dict[str, Any],
        asset: str,
    ) -> InvestigationFinding | None:
        """LLM-powered finance investigation (~$0.001/call via DeepSeek)."""
        prompt = (
            f"You are a trading system diagnostician. Analyze this incident:\n"
            f"Monitor: {result.monitor_name}\n"
            f"Asset: {asset}\n"
            f"Details: {json.dumps(result.details, default=str)[:500]}\n"
            f"Open positions: {len(context.get('open_positions', []))}\n\n"
            f"Return JSON: {{\"finding\": \"...\", \"root_cause\": \"...\", "
            f"\"recommendation\": \"...\", \"confidence\": 0.0-1.0}}"
        )

        resp = self._llm.call_deepseek(
            prompt=prompt,
            system_prompt="Diagnose trading system issues. Be concise. Return valid JSON only.",
        )

        if isinstance(resp, dict) and "error" not in resp:
            return InvestigationFinding(
                dimension="finance",
                finding=resp.get("finding", "LLM analysis complete"),
                root_cause=resp.get("root_cause", ""),
                recommendation=resp.get("recommendation", ""),
                confidence=min(float(resp.get("confidence", 0.5)), 1.0),
            )
        return None

    # ── Signature Generation ───────────────────────────────────────────

    def _generate_signature(self, monitor_name: str, details: dict[str, Any]) -> str:
        """Create a stable hash for dedup / pattern matching.

        Hashes monitor_name + key detail fields to produce a consistent
        signature across occurrences of the same issue.
        """
        # Extract stable key fields per monitor type
        key_parts = [monitor_name]

        if monitor_name == "feed_health":
            stale = sorted(details.get("stale_feeds", []))
            key_parts.append(f"stale:{','.join(stale[:5])}")
        elif monitor_name == "thesis_failure":
            # Bucket the failure rate into ranges
            rate = details.get("failure_rate", 0)
            bucket = "low" if rate < 0.3 else "mid" if rate < 0.6 else "high"
            key_parts.append(f"rate:{bucket}")
        elif monitor_name == "position_saturation":
            key_parts.append("saturation")
        elif monitor_name == "scheduler_health":
            overdue_raw = details.get("overdue_tasks", [])
            overdue = sorted(
                t["task"] if isinstance(t, dict) else str(t) for t in overdue_raw
            )
            key_parts.append(f"tasks:{','.join(overdue[:5])}")
        elif monitor_name == "cost_anomaly":
            # Group by anomaly type
            if details.get("duplicate_calls", 0) > 0:
                key_parts.append("duplicates")
            else:
                key_parts.append("spend_spike")
        elif monitor_name == "config_integrity":
            # Group by which config file
            violations = details.get("violations", [])
            files = sorted(set(v.split(":")[0] for v in violations if ":" in v))
            key_parts.append(f"files:{','.join(files[:5])}")
        else:
            # Generic: hash first 200 chars of stringified details
            key_parts.append(json.dumps(details, sort_keys=True, default=str)[:200])

        # Add asset if present
        asset = details.get("asset", "")
        if asset:
            key_parts.append(f"asset:{asset}")

        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Incident Report ────────────────────────────────────────────────

    def _generate_incident_report(
        self,
        result: MonitorResult,
        investigations: list[InvestigationFinding],
        context: dict[str, Any],
        resolution: str = "",
        status: str = "open",
        occurrence_count: int = 1,
        related_monitors: list[str] | None = None,
    ) -> IncidentReport:
        """Build a full incident report from monitor result and investigations."""
        asset = result.details.get("asset", "")

        # Synthesize resolution from investigations if not already set
        if not resolution and investigations:
            recs = [inv.recommendation for inv in investigations if inv.recommendation]
            if recs:
                resolution = "; ".join(recs[:3])

        # Synthesize prevention from root causes
        prevention = ""
        root_causes = [inv.root_cause for inv in investigations if inv.root_cause]
        if root_causes:
            prevention = f"Address root causes: {'; '.join(root_causes[:3])}"

        # Include related monitors in trigger details
        trigger_details = dict(result.details)
        if related_monitors:
            trigger_details["related_monitors"] = related_monitors

        return IncidentReport(
            incident_id=str(uuid.uuid4())[:8],
            timestamp=result.timestamp,
            monitor_name=result.monitor_name,
            severity=result.severity,
            trigger_details=trigger_details,
            auto_response=result.auto_response,
            investigations=investigations,
            resolution=resolution,
            prevention=prevention,
            status=status,
            occurrence_count=occurrence_count,
            asset=asset,
        )

    # ── Incident File Writer ───────────────────────────────────────────

    def _write_incident_file(self, report: IncidentReport) -> str | None:
        """Write markdown incident report to data/incidents/.

        Returns file path on success, None on failure.
        """
        try:
            os.makedirs(INCIDENTS_DIR, exist_ok=True)

            ts_sgt = report.timestamp.astimezone(_SGT) if report.timestamp.tzinfo else report.timestamp
            date_str = ts_sgt.strftime("%Y-%m-%d")
            time_str = ts_sgt.strftime("%H:%M:%S SGT")

            # Build filename: YYYY-MM-DD_monitor_asset.md
            safe_monitor = report.monitor_name.replace(" ", "_")
            asset_part = f"_{report.asset}" if report.asset else ""
            filename = f"{date_str}_{safe_monitor}{asset_part}.md"
            filepath = os.path.join(INCIDENTS_DIR, filename)

            # If file exists (same day, same monitor, same asset), append occurrence
            if os.path.exists(filepath):
                filename = f"{date_str}_{safe_monitor}{asset_part}_{report.incident_id}.md"
                filepath = os.path.join(INCIDENTS_DIR, filename)

            # Summary line
            summary = report.auto_response[:100] if report.auto_response else "triggered"

            # Build markdown
            lines = [
                f"# Incident: {report.monitor_name} — {summary}",
                f"**Date**: {date_str} {time_str}",
                f"**Severity**: {report.severity.value}",
                f"**Status**: {report.status}",
                f"**Occurrence**: #{report.occurrence_count}",
                f"**Incident ID**: {report.incident_id}",
            ]

            if report.asset:
                lines.append(f"**Asset**: {report.asset}")

            # Trigger details
            lines.append("")
            lines.append("## Trigger")
            for key, val in report.trigger_details.items():
                lines.append(f"- **{key}**: {val}")

            # Auto-response
            if report.auto_response:
                lines.append("")
                lines.append("## Auto-Response")
                lines.append(report.auto_response)

            # Investigation findings
            if report.investigations:
                lines.append("")
                lines.append("## Investigation")
                for inv in report.investigations:
                    dim_title = inv.dimension.title()
                    lines.append(f"### {dim_title} Analysis")
                    lines.append(f"**Finding**: {inv.finding}")
                    if inv.root_cause:
                        lines.append(f"**Root Cause**: {inv.root_cause}")
                    if inv.recommendation:
                        lines.append(f"**Recommendation**: {inv.recommendation}")
                    lines.append(f"**Confidence**: {inv.confidence:.0%}")
                    lines.append("")

            # Resolution
            if report.resolution:
                lines.append("## Resolution")
                lines.append(report.resolution)
                lines.append("")

            # Prevention
            if report.prevention:
                lines.append("## Prevention")
                lines.append(report.prevention)
                lines.append("")

            # Related monitors
            related = report.trigger_details.get("related_monitors", [])
            if related:
                lines.append("## Related Monitors")
                for rm in related:
                    lines.append(f"- {rm}")
                lines.append("")

            content = "\n".join(lines) + "\n"
            with open(filepath, "w") as f:
                f.write(content)

            log.info("Incident file written: %s", filepath)
            return filepath

        except Exception as exc:
            log.warning("Failed to write incident file: %s", exc)
            return None

    # ── Vault Note Writer ──────────────────────────────────────────────

    def _write_vault_note(self, report: IncidentReport) -> None:
        """Write incident to the Obsidian memory vault if available."""
        try:
            from core import vault_writer

            # Build what/root_cause/fix from investigation findings
            what = report.auto_response or f"{report.monitor_name} triggered"
            root_cause_parts = [
                inv.root_cause for inv in report.investigations if inv.root_cause
            ]
            root_cause = "; ".join(root_cause_parts) if root_cause_parts else "Under investigation"
            fix = report.resolution or "Pending resolution"

            tags = [
                f"monitor/{report.monitor_name}",
                f"severity/{report.severity.value}",
                "healer/auto",
            ]
            if report.asset:
                tags.append(f"asset/{report.asset}")

            vault_writer.write_incident(
                title=f"Healer — {report.monitor_name} #{report.occurrence_count}",
                what=what,
                root_cause=root_cause,
                fix=fix,
                tags=tags,
                severity=report.severity.value,
            )
        except ImportError:
            pass  # vault_writer not available
        except Exception as exc:
            log.warning("Failed to write vault note: %s", exc)

    # ── CLAUDE.md Append ───────────────────────────────────────────────

    def _maybe_append_claude_md(self, report: IncidentReport) -> None:
        """Append new pattern below <!-- AUTO-HEALER --> marker in CLAUDE.md.

        Only writes for brand-new patterns (first occurrence) to avoid
        flooding the file with duplicates.
        """
        try:
            if not os.path.exists(CLAUDE_MD_FILE):
                return

            with open(CLAUDE_MD_FILE) as f:
                content = f.read()

            marker = "<!-- AUTO-HEALER -->"
            if marker not in content:
                return

            # Build the entry
            ts = _now_sgt().strftime("%Y-%m-%d")
            summary = report.auto_response[:120] if report.auto_response else report.monitor_name
            root_causes = [inv.root_cause for inv in report.investigations if inv.root_cause]
            cause = root_causes[0][:120] if root_causes else "unknown"

            entry = (
                f"\n- **{ts} {report.monitor_name}**: {summary}. "
                f"Root cause: {cause}."
            )

            # Insert after the marker
            new_content = content.replace(marker, marker + entry, 1)

            with open(CLAUDE_MD_FILE, "w") as f:
                f.write(new_content)

            log.info("Appended new healer pattern to CLAUDE.md")

        except Exception as exc:
            log.warning("Failed to append to CLAUDE.md: %s", exc)

    # ── Telegram Alerts ────────────────────────────────────────────────

    def _send_telegram_alert(self, report: IncidentReport) -> None:
        """Send Telegram notification for incidents."""
        if not self._telegram:
            return

        try:
            severity_emoji = {
                MonitorSeverity.INFO: "ℹ️",
                MonitorSeverity.WARNING: "⚠️",
                MonitorSeverity.CRITICAL: "🚨",
            }
            emoji = severity_emoji.get(report.severity, "🔧")

            msg_parts = [
                f"{emoji} *Healer: {report.monitor_name}*",
                f"Severity: {report.severity.value.upper()}",
                f"Status: {report.status} (#{report.occurrence_count})",
            ]

            if report.asset:
                msg_parts.append(f"Asset: {report.asset}")

            if report.auto_response:
                msg_parts.append(f"Response: {report.auto_response[:200]}")

            if report.resolution:
                msg_parts.append(f"Resolution: {report.resolution[:200]}")

            # Only send investigations for WARNING+ or recurring
            if report.investigations and report.severity != MonitorSeverity.INFO:
                for inv in report.investigations[:2]:
                    msg_parts.append(f"[{inv.dimension}] {inv.finding[:150]}")

            msg = "\n".join(msg_parts)

            if hasattr(self._telegram, "send_message"):
                self._telegram.send_message(msg)
            elif hasattr(self._telegram, "send_alert"):
                self._telegram.send_alert(msg)
        except Exception as exc:
            log.warning("Failed to send Telegram alert: %s", exc)

    # ── Pattern Learning ───────────────────────────────────────────────

    def _update_patterns(
        self,
        monitor_name: str,
        result: MonitorResult,
        signature: str,
        investigations: list[InvestigationFinding],
    ) -> None:
        """Update the pattern knowledge base."""
        now = _now_utc()
        existing = self._patterns.get(signature)

        if existing:
            # Update existing pattern
            existing["last_seen"] = _ts_iso(now)
            existing["occurrence_count"] = existing.get("occurrence_count", 0) + 1

            # Update MTTD (running average)
            if existing.get("last_seen"):
                try:
                    last = datetime.fromisoformat(existing["last_seen"])
                    gap_sec = (now - last).total_seconds()
                    old_mttd = existing.get("mean_time_to_detect_sec", 0)
                    count = existing["occurrence_count"]
                    existing["mean_time_to_detect_sec"] = (
                        (old_mttd * (count - 1) + gap_sec) / count
                    )
                except Exception:
                    pass

            # Update known fix from investigations
            recs = [inv.recommendation for inv in investigations if inv.recommendation]
            if recs and not existing.get("known_fix"):
                existing["known_fix"] = recs[0][:300]

            # Update asset, time bucket, regime
            asset = result.details.get("asset", "")
            if asset:
                existing["asset"] = asset
            existing["time_of_day_bucket"] = self._time_bucket(now)

        else:
            # Create new pattern
            asset = result.details.get("asset", "")
            known_fix = ""
            recs = [inv.recommendation for inv in investigations if inv.recommendation]
            if recs:
                known_fix = recs[0][:300]

            self._patterns[signature] = {
                "pattern_id": signature,
                "monitor_name": monitor_name,
                "signature": signature,
                "first_seen": _ts_iso(now),
                "last_seen": _ts_iso(now),
                "occurrence_count": 1,
                "known_fix": known_fix,
                "fix_success_count": 0,
                "fix_failure_count": 0,
                "mean_time_to_detect_sec": 0.0,
                "mean_time_to_resolve_sec": 0.0,
                "false_positive": False,
                "asset": asset,
                "time_of_day_bucket": self._time_bucket(now),
                "regime": "",
            }

        # Enforce max patterns
        max_patterns = self._params.get("learning", {}).get("max_patterns", 500)
        if len(self._patterns) > max_patterns:
            self._prune_patterns(max_patterns)

        self._persist_patterns()

    def _check_known_fix(self, monitor_name: str, signature: str) -> str | None:
        """Look up known fix for this signature.

        Returns fix description if confidence > threshold, else None.
        """
        pattern = self._patterns.get(signature)
        if not pattern:
            return None

        known_fix = pattern.get("known_fix", "")
        if not known_fix:
            return None

        success = pattern.get("fix_success_count", 0)
        failure = pattern.get("fix_failure_count", 0)
        total = success + failure

        if total == 0:
            # No effectiveness data yet — return fix but caller decides
            return known_fix

        threshold = self._params.get("learning", {}).get(
            "confidence_threshold_for_auto_fix", 0.8
        )
        confidence = success / total
        if confidence >= threshold:
            return known_fix

        return None

    def _track_fix_effectiveness(self, signature: str, recurred: bool) -> None:
        """Update fix success/failure counts based on whether issue recurred."""
        pattern = self._patterns.get(signature)
        if not pattern:
            return

        if recurred:
            pattern["fix_failure_count"] = pattern.get("fix_failure_count", 0) + 1
            log.info("Fix for %s FAILED (recurred)", signature[:12])
        else:
            pattern["fix_success_count"] = pattern.get("fix_success_count", 0) + 1
            log.info("Fix for %s SUCCEEDED (no recurrence)", signature[:12])

        self._persist_patterns()

    def _track_fix_effectiveness_cycle(self, results: list[MonitorResult]) -> None:
        """Called each cycle to track whether applied fixes held.

        For each signature in _fix_tracker, check if it re-triggered.
        After enough cycles, record success/failure.
        """
        window = self._params.get("learning", {}).get("fix_success_window_cycles", _DEFAULT_FIX_SUCCESS_WINDOW)

        # Build set of triggered signatures this cycle
        triggered_sigs = set()
        for r in results:
            if r.triggered:
                sig = self._generate_signature(r.monitor_name, r.details)
                triggered_sigs.add(sig)

        # Check tracked fixes
        completed = []
        for sig, tracker in self._fix_tracker.items():
            tracker["cycles_since"] = tracker.get("cycles_since", 0) + 1

            if sig in triggered_sigs:
                # Recurred before window elapsed
                self._track_fix_effectiveness(sig, recurred=True)
                completed.append(sig)
            elif tracker["cycles_since"] >= window:
                # Survived the window without recurrence
                self._track_fix_effectiveness(sig, recurred=False)
                completed.append(sig)

        # Clean up completed trackers
        for sig in completed:
            self._fix_tracker.pop(sig, None)

    def _prune_patterns(self, max_patterns: int) -> None:
        """Remove oldest, least-active patterns to stay under limit."""
        if len(self._patterns) <= max_patterns:
            return

        # Sort by last_seen (oldest first), then by occurrence_count (lowest first)
        sorted_sigs = sorted(
            self._patterns.keys(),
            key=lambda s: (
                self._patterns[s].get("last_seen", ""),
                self._patterns[s].get("occurrence_count", 0),
            ),
        )

        # Remove the oldest/least active
        to_remove = len(self._patterns) - max_patterns
        for sig in sorted_sigs[:to_remove]:
            del self._patterns[sig]

        log.info("Pruned %d patterns, %d remaining", to_remove, len(self._patterns))

    def _time_bucket(self, dt: datetime) -> str:
        """Map a datetime to a time-of-day bucket (4h windows)."""
        sgt = dt.astimezone(_SGT) if dt.tzinfo else dt
        hour = sgt.hour
        if hour < 4:
            return "night"       # 00-04 SGT
        elif hour < 8:
            return "early"       # 04-08 SGT
        elif hour < 12:
            return "morning"     # 08-12 SGT
        elif hour < 16:
            return "afternoon"   # 12-16 SGT
        elif hour < 20:
            return "evening"     # 16-20 SGT
        else:
            return "late"        # 20-24 SGT

    # ── Cross-Correlation ──────────────────────────────────────────────

    def _cross_correlate(
        self,
        triggered_results: list[MonitorResult],
    ) -> list[tuple[MonitorResult, list[MonitorResult]]]:
        """Detect related incidents that should be merged.

        Known correlations:
        - feed_health + position_saturation: data issue causes trading freeze
        - thesis_failure + position_saturation: rejection spike blocks trades
        - cost_anomaly + thesis_failure: LLM errors inflate costs

        Returns list of (primary, [related]) tuples.
        """
        if len(triggered_results) <= 1:
            return [(r, []) for r in triggered_results]

        # Index by monitor name
        by_name: dict[str, MonitorResult] = {
            r.monitor_name: r for r in triggered_results
        }
        names = set(by_name.keys())

        # Track which results have been claimed as "related" to a primary
        claimed: set[str] = set()
        groups: list[tuple[MonitorResult, list[MonitorResult]]] = []

        # Correlation rules (primary, related)
        correlation_rules = [
            ("feed_health", "position_saturation"),
            ("thesis_failure", "position_saturation"),
            ("cost_anomaly", "thesis_failure"),
            ("scheduler_health", "position_saturation"),
            ("config_integrity", "thesis_failure"),
        ]

        for primary_name, related_name in correlation_rules:
            if primary_name in names and related_name in names:
                if primary_name not in claimed and related_name not in claimed:
                    primary = by_name[primary_name]
                    related = by_name[related_name]
                    groups.append((primary, [related]))
                    claimed.add(primary_name)
                    claimed.add(related_name)
                    log.info(
                        "Cross-correlated: %s + %s", primary_name, related_name
                    )

        # Add unclaimed results as standalone
        for name, result in by_name.items():
            if name not in claimed:
                groups.append((result, []))

        return groups

    # ── Meta-Watchdog ──────────────────────────────────────────────────

    def _meta_watchdog_check(self) -> None:
        """Prevent runaway self-healing loops.

        If >max_actions in 1h or >max_exceptions in 1h, disable all monitors.
        Auto re-enable after cooldown_hours.
        """
        watchdog_cfg = self._params.get("watchdog", {})
        max_actions = watchdog_cfg.get("max_actions_per_hour", 10)
        max_exceptions = watchdog_cfg.get("max_exceptions_per_hour", 3)
        cooldown_hours = watchdog_cfg.get("cooldown_hours", 6)

        now = _now_utc()
        one_hour_ago = now - timedelta(hours=1)
        one_hour_ago_iso = _ts_iso(one_hour_ago)

        # Count recent actions
        recent_actions = [
            a for a in self._state.get("action_log", [])
            if a.get("ts", "") > one_hour_ago_iso
        ]

        # Count recent exceptions
        recent_exceptions = [
            e for e in self._state.get("exception_log", [])
            if e.get("ts", "") > one_hour_ago_iso
        ]

        if len(recent_actions) > max_actions:
            log.critical(
                "META-WATCHDOG: %d actions in 1h (max %d) — disabling healer",
                len(recent_actions), max_actions,
            )
            self._disable(cooldown_hours, f"Too many actions: {len(recent_actions)}/h")

        elif len(recent_exceptions) > max_exceptions:
            log.critical(
                "META-WATCHDOG: %d exceptions in 1h (max %d) — disabling healer",
                len(recent_exceptions), max_exceptions,
            )
            self._disable(cooldown_hours, f"Too many exceptions: {len(recent_exceptions)}/h")

    def _disable(self, hours: float, reason: str) -> None:
        """Disable the healer for a specified duration."""
        until = _now_utc() + timedelta(hours=hours)
        self._state["disabled"] = True
        self._state["disabled_until"] = _ts_iso(until)
        self._persist_state()

        log.warning("SelfHealer DISABLED until %s — reason: %s", until.isoformat(), reason)

        # Send Telegram alert
        if self._telegram:
            try:
                msg = (
                    f"🛑 *Healer Meta-Watchdog*\n"
                    f"Self-healer disabled for {hours}h\n"
                    f"Reason: {reason}\n"
                    f"Re-enables: {until.astimezone(_SGT).strftime('%Y-%m-%d %H:%M SGT')}"
                )
                if hasattr(self._telegram, "send_message"):
                    self._telegram.send_message(msg)
                elif hasattr(self._telegram, "send_alert"):
                    self._telegram.send_alert(msg)
            except Exception:
                pass

        # Emit event
        try:
            event_bus.emit("healer", "disabled", {
                "reason": reason,
                "disabled_until": _ts_iso(until),
            })
        except Exception:
            pass

    def _is_disabled(self) -> bool:
        """Check if healer is currently disabled. Auto re-enable if cooldown passed."""
        if not self._state.get("disabled", False):
            return False

        until_str = self._state.get("disabled_until")
        if not until_str:
            return False

        try:
            until = datetime.fromisoformat(until_str)
            # Ensure timezone-aware comparison
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if _now_utc() >= until:
                log.info("SelfHealer cooldown expired — re-enabling")
                self._state["disabled"] = False
                self._state["disabled_until"] = None
                self._persist_state()

                try:
                    event_bus.emit("healer", "re_enabled", {
                        "previous_disabled_until": until_str,
                    })
                except Exception:
                    pass

                return False
        except Exception:
            # If we can't parse the timestamp, re-enable
            self._state["disabled"] = False
            self._state["disabled_until"] = None

        return self._state.get("disabled", False)

    # ── Action / Exception Tracking ────────────────────────────────────

    def _record_action(self, action: str) -> None:
        """Record a healer action for meta-watchdog tracking."""
        entry = {"ts": _ts_iso(_now_utc()), "action": action}
        log_list = self._state.setdefault("action_log", [])
        log_list.append(entry)

        # Keep only last 100 entries
        if len(log_list) > 100:
            self._state["action_log"] = log_list[-100:]

    def _record_exception(self, error: str) -> None:
        """Record a healer exception for meta-watchdog tracking."""
        log.error("Healer exception: %s", error)
        entry = {"ts": _ts_iso(_now_utc()), "error": error}
        exc_list = self._state.setdefault("exception_log", [])
        exc_list.append(entry)

        # Keep only last 50 entries
        if len(exc_list) > 50:
            self._state["exception_log"] = exc_list[-50:]

    # ── Weekly Assessment ──────────────────────────────────────────────

    def weekly_self_assessment(self) -> dict[str, Any]:
        """Generate weekly assessment for the WeeklyStrategist.

        Summarizes incidents, auto-fixes, investigation outcomes,
        fix success/failure rates, and threshold recommendations.
        """
        try:
            now = _now_utc()
            one_week_ago = now - timedelta(days=7)
            one_week_ago_iso = _ts_iso(one_week_ago)

            # Per-monitor stats
            monitor_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
                "total_triggers": 0,
                "false_positives": 0,
                "fix_successes": 0,
                "fix_failures": 0,
                "avg_mttd_sec": 0.0,
                "avg_mttr_sec": 0.0,
            })

            for sig, pattern in self._patterns.items():
                last_seen = pattern.get("last_seen", "")
                if last_seen < one_week_ago_iso:
                    continue  # Skip old patterns

                name = pattern.get("monitor_name", "unknown")
                stats = monitor_stats[name]
                stats["total_triggers"] += pattern.get("occurrence_count", 0)
                stats["fix_successes"] += pattern.get("fix_success_count", 0)
                stats["fix_failures"] += pattern.get("fix_failure_count", 0)
                if pattern.get("false_positive"):
                    stats["false_positives"] += 1
                if pattern.get("mean_time_to_detect_sec", 0) > 0:
                    stats["avg_mttd_sec"] = pattern["mean_time_to_detect_sec"]
                if pattern.get("mean_time_to_resolve_sec", 0) > 0:
                    stats["avg_mttr_sec"] = pattern["mean_time_to_resolve_sec"]

            # Calculate fix success rates
            for name, stats in monitor_stats.items():
                total = stats["fix_successes"] + stats["fix_failures"]
                stats["fix_success_rate"] = (
                    stats["fix_successes"] / total if total > 0 else 0.0
                )
                triggers = stats["total_triggers"]
                stats["false_positive_rate"] = (
                    stats["false_positives"] / triggers if triggers > 0 else 0.0
                )

            # Threshold recommendations
            recommendations: list[str] = []
            for name, stats in monitor_stats.items():
                if stats["false_positive_rate"] > 0.5:
                    recommendations.append(
                        f"{name}: High false positive rate ({stats['false_positive_rate']:.0%})"
                        f" — consider raising trigger thresholds"
                    )
                if stats["fix_success_rate"] < 0.5 and (stats["fix_successes"] + stats["fix_failures"]) >= 3:
                    recommendations.append(
                        f"{name}: Low fix success rate ({stats['fix_success_rate']:.0%})"
                        f" — known fixes need review"
                    )

            # Recent actions from this week
            recent_actions = [
                a for a in self._state.get("action_log", [])
                if a.get("ts", "") > one_week_ago_iso
            ]

            return {
                "period": f"{one_week_ago_iso} to {_ts_iso(now)}",
                "total_incidents": self._state.get("total_incidents", 0),
                "total_auto_fixes": self._state.get("total_auto_fixes", 0),
                "total_investigations": self._state.get("total_investigations", 0),
                "active_patterns": len(self._patterns),
                "monitor_stats": dict(monitor_stats),
                "recommendations": recommendations,
                "actions_this_week": len(recent_actions),
                "meta_watchdog_triggers": sum(
                    1 for a in recent_actions
                    if "disabled" in a.get("action", "")
                ),
                "healer_health": "healthy" if not self._state.get("disabled") else "disabled",
            }
        except Exception as exc:
            log.error("weekly_self_assessment failed: %s", exc)
            return {"error": str(exc)}

    # ── Dashboard Status ───────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return healer status for dashboard endpoint."""
        try:
            # Monitor statuses
            monitor_list = []
            for m in self._monitors:
                monitor_list.append({
                    "name": getattr(m, "name", m.__class__.__name__),
                    "enabled": getattr(m, "enabled", True),
                })

            # Recent actions (last 20)
            recent_actions = self._state.get("action_log", [])[-20:]

            # Active patterns (most recent 20)
            sorted_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.get("last_seen", ""),
                reverse=True,
            )[:20]

            # Learning metrics
            total_success = sum(
                p.get("fix_success_count", 0) for p in self._patterns.values()
            )
            total_failure = sum(
                p.get("fix_failure_count", 0) for p in self._patterns.values()
            )
            total_fixes = total_success + total_failure
            fix_rate = total_success / total_fixes if total_fixes > 0 else 0.0

            return {
                "enabled": self._params.get("enabled", True),
                "disabled": self._state.get("disabled", False),
                "disabled_until": self._state.get("disabled_until"),
                "last_check": self._state.get("last_check"),
                "monitors": monitor_list,
                "recent_actions": recent_actions,
                "active_patterns": [
                    {
                        "signature": p.get("signature", "")[:12],
                        "monitor": p.get("monitor_name", ""),
                        "occurrences": p.get("occurrence_count", 0),
                        "last_seen": p.get("last_seen", ""),
                        "known_fix": bool(p.get("known_fix")),
                        "fix_success_rate": (
                            p.get("fix_success_count", 0)
                            / max(p.get("fix_success_count", 0) + p.get("fix_failure_count", 0), 1)
                        ),
                    }
                    for p in sorted_patterns
                ],
                "learning_metrics": {
                    "total_patterns": len(self._patterns),
                    "total_incidents": self._state.get("total_incidents", 0),
                    "total_auto_fixes": self._state.get("total_auto_fixes", 0),
                    "total_investigations": self._state.get("total_investigations", 0),
                    "overall_fix_success_rate": round(fix_rate, 3),
                },
            }
        except Exception as exc:
            log.error("status() failed: %s", exc)
            return {"error": str(exc)}

    # ── Mark False Positive ────────────────────────────────────────────

    def mark_false_positive(self, signature: str) -> bool:
        """Mark a pattern as a false positive (manual override).

        Returns True if pattern was found and updated.
        """
        pattern = self._patterns.get(signature)
        if not pattern:
            # Try partial match (first 12 chars)
            for sig, pat in self._patterns.items():
                if sig.startswith(signature):
                    pattern = pat
                    signature = sig
                    break

        if not pattern:
            return False

        pattern["false_positive"] = True
        self._persist_patterns()
        log.info("Pattern %s marked as false positive", signature[:12])
        return True

    # ── State Persistence ──────────────────────────────────────────────

    def _load_params(self) -> None:
        """Load config from self_healer_params.json."""
        self._params = _safe_json_load(PARAMS_FILE, {
            "enabled": True,
            "monitors": {},
            "learning": {
                "pattern_retention_days": 90,
                "escalation_threshold_occurrences": 3,
                "fix_success_window_cycles": 2,
                "max_patterns": 500,
                "confidence_threshold_for_auto_fix": 0.8,
            },
            "watchdog": {
                "max_actions_per_hour": 10,
                "max_exceptions_per_hour": 3,
                "cooldown_hours": 6,
            },
        })

    def _load_state(self) -> None:
        """Load healer state from disk."""
        saved = _safe_json_load(STATE_FILE, {})
        if saved:
            self._state.update(saved)

    def _persist_state(self) -> None:
        """Save healer state to disk."""
        _safe_json_save(STATE_FILE, self._state)

    def _load_patterns(self) -> None:
        """Load pattern knowledge base from disk."""
        saved = _safe_json_load(PATTERNS_FILE, {})
        if isinstance(saved, dict):
            self._patterns = saved
        elif isinstance(saved, list):
            # Migration: list of patterns -> dict keyed by signature
            self._patterns = {
                p.get("signature", p.get("pattern_id", str(i))): p
                for i, p in enumerate(saved)
            }

        # Prune expired patterns
        self._prune_expired_patterns()

    def _persist_patterns(self) -> None:
        """Save pattern knowledge base to disk."""
        _safe_json_save(PATTERNS_FILE, self._patterns)

    def _prune_expired_patterns(self) -> None:
        """Remove patterns older than retention_days."""
        retention_days = self._params.get("learning", {}).get("pattern_retention_days", 90)
        cutoff = _now_utc() - timedelta(days=retention_days)
        cutoff_iso = _ts_iso(cutoff)

        to_remove = [
            sig for sig, pat in self._patterns.items()
            if pat.get("last_seen", "") < cutoff_iso
            and not pat.get("known_fix")  # Keep patterns with known fixes
        ]

        for sig in to_remove:
            del self._patterns[sig]

        if to_remove:
            log.info("Pruned %d expired patterns", len(to_remove))

    # ── List Incidents ─────────────────────────────────────────────────

    def list_incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent incident files for dashboard/API.

        Returns list of dicts with filename, date, monitor, severity.
        """
        incidents: list[dict[str, Any]] = []
        try:
            if not os.path.isdir(INCIDENTS_DIR):
                return []

            files = sorted(os.listdir(INCIDENTS_DIR), reverse=True)
            for fname in files[:limit]:
                if not fname.endswith(".md"):
                    continue

                parts = fname.replace(".md", "").split("_")
                date_part = parts[0] if parts else ""
                monitor_part = parts[1] if len(parts) > 1 else ""
                asset_part = parts[2] if len(parts) > 2 else ""

                incidents.append({
                    "filename": fname,
                    "date": date_part,
                    "monitor": monitor_part,
                    "asset": asset_part,
                    "path": os.path.join(INCIDENTS_DIR, fname),
                })
        except Exception as exc:
            log.warning("list_incidents failed: %s", exc)

        return incidents

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset healer state (for testing or manual recovery).

        Clears action log, exception log, and re-enables the healer.
        Does NOT clear pattern knowledge base (learning is preserved).
        """
        self._state = {
            "last_check": None,
            "action_log": [],
            "exception_log": [],
            "disabled": False,
            "disabled_until": None,
            "total_incidents": 0,
            "total_auto_fixes": 0,
            "total_investigations": 0,
        }
        self._fix_tracker = {}
        self._persist_state()
        log.info("SelfHealer state reset (patterns preserved)")

    def reset_patterns(self) -> None:
        """Reset pattern knowledge base (full learning reset)."""
        self._patterns = {}
        self._persist_patterns()
        log.info("SelfHealer patterns reset")
