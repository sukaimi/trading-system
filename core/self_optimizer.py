"""Self-Optimizer — translates weekly strategy directives into config changes.

Takes WeeklyDirective from the Weekly Strategist and updates agent
parameter JSON files. Supports rollback to previous versions.
Sends Telegram notification after applying changes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.logger import setup_logger

log = setup_logger("trading.optimizer")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OPT_LOG_FILE = os.path.join(DATA_DIR, "optimization_log.json")
VERSION_FILE = os.path.join(CONFIG_DIR, "strategy_version.json")
DIRECTIVES_FILE = os.path.join(DATA_DIR, "data_driven_directives.json")

# Learning system data files (read-only — produced by each learning module)
_LEARNING_FILES = {
    "stop_recommendations": os.path.join(DATA_DIR, "stop_recommendations.json"),
    "confidence_calibration": os.path.join(DATA_DIR, "confidence_calibration.json"),
    "phantom_analysis": os.path.join(DATA_DIR, "phantom_trades.json"),
    "session_analysis": os.path.join(DATA_DIR, "session_analysis.json"),
}


class SelfOptimizer:
    """Translates Sonnet strategy directives into actual config changes."""

    def __init__(
        self,
        telegram: Any | None = None,
        portfolio: Any | None = None,
    ):
        self._telegram = telegram
        self._portfolio = portfolio

    def apply_directives(self, weekly_directive: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply parameter changes from a weekly directive.

        Args:
            weekly_directive: Dict with 'parameter_changes' list and 'week_reviewed'.

        Returns:
            List of change records applied.
        """
        changes_applied: list[dict[str, Any]] = []

        for change in weekly_directive.get("parameter_changes", []):
            target_agent = change["target"]
            param_path = change["parameter"]
            new_value = change["new_value"]
            old_value = change["old_value"]

            self._update_config(target_agent, param_path, new_value)
            version = self._increment_version()

            changes_applied.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": target_agent,
                    "parameter": param_path,
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": change.get("reason", ""),
                    "directive_source": weekly_directive.get("week_reviewed", ""),
                    "version": version,
                }
            )

        if changes_applied:
            self._append_to_log(changes_applied, weekly_directive)
            log.info("Applied %d parameter changes", len(changes_applied))

            # Send Telegram notification
            if self._telegram:
                try:
                    self._telegram.send_optimization_summary(changes_applied)
                except Exception as e:
                    log.error("Failed to send optimization summary: %s", e)

        return changes_applied

    # ── Learning system integration ─────────────────────────────────────

    def gather_learning_insights(self) -> dict[str, Any]:
        """Collect insights from all Tier 0 learning systems.

        Reads from JSON files persisted by each learning system.
        Returns a summary that the Weekly Strategist can use for context.
        """
        result: dict[str, Any] = {
            "stop_recommendations": None,
            "confidence_calibration": None,
            "phantom_analysis": None,
            "session_analysis": None,
            "actionable_insights": [],
        }

        # Load each learning system's persisted data
        for key, filepath in _LEARNING_FILES.items():
            data = self._read_json_safe(filepath)
            if data is not None:
                result[key] = data

        # Generate actionable insights from the data
        result["actionable_insights"] = self._derive_insights(result)

        return result

    def generate_data_driven_directives(self) -> list[dict[str, Any]]:
        """Generate parameter change suggestions from learning data.

        These are SUGGESTIONS -- they are persisted to data/data_driven_directives.json
        for the Weekly Strategist (or human) to review. They are NOT auto-applied.

        Returns list of directive dicts.
        """
        insights = self.gather_learning_insights()
        directives: list[dict[str, Any]] = []

        # Load current config values for comparison
        risk_params = self._read_json_safe(
            os.path.join(CONFIG_DIR, "risk_params.json")
        ) or {}

        # 1. Stop-loss directives from AdaptiveStopOptimizer
        stop_recs = insights.get("stop_recommendations")
        if stop_recs and isinstance(stop_recs, dict):
            self._directives_from_stops(stop_recs, risk_params, directives)

        # 2. Confidence calibration directives
        cal = insights.get("confidence_calibration")
        if cal and isinstance(cal, dict):
            self._directives_from_calibration(cal, directives)

        # 3. Phantom analysis directives
        phantom = insights.get("phantom_analysis")
        if phantom and isinstance(phantom, (dict, list)):
            self._directives_from_phantom(phantom, directives)

        # 4. Session analysis directives
        session = insights.get("session_analysis")
        if session and isinstance(session, dict):
            self._directives_from_sessions(session, directives)

        # Persist directives for review
        self._persist_directives(directives)

        log.info("Generated %d data-driven directive suggestions", len(directives))
        return directives

    def _read_json_safe(self, filepath: str) -> Any | None:
        """Read a JSON file, returning None on any error."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read %s: %s", filepath, e)
            return None

    def _derive_insights(self, data: dict[str, Any]) -> list[str]:
        """Derive human-readable actionable insights from learning data."""
        insights: list[str] = []

        # Stop recommendations insights
        stop_recs = data.get("stop_recommendations")
        if stop_recs and isinstance(stop_recs, dict):
            per_asset = stop_recs.get("per_asset", {})
            for asset, rec in per_asset.items():
                if isinstance(rec, dict):
                    suggested = rec.get("suggested_stop_pct")
                    current = rec.get("current_stop_pct")
                    if suggested is not None and current is not None:
                        if abs(suggested - current) > 0.3:
                            direction = "widened" if suggested > current else "tightened"
                            insights.append(
                                f"Stop-loss for {asset} should be {direction} from "
                                f"{current}% to {suggested}% based on MAE analysis"
                            )

        # Confidence calibration insights
        cal = data.get("confidence_calibration")
        if cal and isinstance(cal, dict):
            buckets = cal.get("buckets", {})
            for bucket_name, bucket_data in buckets.items():
                if not isinstance(bucket_data, dict):
                    continue
                actual_wr = bucket_data.get("actual_win_rate")
                avg_conf = bucket_data.get("avg_confidence")
                count = bucket_data.get("count", 0)
                if actual_wr is not None and avg_conf is not None and count >= 5:
                    if avg_conf > 0 and actual_wr < avg_conf * 0.7:
                        insights.append(
                            f"System is overconfident at {bucket_name} confidence "
                            f"levels ({avg_conf:.0%}) -- actual win rate is "
                            f"{actual_wr:.0%}"
                        )
                    elif avg_conf > 0 and actual_wr > avg_conf * 1.3:
                        insights.append(
                            f"System is underconfident at {bucket_name} confidence "
                            f"levels ({avg_conf:.0%}) -- actual win rate is "
                            f"{actual_wr:.0%}"
                        )

        # Phantom trade insights
        phantom = data.get("phantom_analysis")
        if phantom and isinstance(phantom, (dict, list)):
            trades = phantom if isinstance(phantom, list) else phantom.get("recent", [])
            if trades:
                # Count kills by agent
                kill_counts: dict[str, int] = {}
                profitable_kills: dict[str, int] = {}
                for t in trades:
                    if not isinstance(t, dict):
                        continue
                    killer = t.get("killed_by", "unknown")
                    kill_counts[killer] = kill_counts.get(killer, 0) + 1
                    if t.get("outcome_pnl_pct") is not None and t["outcome_pnl_pct"] > 0:
                        profitable_kills[killer] = profitable_kills.get(killer, 0) + 1

                for killer, total in kill_counts.items():
                    if total >= 5:
                        profitable = profitable_kills.get(killer, 0)
                        false_kill_rate = profitable / total
                        if false_kill_rate > 0.5:
                            insights.append(
                                f"{killer} has a {false_kill_rate:.0%} false kill rate "
                                f"({profitable}/{total} killed trades would have been profitable)"
                            )

        # Session analysis insights
        session = data.get("session_analysis")
        if session and isinstance(session, dict):
            per_asset = session.get("per_asset", {})
            for asset, sessions in per_asset.items():
                if not isinstance(sessions, dict):
                    continue
                best = sessions.get("best_session")
                if best:
                    best_data = sessions.get(best, {})
                    if isinstance(best_data, dict) and best_data.get("trades", 0) >= 5:
                        insights.append(
                            f"{best.capitalize()} session outperforms for {asset} "
                            f"(win rate: {best_data.get('win_rate', 0):.0%}, "
                            f"avg PnL: {best_data.get('avg_pnl_pct', 0):.1f}%)"
                        )

        return insights

    def _directives_from_stops(
        self,
        stop_recs: dict[str, Any],
        risk_params: dict[str, Any],
        directives: list[dict[str, Any]],
    ) -> None:
        """Generate directives from stop-loss recommendations."""
        current_default = risk_params.get("default_stop_loss_pct", 3.0)

        # Check for a global recommendation
        global_rec = stop_recs.get("global_suggested_stop_pct")
        if global_rec is not None and abs(global_rec - current_default) > 0.2:
            directives.append({
                "target": "risk_params",
                "parameter": "default_stop_loss_pct",
                "old_value": current_default,
                "new_value": round(global_rec, 1),
                "reason": "MAE analysis suggests adjusting default stop-loss distance",
                "confidence": 0.7,
                "source": "adaptive_stops",
            })

        # Per-asset ATR multiplier suggestions
        per_asset = stop_recs.get("per_asset", {})
        for asset, rec in per_asset.items():
            if not isinstance(rec, dict):
                continue
            suggested_atr_mult = rec.get("suggested_atr_mult")
            current_atr_mult = risk_params.get("stop_loss_atr_mult", 2.0)
            if (
                suggested_atr_mult is not None
                and abs(suggested_atr_mult - current_atr_mult) > 0.2
            ):
                directives.append({
                    "target": "risk_params",
                    "parameter": "stop_loss_atr_mult",
                    "old_value": current_atr_mult,
                    "new_value": round(suggested_atr_mult, 1),
                    "reason": f"MAE analysis for {asset} suggests ATR multiplier adjustment",
                    "confidence": 0.6,
                    "source": "adaptive_stops",
                })
                break  # Only one global ATR mult directive

    def _directives_from_calibration(
        self,
        cal: dict[str, Any],
        directives: list[dict[str, Any]],
    ) -> None:
        """Generate directives from confidence calibration data."""
        overall = cal.get("overall_correction")
        if overall is not None and isinstance(overall, (int, float)):
            if abs(overall) > 0.05:
                direction = "down" if overall < 0 else "up"
                directives.append({
                    "target": "risk_params",
                    "parameter": "confidence_correction_factor",
                    "old_value": 0.0,
                    "new_value": round(overall, 2),
                    "reason": f"Confidence calibration suggests adjusting confidence {direction} by {abs(overall):.2f}",
                    "confidence": 0.65,
                    "source": "confidence_calibrator",
                })

    def _directives_from_phantom(
        self,
        phantom: dict[str, Any] | list[Any],
        directives: list[dict[str, Any]],
    ) -> None:
        """Generate directives from phantom trade analysis."""
        trades = phantom if isinstance(phantom, list) else phantom.get("recent", [])
        if not trades or len(trades) < 10:
            return

        # Check if too many profitable trades are being killed
        checked = [t for t in trades if isinstance(t, dict) and t.get("outcome_pnl_pct") is not None]
        if not checked:
            return

        profitable = [t for t in checked if t["outcome_pnl_pct"] > 0]
        false_kill_rate = len(profitable) / len(checked) if checked else 0

        if false_kill_rate > 0.5:
            directives.append({
                "target": "devils_advocate_params",
                "parameter": "kill_threshold",
                "old_value": None,
                "new_value": "review_needed",
                "reason": (
                    f"Phantom analysis shows {false_kill_rate:.0%} of killed trades "
                    f"would have been profitable ({len(profitable)}/{len(checked)})"
                ),
                "confidence": 0.5,
                "source": "phantom_tracker",
            })

    def _directives_from_sessions(
        self,
        session: dict[str, Any],
        directives: list[dict[str, Any]],
    ) -> None:
        """Generate directives from session analysis data."""
        if not session.get("sufficient_data"):
            return

        overall = session.get("overall", {})
        if not overall:
            return

        # Find best and worst sessions
        best_session = None
        best_score = -1.0
        worst_session = None
        worst_score = 2.0

        for name, stats in overall.items():
            if not isinstance(stats, dict):
                continue
            score = stats.get("score", 0.5)
            if stats.get("trades", 0) >= 5:
                if score > best_score:
                    best_score = score
                    best_session = name
                if score < worst_score:
                    worst_score = score
                    worst_session = name

        if best_session and worst_session and best_session != worst_session:
            if best_score - worst_score > 0.15:
                directives.append({
                    "target": "risk_params",
                    "parameter": "session_preference",
                    "old_value": None,
                    "new_value": {
                        "best": best_session,
                        "worst": worst_session,
                        "score_gap": round(best_score - worst_score, 2),
                    },
                    "reason": (
                        f"{best_session.capitalize()} session outperforms "
                        f"{worst_session} by {best_score - worst_score:.2f} score points"
                    ),
                    "confidence": 0.55,
                    "source": "session_analyzer",
                })

    def _persist_directives(self, directives: list[dict[str, Any]]) -> None:
        """Save generated directives to data/data_driven_directives.json."""
        os.makedirs(DATA_DIR, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(directives),
            "directives": directives,
        }

        try:
            with open(DIRECTIVES_FILE, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            log.info("Persisted %d directives to %s", len(directives), DIRECTIVES_FILE)
        except OSError as e:
            log.error("Failed to persist directives: %s", e)

    def _update_config(self, agent: str, param_path: str, value: Any) -> None:
        """Update an agent's dynamic parameters JSON file."""
        config_file = os.path.join(CONFIG_DIR, f"{agent}_params.json")
        if not os.path.exists(config_file):
            log.warning("Config file not found: %s", config_file)
            return

        with open(config_file) as f:
            config = json.load(f)

        # Navigate nested path (e.g., "signal_weights.regulatory_crypto")
        keys = param_path.split(".")
        obj = config
        for key in keys[:-1]:
            obj = obj.setdefault(key, {})
        obj[keys[-1]] = value

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        log.info("Updated %s.%s = %s", agent, param_path, value)

    def _increment_version(self) -> int:
        """Increment and return the strategy version number."""
        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                v = json.load(f)
        else:
            v = {"version": 0}

        v["version"] += 1
        v["last_updated"] = datetime.utcnow().isoformat()

        with open(VERSION_FILE, "w") as f:
            json.dump(v, f, indent=2)

        return v["version"]

    def _append_to_log(
        self, changes: list[dict[str, Any]], directive: dict[str, Any]
    ) -> None:
        """Append optimization changes to the log file."""
        os.makedirs(DATA_DIR, exist_ok=True)

        log_entries: list[dict[str, Any]] = []
        if os.path.exists(OPT_LOG_FILE):
            with open(OPT_LOG_FILE) as f:
                log_entries = json.load(f)

        # Populate portfolio value if available
        portfolio_value = 0.0
        cumulative_return = 0.0
        if self._portfolio:
            try:
                portfolio_value = getattr(self._portfolio, "equity", 0.0)
                initial = getattr(self._portfolio, "initial_capital", portfolio_value)
                if initial > 0:
                    cumulative_return = (portfolio_value - initial) / initial * 100
            except Exception:
                pass

        log_entries.append(
            {
                "version": changes[-1]["version"] if changes else 0,
                "timestamp": datetime.utcnow().isoformat(),
                "directive_week": directive.get("week_reviewed", ""),
                "changes": changes,
                "portfolio_value_at_change": portfolio_value,
                "cumulative_return_pct": cumulative_return,
            }
        )

        with open(OPT_LOG_FILE, "w") as f:
            json.dump(log_entries, f, indent=2)

    def rollback(self, version_to_restore: int) -> bool:
        """Restore parameters from a previous version.

        Reads the optimization log, finds all changes after the target
        version, and reverses them in order.

        Returns True if rollback succeeded.
        """
        if not os.path.exists(OPT_LOG_FILE):
            log.warning("No optimization log found — nothing to rollback")
            return False

        with open(OPT_LOG_FILE) as f:
            log_entries = json.load(f)

        # Find entries to reverse (everything after target version)
        to_reverse = [
            entry
            for entry in log_entries
            if entry.get("version", 0) > version_to_restore
        ]

        if not to_reverse:
            log.info("No changes found after version %d", version_to_restore)
            return False

        # Reverse in newest-first order
        for entry in reversed(to_reverse):
            for change in reversed(entry.get("changes", [])):
                self._update_config(
                    change["agent"], change["parameter"], change["old_value"]
                )

        log.info("Rolled back to version %d (%d entries reversed)",
                 version_to_restore, len(to_reverse))
        return True
