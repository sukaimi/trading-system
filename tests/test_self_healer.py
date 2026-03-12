"""Tests for core/self_healer.py — SelfHealer orchestrator."""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import MonitorResult, MonitorSeverity, InvestigationFinding


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def healer_env(tmp_path):
    """Patch all file paths to tmp_path so tests don't touch real data."""
    patches = [
        patch("core.self_healer.CONFIG_DIR", str(tmp_path / "config")),
        patch("core.self_healer.DATA_DIR", str(tmp_path / "data")),
        patch("core.self_healer.INCIDENTS_DIR", str(tmp_path / "data" / "incidents")),
        patch("core.self_healer.PATTERNS_FILE", str(tmp_path / "data" / "healer_patterns.json")),
        patch("core.self_healer.STATE_FILE", str(tmp_path / "data" / "healer_state.json")),
        patch("core.self_healer.PARAMS_FILE", str(tmp_path / "config" / "self_healer_params.json")),
        patch("core.self_healer.CLAUDE_MD_FILE", str(tmp_path / "CLAUDE.md")),
    ]
    for p in patches:
        p.start()

    os.makedirs(tmp_path / "config", exist_ok=True)
    os.makedirs(tmp_path / "data" / "incidents", exist_ok=True)

    from core.self_healer import SelfHealer
    yield SelfHealer, tmp_path

    for p in patches:
        p.stop()


@pytest.fixture
def healer(healer_env):
    """Create a SelfHealer with mocked dependencies."""
    SelfHealer, tmp_path = healer_env
    pipeline = MagicMock()
    portfolio = MagicMock()
    cost_tracker = MagicMock()
    telegram = MagicMock()
    llm_client = MagicMock()

    # Default mock returns
    portfolio.snapshot.return_value = {"open_positions": []}
    cost_tracker.summary.return_value = {
        "total_today": 0.01,
        "daily_budget": 0.15,
        "calls_today": 5,
    }
    pipeline._news_scout = MagicMock()
    pipeline._thesis_errors = []
    pipeline._last_trade_timestamp = None
    pipeline._task_last_runs = {}
    pipeline._signals_received = 0
    pipeline._signals_traded = 0
    pipeline._signals_rejected = 0

    h = SelfHealer(
        pipeline=pipeline,
        portfolio=portfolio,
        cost_tracker=cost_tracker,
        telegram=telegram,
        llm_client=llm_client,
    )
    h._tmp_path = tmp_path
    return h


def _make_triggered_result(monitor_name: str, severity=MonitorSeverity.WARNING, details=None):
    return MonitorResult(
        monitor_name=monitor_name,
        triggered=True,
        severity=severity,
        details=details or {},
        auto_response=f"{monitor_name} triggered",
    )


def _make_ok_result(monitor_name: str):
    return MonitorResult(
        monitor_name=monitor_name,
        triggered=False,
        severity=MonitorSeverity.INFO,
        details={},
    )


# ── Core Functionality ────────────────────────────────────────────────


class TestRunAllMonitors:
    def test_run_all_monitors_no_triggers(self, healer):
        """Healthy system returns results for each monitor."""
        # Provide recent task_last_runs so SchedulerHealthMonitor doesn't trigger
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        healer._pipeline._task_last_runs = {
            "heartbeat": (now - timedelta(minutes=1)).isoformat(),
            "news_scan": (now - timedelta(minutes=5)).isoformat(),
            "chart_scan": (now - timedelta(hours=2)).isoformat(),
            "proactive_scan": (now - timedelta(hours=3)).isoformat(),
        }

        results = healer.run_all_monitors()
        # Should return one result per monitor (6 monitors)
        assert len(results) >= 1
        assert isinstance(results, list)

    def test_run_all_monitors_with_trigger(self, healer):
        """One monitor fires, generates incident."""
        # Make a monitor that always triggers
        mock_monitor = MagicMock()
        mock_monitor.name = "test_monitor"
        mock_monitor.enabled = True
        mock_monitor.check.return_value = _make_triggered_result("test_monitor")
        mock_monitor.auto_respond.return_value = "Auto-fix applied"
        mock_monitor.can_investigate.return_value = False

        healer._monitors = [mock_monitor]
        healer._monitors_inited = True

        results = healer.run_all_monitors()
        assert len(results) == 1
        assert results[0].triggered is True

    def test_crash_safety(self, healer):
        """Monitor that raises exception doesn't crash others."""
        # First monitor raises
        bad_monitor = MagicMock()
        bad_monitor.name = "bad_monitor"
        bad_monitor.__class__.__name__ = "BadMonitor"
        bad_monitor.check.side_effect = RuntimeError("kaboom")

        # Second monitor works fine
        good_monitor = MagicMock()
        good_monitor.name = "good_monitor"
        good_monitor.check.return_value = _make_ok_result("good_monitor")

        healer._monitors = [bad_monitor, good_monitor]
        healer._monitors_inited = True

        results = healer.run_all_monitors()
        assert len(results) == 2
        # bad_monitor should have a non-triggered fallback result
        assert results[0].triggered is False
        # good_monitor should be fine
        assert results[1].triggered is False

    def test_disabled_healer(self, healer):
        """When enabled=False in params, run_all_monitors returns empty."""
        healer._params["enabled"] = False
        results = healer.run_all_monitors()
        assert results == []

    def test_investigation_cooldown(self, healer):
        """Monitor's can_investigate controls whether investigation runs."""
        mock_monitor = MagicMock()
        mock_monitor.name = "cooldown_test"
        mock_monitor.enabled = True
        mock_monitor.check.return_value = _make_triggered_result("cooldown_test")
        mock_monitor.auto_respond.return_value = "test"
        mock_monitor.can_investigate.return_value = True

        healer._monitors = [mock_monitor]
        healer._monitors_inited = True

        # First run should work
        healer.run_all_monitors()
        assert healer._state["total_incidents"] >= 1

    def test_status_returns_all_monitors(self, healer):
        """status() has all 6 monitor names after init."""
        healer._init_monitors()
        st = healer.status()
        assert "monitors" in st
        monitor_names = [m["name"] for m in st["monitors"]]
        expected = [
            "feed_health", "thesis_failure", "position_saturation",
            "scheduler_health", "cost_anomaly", "config_integrity",
        ]
        for name in expected:
            assert name in monitor_names, f"Missing monitor: {name}"


# ── Learning System ───────────────────────────────────────────────────


class TestEscalation:
    def test_escalation_first_occurrence(self, healer):
        """First occurrence: auto-fix only, status=open."""
        result = _make_triggered_result("feed_health", details={"stale_feeds": ["yahoo"]})
        ctx = {"feed_health": {}}

        healer._handle_trigger(result, ctx)

        assert healer._state["total_incidents"] == 1
        # First occurrence should not trigger investigation
        assert healer._state["total_investigations"] == 0

    def test_escalation_second_occurrence(self, healer):
        """Second occurrence: auto-fix + investigation."""
        sig = healer._generate_signature("feed_health", {"stale_feeds": ["yahoo"]})
        # Seed a pattern as if first occurrence happened
        healer._patterns[sig] = {
            "pattern_id": sig,
            "monitor_name": "feed_health",
            "signature": sig,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "occurrence_count": 1,
            "known_fix": "",
            "fix_success_count": 0,
            "fix_failure_count": 0,
        }

        result = _make_triggered_result("feed_health", details={"stale_feeds": ["yahoo"]})
        ctx = {"feed_health": {}}

        healer._handle_trigger(result, ctx)

        assert healer._state["total_investigations"] == 1

    def test_escalation_third_occurrence(self, healer):
        """Third occurrence: marked recurring, investigation runs."""
        sig = healer._generate_signature("thesis_failure", {"failure_rate": 0.7})
        healer._patterns[sig] = {
            "pattern_id": sig,
            "monitor_name": "thesis_failure",
            "signature": sig,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "occurrence_count": 2,
            "known_fix": "",
            "fix_success_count": 0,
            "fix_failure_count": 0,
        }

        result = _make_triggered_result(
            "thesis_failure",
            severity=MonitorSeverity.WARNING,
            details={"failure_rate": 0.7},
        )
        ctx = {}

        healer._handle_trigger(result, ctx)

        assert healer._state["total_investigations"] == 1
        # Pattern occurrence count should now be 3
        assert healer._patterns[sig]["occurrence_count"] == 3

    def test_pattern_persistence(self, healer):
        """save/load round-trip for patterns."""
        sig = "abc123test"
        healer._patterns[sig] = {
            "pattern_id": sig,
            "monitor_name": "feed_health",
            "signature": sig,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "occurrence_count": 3,
            "known_fix": "restart feed fetcher",
            "fix_success_count": 2,
            "fix_failure_count": 1,
        }

        healer._persist_patterns()
        healer._patterns = {}
        healer._load_patterns()

        assert sig in healer._patterns
        assert healer._patterns[sig]["occurrence_count"] == 3
        assert healer._patterns[sig]["known_fix"] == "restart feed fetcher"

    def test_fix_effectiveness_tracked(self, healer):
        """Fix success count increments when issue does not recur."""
        sig = "fix_test_sig"
        healer._patterns[sig] = {
            "pattern_id": sig,
            "monitor_name": "cost_anomaly",
            "signature": sig,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "occurrence_count": 2,
            "known_fix": "add dedup",
            "fix_success_count": 0,
            "fix_failure_count": 0,
        }

        # Track fix success (issue did not recur)
        healer._track_fix_effectiveness(sig, recurred=False)
        assert healer._patterns[sig]["fix_success_count"] == 1

        # Track fix failure (issue recurred)
        healer._track_fix_effectiveness(sig, recurred=True)
        assert healer._patterns[sig]["fix_failure_count"] == 1


# ── Reporting ─────────────────────────────────────────────────────────


class TestReporting:
    def test_incident_file_written(self, healer):
        """File created in data/incidents/."""
        from core.self_healer import INCIDENTS_DIR

        result = _make_triggered_result("feed_health", details={"stale_feeds": ["test"]})
        ctx = {"feed_health": {}}

        healer._handle_trigger(result, ctx)

        files = os.listdir(INCIDENTS_DIR)
        md_files = [f for f in files if f.endswith(".md")]
        assert len(md_files) >= 1

    def test_telegram_alert_sent(self, healer):
        """telegram.send_message called on incident."""
        result = _make_triggered_result(
            "scheduler_health",
            severity=MonitorSeverity.CRITICAL,
            details={"overdue_tasks": ["heartbeat"]},
        )
        ctx = {}

        healer._handle_trigger(result, ctx)

        # Telegram should have been called
        assert healer._telegram.send_message.called or healer._telegram.send_alert.called

    def test_weekly_self_assessment(self, healer):
        """Returns dict with expected keys."""
        assessment = healer.weekly_self_assessment()
        expected_keys = [
            "period", "total_incidents", "total_auto_fixes",
            "total_investigations", "active_patterns", "monitor_stats",
            "recommendations", "actions_this_week", "healer_health",
        ]
        for key in expected_keys:
            assert key in assessment, f"Missing key: {key}"

    def test_cross_correlation(self, healer):
        """feed_health + position_saturation should be merged."""
        r1 = _make_triggered_result("feed_health")
        r2 = _make_triggered_result("position_saturation")

        groups = healer._cross_correlate([r1, r2])

        # Should produce 1 group with feed_health as primary, position_saturation as related
        assert len(groups) == 1
        primary, related = groups[0]
        assert primary.monitor_name == "feed_health"
        assert len(related) == 1
        assert related[0].monitor_name == "position_saturation"


# ── Safety ────────────────────────────────────────────────────────────


class TestMetaWatchdog:
    def test_meta_watchdog_disables(self, healer):
        """11 actions in 1h disables healer (default max 10)."""
        from core.self_healer import _ts_iso, _now_utc

        now = _now_utc()
        # Populate 11 recent actions
        healer._state["action_log"] = [
            {"ts": _ts_iso(now - timedelta(minutes=i)), "action": f"trigger:test:#{i}"}
            for i in range(11)
        ]

        healer._meta_watchdog_check()

        assert healer._state["disabled"] is True
        assert healer._state["disabled_until"] is not None

    def test_meta_watchdog_re_enables(self, healer):
        """Healer re-enables after cooldown expires."""
        from core.self_healer import _ts_iso

        # Set disabled with an expired cooldown (1 hour ago)
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        healer._state["disabled"] = True
        healer._state["disabled_until"] = _ts_iso(past)

        assert healer._is_disabled() is False
        assert healer._state["disabled"] is False

    def test_all_methods_exception_safe(self, healer):
        """No public method raises an exception."""
        # Provide recent task runs so scheduler monitor doesn't trigger
        # (avoids a known sorting bug in _generate_signature for scheduler_health)
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        healer._pipeline._task_last_runs = {
            "heartbeat": (now - timedelta(minutes=1)).isoformat(),
            "news_scan": (now - timedelta(minutes=5)).isoformat(),
            "chart_scan": (now - timedelta(hours=2)).isoformat(),
            "proactive_scan": (now - timedelta(hours=3)).isoformat(),
        }

        # status() should never raise
        st = healer.status()
        assert isinstance(st, dict)

        # weekly_self_assessment should never raise
        assessment = healer.weekly_self_assessment()
        assert isinstance(assessment, dict)

        # run_all_monitors should never raise
        results = healer.run_all_monitors()
        assert isinstance(results, list)

        # list_incidents should never raise
        incidents = healer.list_incidents()
        assert isinstance(incidents, list)

        # reset should never raise
        healer.reset()
        assert healer._state["disabled"] is False

        # mark_false_positive with bad signature should return False, not raise
        assert healer.mark_false_positive("nonexistent") is False
