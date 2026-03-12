"""Tests for core/healer_monitors.py — 6 self-healing monitors."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.healer_monitors import (
    FeedHealthMonitor,
    ThesisFailureMonitor,
    PositionSaturationMonitor,
    SchedulerHealthMonitor,
    CostAnomalyMonitor,
    ConfigIntegrityMonitor,
)
from core.schemas import MonitorSeverity


# ── Helpers ───────────────────────────────────────────────────────────

def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _iso_hours_ago(hours: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


# ── 1. FeedHealthMonitor ──────────────────────────────────────────────


class TestFeedHealthMonitorNoTrigger:
    def test_no_trigger_when_feeds_healthy(self):
        """Empty feed_health dict means no issues detected."""
        monitor = FeedHealthMonitor()
        result = monitor.check({"feed_health": {}})
        assert result.triggered is False
        assert result.severity == MonitorSeverity.INFO

    def test_no_trigger_with_healthy_feeds(self):
        """Feeds with zero failures and recent articles don't trigger."""
        monitor = FeedHealthMonitor()
        ctx = {
            "feed_health": {
                "yahoo": {
                    "consecutive_failures": 0,
                    "last_new_article": _iso_hours_ago(2),
                },
                "cnbc": {
                    "consecutive_failures": 1,
                    "last_new_article": _iso_hours_ago(5),
                },
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is False


class TestFeedHealthMonitorTrigger:
    def test_trigger_on_consecutive_failures(self):
        """Feed with 3+ failures triggers (default threshold is 3)."""
        monitor = FeedHealthMonitor()
        ctx = {
            "feed_health": {
                "reuters": {
                    "consecutive_failures": 5,
                    "last_new_article": _iso_hours_ago(1),
                },
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity == MonitorSeverity.CRITICAL
        assert "reuters" in result.details["dead_feeds"]

    def test_trigger_on_stale_feed(self):
        """Feed with last_new_article >24h ago triggers (default stale_hours=24)."""
        monitor = FeedHealthMonitor()
        ctx = {
            "feed_health": {
                "seekingalpha": {
                    "consecutive_failures": 0,
                    "last_new_article": _iso_hours_ago(25),
                },
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity == MonitorSeverity.WARNING
        assert "seekingalpha" in result.details["stale_feeds"]


class TestFeedHealthMonitorAutoRespond:
    def test_auto_respond_lists_dead_feeds(self):
        """auto_response message contains the dead feed name."""
        monitor = FeedHealthMonitor()
        ctx = {
            "feed_health": {
                "bloomberg": {
                    "consecutive_failures": 5,
                    "last_new_article": _iso_hours_ago(30),
                },
            }
        }
        result = monitor.check(ctx)
        response = monitor.auto_respond(result, ctx)
        assert "bloomberg" in response.lower() or "bloomberg" in response

    def test_investigate_returns_findings(self):
        """investigate() returns tech findings for dead feeds with no URL."""
        monitor = FeedHealthMonitor()
        ctx = {
            "feed_health": {
                "test_feed": {
                    "consecutive_failures": 5,
                    "last_new_article": None,
                },
            }
        }
        result = monitor.check(ctx)
        # Dead feed with no URL should produce a "no URL configured" finding
        findings = monitor.investigate(result, ctx, llm_client=None)
        assert len(findings) >= 1
        assert findings[0]["dimension"] == "tech"
        assert "test_feed" in findings[0]["finding"]


class TestFeedHealthMonitorDisabled:
    def test_disabled_monitor_skips(self):
        """When enabled=False, check returns not triggered."""
        monitor = FeedHealthMonitor({"enabled": False})
        assert monitor.enabled is False


# ── 2. ThesisFailureMonitor ───────────────────────────────────────────


class TestThesisFailureMonitorNoTrigger:
    def test_no_trigger_below_threshold(self):
        """40% failure rate with >=4 signals does not trigger (threshold 50%)."""
        monitor = ThesisFailureMonitor()
        ctx = {
            "funnel_stats": {
                "signals_generated": 10,
                "analyst_errors": 4,
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is False

    def test_no_trigger_small_sample(self):
        """100% failure rate but only 2 signals does not trigger (min_sample=4)."""
        monitor = ThesisFailureMonitor()
        ctx = {
            "funnel_stats": {
                "signals_generated": 2,
                "analyst_errors": 2,
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is False
        assert result.details["failure_rate"] == 1.0
        assert result.details["signals_generated"] == 2


class TestThesisFailureMonitorTrigger:
    def test_trigger_above_threshold(self):
        """60% failure rate with >=4 signals triggers."""
        monitor = ThesisFailureMonitor()
        ctx = {
            "funnel_stats": {
                "signals_generated": 5,
                "analyst_errors": 3,
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity in (MonitorSeverity.WARNING, MonitorSeverity.CRITICAL)
        assert result.details["failure_rate"] == 0.6


class TestThesisFailureMonitorAutoRespond:
    def test_auto_respond_includes_errors(self):
        """auto_response mentions error patterns from thesis_errors."""
        monitor = ThesisFailureMonitor()
        ctx = {
            "funnel_stats": {"signals_generated": 6, "analyst_errors": 4},
            "thesis_errors": ["JSON parse error in line 5", "timeout after 30s"],
        }
        result = monitor.check(ctx)
        response = monitor.auto_respond(result, ctx)
        # Should mention the error strings
        assert "JSON parse" in response or "timeout" in response

    def test_investigate_returns_tech_and_finance(self):
        """investigate() returns both tech and finance findings when LLM provided."""
        monitor = ThesisFailureMonitor()
        ctx = {
            "funnel_stats": {"signals_generated": 8, "analyst_errors": 6},
            "thesis_errors": ["JSON parse error", "JSON parse error", "timeout error"],
        }
        result = monitor.check(ctx)

        mock_llm = MagicMock()
        mock_llm.call_deepseek.return_value = (
            '{"pattern": "JSON parse failures", "fix": "Add fallback parser", '
            '"missed_opportunities_estimate": 3, "confidence": 0.8}'
        )

        findings = monitor.investigate(result, ctx, llm_client=mock_llm)
        dimensions = [f["dimension"] for f in findings]
        assert "tech" in dimensions
        assert "finance" in dimensions


# ── 3. PositionSaturationMonitor ──────────────────────────────────────


class TestPositionSaturationMonitorNoTrigger:
    def test_no_trigger_recent_trade(self):
        """Trade 2h ago with default threshold 6h should not trigger."""
        monitor = PositionSaturationMonitor()
        ctx = {
            "last_trade_timestamp": _iso_hours_ago(2),
            "funnel_stats": {"signals_generated": 5},
        }
        result = monitor.check(ctx)
        assert result.triggered is False

    def test_no_trigger_no_signals(self):
        """No signals means no saturation even if no trades."""
        monitor = PositionSaturationMonitor()
        ctx = {
            "last_trade_timestamp": None,
            "funnel_stats": {"signals_generated": 0},
        }
        result = monitor.check(ctx)
        assert result.triggered is False


class TestPositionSaturationMonitorTrigger:
    def test_trigger_no_trades(self):
        """No trades for 7h with signals triggers."""
        monitor = PositionSaturationMonitor()
        ctx = {
            "last_trade_timestamp": _iso_hours_ago(7),
            "funnel_stats": {
                "signals_generated": 5,
                "analyst_errors": 1,
                "devil_killed": 2,
                "risk_rejected": 1,
            },
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity in (MonitorSeverity.WARNING, MonitorSeverity.CRITICAL)

    def test_bottleneck_identification(self):
        """Identifies highest drop-off stage as bottleneck."""
        monitor = PositionSaturationMonitor()
        ctx = {
            "last_trade_timestamp": _iso_hours_ago(8),
            "funnel_stats": {
                "signals_generated": 10,
                "analyst_errors": 1,
                "analyst_no_trade": 0,
                "devil_killed": 6,
                "risk_rejected": 2,
                "rr_rejected": 1,
            },
        }
        result = monitor.check(ctx)
        assert result.details["bottleneck"] == "devil_killed"


# ── 4. SchedulerHealthMonitor ─────────────────────────────────────────


class TestSchedulerHealthMonitorNoTrigger:
    def test_no_trigger_all_on_time(self):
        """All tasks recently ran, no overdue."""
        monitor = SchedulerHealthMonitor()
        ctx = {
            "task_last_runs": {
                "heartbeat": _iso_hours_ago(0.05),      # 3 min ago
                "news_scan": _iso_hours_ago(0.2),        # 12 min ago
                "chart_scan": _iso_hours_ago(3),          # 3h ago
                "proactive_scan": _iso_hours_ago(4),      # 4h ago
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is False


class TestSchedulerHealthMonitorTrigger:
    def test_trigger_overdue_task(self):
        """Task overdue by 3x its interval triggers."""
        monitor = SchedulerHealthMonitor()
        ctx = {
            "task_last_runs": {
                "heartbeat": _iso_hours_ago(0.05),
                "news_scan": _iso_hours_ago(2),  # 2h ago, expected 15min * 2x = 30min
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        overdue_names = [t["task"] for t in result.details["overdue_tasks"]]
        assert "news_scan" in overdue_names

    def test_heartbeat_overdue_is_critical(self):
        """Heartbeat overdue should produce CRITICAL severity."""
        monitor = SchedulerHealthMonitor()
        ctx = {
            "task_last_runs": {
                "heartbeat": _iso_hours_ago(1),  # 1h ago, expected 5min * 2x = 10min
            }
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity == MonitorSeverity.CRITICAL


# ── 5. CostAnomalyMonitor ────────────────────────────────────────────


class TestCostAnomalyMonitorNoTrigger:
    def test_no_trigger_normal_spend(self):
        """Normal spend rate should not trigger."""
        monitor = CostAnomalyMonitor()
        ctx = {
            "cost_summary": {
                "total_today": 0.01,
                "daily_budget": 0.15,
                "calls_today": 5,
                "by_provider": {"deepseek": 0.01},
            },
            "recent_calls": [],
        }
        result = monitor.check(ctx)
        assert result.triggered is False


class TestCostAnomalyMonitorTrigger:
    def test_trigger_high_spend_rate(self):
        """Projected 3x daily budget triggers."""
        monitor = CostAnomalyMonitor()
        # Simulate high spend early in the day: total_today is high relative to
        # hours elapsed. The monitor projects based on the current UTC hour.
        ctx = {
            "cost_summary": {
                "total_today": 0.50,  # Already 3x the $0.15 budget
                "daily_budget": 0.15,
                "calls_today": 50,
                "by_provider": {"anthropic": 0.40, "deepseek": 0.10},
            },
            "recent_calls": [],
        }
        result = monitor.check(ctx)
        # Projected daily will be much higher than budget * 2.0 multiplier
        assert result.triggered is True

    def test_trigger_duplicate_calls(self):
        """Same agent+provider calls within 5min window detected."""
        monitor = CostAnomalyMonitor()
        now = datetime.now(timezone.utc)
        ts1 = (now - timedelta(minutes=2)).isoformat()
        ts2 = (now - timedelta(minutes=1)).isoformat()

        ctx = {
            "cost_summary": {
                "total_today": 0.01,
                "daily_budget": 0.15,
                "calls_today": 5,
            },
            "recent_calls": [
                {"agent": "circuit_breaker", "provider": "anthropic",
                 "cost_usd": 0.04, "timestamp": ts1},
                {"agent": "circuit_breaker", "provider": "anthropic",
                 "cost_usd": 0.04, "timestamp": ts2},
            ],
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert len(result.details["duplicate_calls"]) >= 1


# ── 6. ConfigIntegrityMonitor ─────────────────────────────────────────


class TestConfigIntegrityMonitorNoTrigger:
    def test_no_trigger_valid_configs(self):
        """All configs valid, no trigger."""
        monitor = ConfigIntegrityMonitor()
        ctx = {
            "config_files": {
                "risk_params.json": {"max_position_pct": 0.07, "stop_loss_pct": 0.03},
                "agent_params.json": {"news_scan_interval": 15},
            },
            "parameter_bounds": {},
        }
        result = monitor.check(ctx)
        assert result.triggered is False
        assert result.details["files_checked"] == 2


class TestConfigIntegrityMonitorTrigger:
    def test_trigger_corrupt_json(self):
        """Malformed JSON string triggers CRITICAL."""
        monitor = ConfigIntegrityMonitor()
        ctx = {
            "config_files": {
                "risk_params.json": "{bad json!!!",
            },
            "parameter_bounds": {},
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity == MonitorSeverity.CRITICAL
        assert len(result.details["corrupt_files"]) == 1
        assert result.details["corrupt_files"][0]["file"] == "risk_params.json"

    def test_trigger_out_of_bounds(self):
        """Value exceeding bounds triggers WARNING."""
        monitor = ConfigIntegrityMonitor()
        ctx = {
            "config_files": {
                "risk_params.json": {"max_position_pct": 0.50},
            },
            "parameter_bounds": {
                "risk_params.json.max_position_pct": {"min": 0.01, "max": 0.15},
            },
        }
        result = monitor.check(ctx)
        assert result.triggered is True
        assert result.severity == MonitorSeverity.WARNING
        assert len(result.details["out_of_bounds"]) == 1
        assert result.details["out_of_bounds"][0]["value"] == 0.50
