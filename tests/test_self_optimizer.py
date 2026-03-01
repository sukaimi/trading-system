"""Tests for core/self_optimizer.py"""

import json
import os
from unittest.mock import MagicMock

import pytest

from core.self_optimizer import SelfOptimizer


@pytest.fixture
def tmp_config(tmp_path):
    """Create temporary config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # news_scout_params.json
    ns_params = {
        "signal_weights": {"central_bank": 0.9, "geopolitical": 0.8},
        "min_signal_threshold": 0.4,
    }
    (config_dir / "news_scout_params.json").write_text(json.dumps(ns_params))

    # risk_params.json
    risk_params = {"max_position_pct": 7.0, "max_daily_loss_pct": 5.0}
    (config_dir / "risk_params.json").write_text(json.dumps(risk_params))

    # strategy_version.json
    version = {"version": 1, "last_updated": "2026-01-01"}
    (config_dir / "strategy_version.json").write_text(json.dumps(version))

    return str(config_dir), str(data_dir)


@pytest.fixture
def optimizer(tmp_config):
    from unittest.mock import patch
    config_dir, data_dir = tmp_config
    with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
         patch("core.self_optimizer.DATA_DIR", data_dir), \
         patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")), \
         patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
        yield SelfOptimizer(), config_dir, data_dir


class TestApplyDirectives:
    def test_applies_changes(self, optimizer):
        opt, config_dir, _ = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-01",
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "min_signal_threshold",
                     "old_value": 0.4, "new_value": 0.35, "reason": "Lower threshold"},
                ],
            }
            changes = opt.apply_directives(directive)
            assert len(changes) == 1
            assert changes[0]["new_value"] == 0.35

            # Verify file updated
            with open(os.path.join(config_dir, "news_scout_params.json")) as f:
                data = json.load(f)
            assert data["min_signal_threshold"] == 0.35

    def test_nested_param_path(self, optimizer):
        opt, config_dir, _ = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-01",
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "signal_weights.central_bank",
                     "old_value": 0.9, "new_value": 0.95, "reason": "Fed focus"},
                ],
            }
            opt.apply_directives(directive)
            with open(os.path.join(config_dir, "news_scout_params.json")) as f:
                data = json.load(f)
            assert data["signal_weights"]["central_bank"] == 0.95

    def test_increments_version(self, optimizer):
        opt, config_dir, _ = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-01",
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "min_signal_threshold",
                     "old_value": 0.4, "new_value": 0.35, "reason": "test"},
                ],
            }
            changes = opt.apply_directives(directive)
            assert changes[0]["version"] == 2  # Was 1, now 2

    def test_sends_telegram(self):
        mock_telegram = MagicMock()
        opt = SelfOptimizer(telegram=mock_telegram)
        directive = {
            "week_reviewed": "2026-03-01",
            "parameter_changes": [
                {"target": "news_scout", "parameter": "min_signal_threshold",
                 "old_value": 0.4, "new_value": 0.35, "reason": "test"},
            ],
        }
        opt.apply_directives(directive)
        mock_telegram.send_optimization_summary.assert_called_once()

    def test_empty_changes(self, optimizer):
        opt, _, _ = optimizer
        directive = {"week_reviewed": "2026-03-01", "parameter_changes": []}
        changes = opt.apply_directives(directive)
        assert changes == []

    def test_missing_config_file(self, optimizer):
        opt, config_dir, _ = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-01",
                "parameter_changes": [
                    {"target": "nonexistent_agent", "parameter": "foo",
                     "old_value": 1, "new_value": 2, "reason": "test"},
                ],
            }
            # Should not crash — just warn and skip
            changes = opt.apply_directives(directive)
            assert len(changes) == 1  # Change recorded even if file not found


class TestRollback:
    def test_rollback_reverses_changes(self, optimizer):
        opt, config_dir, data_dir = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.DATA_DIR", data_dir), \
             patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            # Apply a change
            directive = {
                "week_reviewed": "2026-03-01",
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "min_signal_threshold",
                     "old_value": 0.4, "new_value": 0.3, "reason": "test"},
                ],
            }
            opt.apply_directives(directive)

            # Rollback to version 1
            result = opt.rollback(1)
            assert result is True

            # Verify value restored
            with open(os.path.join(config_dir, "news_scout_params.json")) as f:
                data = json.load(f)
            assert data["min_signal_threshold"] == 0.4

    def test_rollback_nothing_to_reverse(self, optimizer):
        opt, config_dir, data_dir = optimizer
        from unittest.mock import patch
        with patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")):
            result = opt.rollback(100)
            assert result is False
