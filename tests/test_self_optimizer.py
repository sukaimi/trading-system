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
         patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")), \
         patch("core.self_optimizer.APPLIED_DIRECTIVES_FILE", os.path.join(data_dir, "applied_directives.json")):
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

    def test_sends_telegram(self, optimizer, tmp_config):
        config_dir, data_dir = tmp_config
        from unittest.mock import patch
        mock_telegram = MagicMock()
        opt = SelfOptimizer(telegram=mock_telegram)
        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.DATA_DIR", data_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")), \
             patch("core.self_optimizer.APPLIED_DIRECTIVES_FILE", os.path.join(data_dir, "applied_directives.json")):
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


class TestDottedKeyResolution:
    """Tests for _resolve_path handling dotted keys like '0.4-0.5'."""

    def test_dotted_key_not_split(self, optimizer):
        """Ensure 'confidence_size_map.0.4-0.5' updates the correct key."""
        opt, config_dir, _ = optimizer
        from unittest.mock import patch

        # Create a config with dotted keys
        da_params = {
            "confidence_size_map": {
                "0.4-0.5": 2.0,
                "0.5-0.65": 4.0,
                "0.65+": 7.0,
            }
        }
        with open(os.path.join(config_dir, "devils_advocate_params.json"), "w") as f:
            json.dump(da_params, f)

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "devils_advocate",
                    "parameter": "confidence_size_map.0.4-0.5",
                    "old_value": 2.0,
                    "new_value": 1.8,
                    "reason": "test",
                }],
            }
            opt.apply_directives(directive)

        with open(os.path.join(config_dir, "devils_advocate_params.json")) as f:
            data = json.load(f)
        assert data["confidence_size_map"]["0.4-0.5"] == 1.8
        # No garbage nested "0" key
        assert "0" not in data["confidence_size_map"]


class TestTypeCoercion:
    """Tests for _coerce_type preserving original types."""

    def test_float_coerced_to_int(self, optimizer):
        """Ensure float 5.0 becomes int 5 when original is int."""
        opt, config_dir, _ = optimizer
        from unittest.mock import patch

        # Create config with int value
        params = {"max_open_positions": 10, "min_threshold": 0.4}
        with open(os.path.join(config_dir, "market_analyst_params.json"), "w") as f:
            json.dump(params, f)

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "market_analyst",
                    "parameter": "max_open_positions",
                    "old_value": 10,
                    "new_value": 5.0,  # Float from LLM
                    "reason": "test",
                }],
            }
            opt.apply_directives(directive)

        with open(os.path.join(config_dir, "market_analyst_params.json")) as f:
            data = json.load(f)
        assert data["max_open_positions"] == 5
        assert isinstance(data["max_open_positions"], int)


class TestBoundsValidation:
    """Tests for _check_bounds clamping values to safe ranges."""

    def test_clamps_out_of_bounds_high(self, optimizer):
        """Values above max are clamped down."""
        opt, config_dir, _ = optimizer
        from unittest.mock import patch

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "risk",
                    "parameter": "max_daily_loss_pct",
                    "old_value": 5.0,
                    "new_value": 50.0,  # Way too high
                    "reason": "test",
                }],
            }
            opt.apply_directives(directive)

        with open(os.path.join(config_dir, "risk_params.json")) as f:
            data = json.load(f)
        assert data["max_daily_loss_pct"] == 10.0  # Clamped to max bound

    def test_clamps_out_of_bounds_low(self, optimizer):
        """Values below min are clamped up."""
        opt, config_dir, _ = optimizer
        from unittest.mock import patch

        # Create news_scout config with int field
        ns_params = {
            "signal_weights": {"central_bank": 0.9},
            "min_signal_threshold": 0.4,
        }
        with open(os.path.join(config_dir, "news_scout_params.json"), "w") as f:
            json.dump(ns_params, f)

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "news_scout",
                    "parameter": "min_signal_threshold",
                    "old_value": 0.4,
                    "new_value": 0.05,  # Too low
                    "reason": "test",
                }],
            }
            opt.apply_directives(directive)

        with open(os.path.join(config_dir, "news_scout_params.json")) as f:
            data = json.load(f)
        assert data["min_signal_threshold"] == 0.20  # Clamped to min bound


class TestDeduplication:
    """Tests for directive fingerprint deduplication."""

    def test_duplicate_directive_skipped(self, optimizer):
        """Applying the same directive twice is idempotent."""
        opt, config_dir, data_dir = optimizer
        from unittest.mock import patch

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.DATA_DIR", data_dir), \
             patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")), \
             patch("core.self_optimizer.APPLIED_DIRECTIVES_FILE", os.path.join(data_dir, "applied_directives.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "news_scout",
                    "parameter": "min_signal_threshold",
                    "old_value": 0.4,
                    "new_value": 0.35,
                    "reason": "test",
                }],
            }

            # First application
            changes1 = opt.apply_directives(directive)
            assert len(changes1) == 1

            # Second application (duplicate)
            changes2 = opt.apply_directives(directive)
            assert len(changes2) == 0  # Skipped

    def test_version_increments_once_per_review(self, optimizer):
        """Multiple parameter changes in one review produce one version bump."""
        opt, config_dir, data_dir = optimizer
        from unittest.mock import patch

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.DATA_DIR", data_dir), \
             patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")), \
             patch("core.self_optimizer.APPLIED_DIRECTIVES_FILE", os.path.join(data_dir, "applied_directives.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [
                    {"target": "news_scout", "parameter": "min_signal_threshold",
                     "old_value": 0.4, "new_value": 0.35, "reason": "a"},
                    {"target": "risk", "parameter": "max_daily_loss_pct",
                     "old_value": 5.0, "new_value": 4.0, "reason": "b"},
                ],
            }
            changes = opt.apply_directives(directive)
            assert len(changes) == 2
            # Both changes share the same version
            assert changes[0]["version"] == changes[1]["version"] == 2


class TestTrailingNewline:
    """Tests for trailing newline on JSON files."""

    def test_config_file_ends_with_newline(self, optimizer):
        """Config files written by _update_config end with newline."""
        opt, config_dir, _ = optimizer
        from unittest.mock import patch

        with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
             patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")):
            directive = {
                "week_reviewed": "2026-03-15",
                "parameter_changes": [{
                    "target": "news_scout",
                    "parameter": "min_signal_threshold",
                    "old_value": 0.4,
                    "new_value": 0.35,
                    "reason": "test",
                }],
            }
            opt.apply_directives(directive)

        filepath = os.path.join(config_dir, "news_scout_params.json")
        with open(filepath, "rb") as f:
            content = f.read()
        assert content.endswith(b"\n")
