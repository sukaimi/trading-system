"""Tests for SelfOptimizer learning system integration (gather_learning_insights, generate_data_driven_directives)."""

import json
import os
from unittest.mock import patch

import pytest

from core.self_optimizer import SelfOptimizer


@pytest.fixture
def tmp_env(tmp_path):
    """Set up temporary config and data directories for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create risk_params.json
    risk_params = {
        "max_position_pct": 7.0,
        "default_stop_loss_pct": 3.0,
        "stop_loss_atr_mult": 2.0,
    }
    (config_dir / "risk_params.json").write_text(json.dumps(risk_params))

    # Create strategy_version.json
    (config_dir / "strategy_version.json").write_text(
        json.dumps({"version": 1, "last_updated": "2026-01-01"})
    )

    return str(config_dir), str(data_dir)


@pytest.fixture
def optimizer(tmp_env):
    config_dir, data_dir = tmp_env
    learning_files = {
        "stop_recommendations": os.path.join(data_dir, "stop_recommendations.json"),
        "confidence_calibration": os.path.join(data_dir, "confidence_calibration.json"),
        "phantom_analysis": os.path.join(data_dir, "phantom_trades.json"),
        "session_analysis": os.path.join(data_dir, "session_analysis.json"),
    }
    with patch("core.self_optimizer.CONFIG_DIR", config_dir), \
         patch("core.self_optimizer.DATA_DIR", data_dir), \
         patch("core.self_optimizer.OPT_LOG_FILE", os.path.join(data_dir, "optimization_log.json")), \
         patch("core.self_optimizer.VERSION_FILE", os.path.join(config_dir, "strategy_version.json")), \
         patch("core.self_optimizer.DIRECTIVES_FILE", os.path.join(data_dir, "data_driven_directives.json")), \
         patch("core.self_optimizer._LEARNING_FILES", learning_files):
        yield SelfOptimizer(), config_dir, data_dir, learning_files


class TestGatherLearningInsights:
    def test_no_data_files_returns_none_for_each(self, optimizer):
        opt, _, _, _ = optimizer
        result = opt.gather_learning_insights()
        assert result["stop_recommendations"] is None
        assert result["confidence_calibration"] is None
        assert result["phantom_analysis"] is None
        assert result["session_analysis"] is None
        assert isinstance(result["actionable_insights"], list)
        assert len(result["actionable_insights"]) == 0

    def test_loads_stop_recommendations(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        stop_data = {
            "per_asset": {
                "BTC": {"suggested_stop_pct": 4.2, "current_stop_pct": 3.0}
            }
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        result = opt.gather_learning_insights()
        assert result["stop_recommendations"] is not None
        assert result["stop_recommendations"]["per_asset"]["BTC"]["suggested_stop_pct"] == 4.2

    def test_loads_confidence_calibration(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        cal_data = {
            "buckets": {
                "high": {"avg_confidence": 0.75, "actual_win_rate": 0.45, "count": 10}
            }
        }
        with open(learning_files["confidence_calibration"], "w") as f:
            json.dump(cal_data, f)

        result = opt.gather_learning_insights()
        assert result["confidence_calibration"] is not None

    def test_loads_phantom_analysis(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        phantom_data = {"total_missed": 5, "recent": []}
        with open(learning_files["phantom_analysis"], "w") as f:
            json.dump(phantom_data, f)

        result = opt.gather_learning_insights()
        assert result["phantom_analysis"] is not None
        assert result["phantom_analysis"]["total_missed"] == 5

    def test_loads_session_analysis(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        session_data = {"sufficient_data": True, "per_asset": {}, "overall": {}}
        with open(learning_files["session_analysis"], "w") as f:
            json.dump(session_data, f)

        result = opt.gather_learning_insights()
        assert result["session_analysis"] is not None

    def test_corrupt_json_returns_none(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        with open(learning_files["stop_recommendations"], "w") as f:
            f.write("not valid json!!!")

        result = opt.gather_learning_insights()
        assert result["stop_recommendations"] is None


class TestActionableInsights:
    def test_stop_loss_widening_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        stop_data = {
            "per_asset": {
                "BTC": {"suggested_stop_pct": 4.5, "current_stop_pct": 3.0}
            }
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("widened" in i and "BTC" in i for i in insights)

    def test_stop_loss_tightening_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        stop_data = {
            "per_asset": {
                "ETH": {"suggested_stop_pct": 2.0, "current_stop_pct": 3.5}
            }
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("tightened" in i and "ETH" in i for i in insights)

    def test_overconfidence_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        cal_data = {
            "buckets": {
                "high": {"avg_confidence": 0.75, "actual_win_rate": 0.40, "count": 10}
            }
        }
        with open(learning_files["confidence_calibration"], "w") as f:
            json.dump(cal_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("overconfident" in i for i in insights)

    def test_underconfidence_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        cal_data = {
            "buckets": {
                "low": {"avg_confidence": 0.2, "actual_win_rate": 0.50, "count": 10}
            }
        }
        with open(learning_files["confidence_calibration"], "w") as f:
            json.dump(cal_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("underconfident" in i for i in insights)

    def test_phantom_false_kill_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        phantom_data = [
            {"killed_by": "DA", "outcome_pnl_pct": 2.5} for _ in range(4)
        ] + [
            {"killed_by": "DA", "outcome_pnl_pct": -1.0} for _ in range(2)
        ]
        with open(learning_files["phantom_analysis"], "w") as f:
            json.dump(phantom_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("DA" in i and "false kill" in i for i in insights)

    def test_session_outperformance_insight(self, optimizer):
        opt, _, _, learning_files = optimizer
        session_data = {
            "sufficient_data": True,
            "per_asset": {
                "BTC": {
                    "asian": {"trades": 10, "win_rate": 0.7, "avg_pnl_pct": 2.5, "score": 0.72},
                    "european": {"trades": 3, "win_rate": 0.5, "avg_pnl_pct": 0.5, "score": 0.5},
                    "us": {"trades": 8, "win_rate": 0.4, "avg_pnl_pct": -0.5, "score": 0.43},
                    "best_session": "asian",
                }
            },
            "overall": {},
        }
        with open(learning_files["session_analysis"], "w") as f:
            json.dump(session_data, f)

        result = opt.gather_learning_insights()
        insights = result["actionable_insights"]
        assert any("Asian" in i and "BTC" in i for i in insights)

    def test_no_insights_with_insufficient_data(self, optimizer):
        """Small differences or few trades should not generate insights."""
        opt, _, _, learning_files = optimizer
        stop_data = {
            "per_asset": {
                "BTC": {"suggested_stop_pct": 3.1, "current_stop_pct": 3.0}
            }
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        result = opt.gather_learning_insights()
        # Difference is only 0.1, below the 0.3 threshold
        assert len(result["actionable_insights"]) == 0


class TestGenerateDataDrivenDirectives:
    def test_produces_valid_directive_structure(self, optimizer):
        opt, config_dir, _, learning_files = optimizer
        stop_data = {
            "global_suggested_stop_pct": 4.0,
            "per_asset": {},
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        directives = opt.generate_data_driven_directives()
        assert len(directives) >= 1
        d = directives[0]
        assert "target" in d
        assert "parameter" in d
        assert "old_value" in d
        assert "new_value" in d
        assert "reason" in d
        assert "confidence" in d
        assert "source" in d

    def test_stop_directive_from_global_recommendation(self, optimizer):
        opt, config_dir, _, learning_files = optimizer
        stop_data = {
            "global_suggested_stop_pct": 4.5,
            "per_asset": {},
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        directives = opt.generate_data_driven_directives()
        stop_dirs = [d for d in directives if d["source"] == "adaptive_stops"]
        assert len(stop_dirs) >= 1
        assert stop_dirs[0]["old_value"] == 3.0  # from risk_params
        assert stop_dirs[0]["new_value"] == 4.5

    def test_no_directive_when_values_similar(self, optimizer):
        opt, _, _, learning_files = optimizer
        stop_data = {
            "global_suggested_stop_pct": 3.1,  # Close to current 3.0
            "per_asset": {},
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        directives = opt.generate_data_driven_directives()
        stop_dirs = [d for d in directives if d["source"] == "adaptive_stops"]
        assert len(stop_dirs) == 0

    def test_directives_persisted_to_json(self, optimizer):
        opt, _, data_dir, learning_files = optimizer
        stop_data = {
            "global_suggested_stop_pct": 5.0,
            "per_asset": {},
        }
        with open(learning_files["stop_recommendations"], "w") as f:
            json.dump(stop_data, f)

        opt.generate_data_driven_directives()

        directives_file = os.path.join(data_dir, "data_driven_directives.json")
        assert os.path.exists(directives_file)
        with open(directives_file) as f:
            saved = json.load(f)
        assert "directives" in saved
        assert "generated_at" in saved
        assert saved["count"] >= 1

    def test_empty_directives_when_no_data(self, optimizer):
        opt, _, _, _ = optimizer
        directives = opt.generate_data_driven_directives()
        assert directives == []

    def test_phantom_directive_with_high_false_kill_rate(self, optimizer):
        opt, _, _, learning_files = optimizer
        # Need 10+ trades to trigger
        phantom_data = [
            {"killed_by": "DA", "outcome_pnl_pct": 2.0} for _ in range(8)
        ] + [
            {"killed_by": "DA", "outcome_pnl_pct": -1.0} for _ in range(4)
        ]
        with open(learning_files["phantom_analysis"], "w") as f:
            json.dump(phantom_data, f)

        directives = opt.generate_data_driven_directives()
        phantom_dirs = [d for d in directives if d["source"] == "phantom_tracker"]
        assert len(phantom_dirs) == 1
        assert "profitable" in phantom_dirs[0]["reason"]

    def test_session_directive_with_significant_gap(self, optimizer):
        opt, _, _, learning_files = optimizer
        session_data = {
            "sufficient_data": True,
            "per_asset": {},
            "overall": {
                "asian": {"trades": 10, "score": 0.8},
                "european": {"trades": 8, "score": 0.5},
                "us": {"trades": 12, "score": 0.6},
            },
        }
        with open(learning_files["session_analysis"], "w") as f:
            json.dump(session_data, f)

        directives = opt.generate_data_driven_directives()
        session_dirs = [d for d in directives if d["source"] == "session_analyzer"]
        assert len(session_dirs) == 1
        assert session_dirs[0]["new_value"]["best"] == "asian"

    def test_confidence_calibration_directive(self, optimizer):
        opt, _, _, learning_files = optimizer
        cal_data = {"overall_correction": -0.12}
        with open(learning_files["confidence_calibration"], "w") as f:
            json.dump(cal_data, f)

        directives = opt.generate_data_driven_directives()
        cal_dirs = [d for d in directives if d["source"] == "confidence_calibrator"]
        assert len(cal_dirs) == 1
        assert cal_dirs[0]["new_value"] == -0.12
