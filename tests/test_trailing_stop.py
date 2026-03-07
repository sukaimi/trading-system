"""Tests for the trailing stop-loss mechanism in core/pipeline.py."""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import LLMClient
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager


@pytest.fixture
def trailing_risk_config(risk_config):
    """Risk config extended with trailing stop parameters."""
    return {
        **risk_config,
        "trailing_stop_activation_pct": 2.0,
        "trailing_stop_distance_pct": 1.5,
        "trailing_stop_atr_mult": 1.0,
    }


@pytest.fixture
def pipeline(trailing_risk_config):
    portfolio = PortfolioState()
    risk_manager = RiskManager(config=trailing_risk_config)
    executor = MagicMock()
    telegram = MagicMock()
    llm = LLMClient(mock_mode=True)
    p = TradingPipeline(
        portfolio=portfolio,
        risk_manager=risk_manager,
        executor=executor,
        telegram=telegram,
        llm_client=llm,
    )
    # Override _risk_params so the method reads our test config
    p._risk_params = trailing_risk_config
    return p


def _add_position(pipeline, trade_id, asset, direction, entry, stop, qty=0.01,
                   pct=5.0, supporting_data=None):
    """Helper to add a position with stop-loss fields."""
    pos = {
        "trade_id": trade_id,
        "asset": asset,
        "direction": direction,
        "entry_price": entry,
        "position_size_pct": pct,
        "quantity": qty,
        "stop_loss_price": stop,
    }
    if supporting_data is not None:
        pos["supporting_data"] = supporting_data
    pipeline._portfolio.add_position(pos)


# ── Activation Threshold Tests ────────────────────────────────────────


class TestActivationThreshold:

    @patch("core.pipeline.MarketDataFetcher")
    def test_no_activation_below_threshold_long(self, MockMDF, pipeline):
        """Long position with <2% gain should NOT activate trailing stop."""
        MockMDF.return_value.get_price.return_value = {"price": 66000}  # 1.54% above 65000
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        result = pipeline.update_trailing_stops()

        assert result == []
        pos = pipeline._portfolio.open_positions[0]
        assert pos["stop_loss_price"] == 61750  # unchanged
        assert pos.get("trailing_stop_active") is None

    @patch("core.pipeline.MarketDataFetcher")
    def test_no_activation_below_threshold_short(self, MockMDF, pipeline):
        """Short position with <2% favorable move should NOT activate trailing stop."""
        MockMDF.return_value.get_price.return_value = {"price": 64000}  # 1.54% below 65000
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)

        result = pipeline.update_trailing_stops()

        assert result == []
        pos = pipeline._portfolio.open_positions[0]
        assert pos["stop_loss_price"] == 68250  # unchanged

    @patch("core.pipeline.MarketDataFetcher")
    def test_no_activation_when_price_unfavorable_long(self, MockMDF, pipeline):
        """Long position with price below entry should NOT activate."""
        MockMDF.return_value.get_price.return_value = {"price": 63000}  # below entry
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        result = pipeline.update_trailing_stops()

        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_no_activation_when_price_unfavorable_short(self, MockMDF, pipeline):
        """Short position with price above entry should NOT activate."""
        MockMDF.return_value.get_price.return_value = {"price": 67000}  # above entry
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)

        result = pipeline.update_trailing_stops()

        assert result == []


# ── Long Position Trailing ────────────────────────────────────────────


class TestLongTrailing:

    @patch("core.pipeline.MarketDataFetcher")
    def test_trailing_activates_long(self, MockMDF, pipeline):
        """Long position with >2% gain should activate trailing and move stop up."""
        # 3.08% above entry — well above 2% threshold
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        assert result[0]["trade_id"] == "t1"
        assert result[0]["first_activation"] is True
        pos = pipeline._portfolio.open_positions[0]
        # Trail distance = 67000 * 0.015 = 1005, new_stop = 67000 - 1005 = 65995
        assert pos["stop_loss_price"] == 65995.0
        assert pos["trailing_stop_active"] is True
        assert pos["original_stop_loss_price"] == 61750

    @patch("core.pipeline.MarketDataFetcher")
    def test_trailing_moves_stop_higher_on_subsequent_calls(self, MockMDF, pipeline):
        """As price rises, trailing stop should ratchet upward."""
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        # First call: 3.08% gain
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        pipeline.update_trailing_stops()
        first_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        assert first_stop == 65995.0

        # Second call: price went higher
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        pipeline.update_trailing_stops()
        second_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        # Trail distance = 69000 * 0.015 = 1035, new_stop = 69000 - 1035 = 67965
        assert second_stop == 67965.0
        assert second_stop > first_stop


# ── Short Position Trailing ───────────────────────────────────────────


class TestShortTrailing:

    @patch("core.pipeline.MarketDataFetcher")
    def test_trailing_activates_short(self, MockMDF, pipeline):
        """Short position with >2% favorable move should activate trailing and move stop down."""
        # Price dropped to 63000 — 3.08% below entry of 65000
        MockMDF.return_value.get_price.return_value = {"price": 63000}
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        assert result[0]["first_activation"] is True
        pos = pipeline._portfolio.open_positions[0]
        # Trail distance = 63000 * 0.015 = 945, new_stop = 63000 + 945 = 63945
        assert pos["stop_loss_price"] == 63945.0
        assert pos["trailing_stop_active"] is True
        assert pos["original_stop_loss_price"] == 68250

    @patch("core.pipeline.MarketDataFetcher")
    def test_trailing_moves_stop_lower_on_subsequent_calls_short(self, MockMDF, pipeline):
        """As price drops, trailing stop for shorts should ratchet downward."""
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)

        # First call: price at 63000
        MockMDF.return_value.get_price.return_value = {"price": 63000}
        pipeline.update_trailing_stops()
        first_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        assert first_stop == 63945.0

        # Second call: price dropped further to 61000
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        pipeline.update_trailing_stops()
        second_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        # Trail distance = 61000 * 0.015 = 915, new_stop = 61000 + 915 = 61915
        assert second_stop == 61915.0
        assert second_stop < first_stop


# ── Stop Never Moves Backward ────────────────────────────────────────


class TestNoBackwardMovement:

    @patch("core.pipeline.MarketDataFetcher")
    def test_long_stop_never_moves_down(self, MockMDF, pipeline):
        """If price retraces after trailing is active, stop must NOT move backward."""
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        # First: price at 69000
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        pipeline.update_trailing_stops()
        high_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        # new_stop = 69000 - 1035 = 67965
        assert high_stop == 67965.0

        # Price retraces to 67000 — still above activation threshold but stop would be lower
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        result = pipeline.update_trailing_stops()
        # new_stop would be 67000 - 1005 = 65995, which is < 67965 — should NOT update
        assert result == []
        assert pipeline._portfolio.open_positions[0]["stop_loss_price"] == 67965.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_short_stop_never_moves_up(self, MockMDF, pipeline):
        """If price bounces after trailing is active for short, stop must NOT move backward."""
        _add_position(pipeline, "t1", "BTC", "short", 65000, 68250)

        # First: price at 61000
        MockMDF.return_value.get_price.return_value = {"price": 61000}
        pipeline.update_trailing_stops()
        low_stop = pipeline._portfolio.open_positions[0]["stop_loss_price"]
        # new_stop = 61000 + 915 = 61915
        assert low_stop == 61915.0

        # Price bounces to 63000 — stop would be 63945 which is > 61915 — should NOT update
        MockMDF.return_value.get_price.return_value = {"price": 63000}
        result = pipeline.update_trailing_stops()
        assert result == []
        assert pipeline._portfolio.open_positions[0]["stop_loss_price"] == 61915.0


# ── ATR-based Trail Distance ─────────────────────────────────────────


class TestATRTrailDistance:

    @patch("core.pipeline.MarketDataFetcher")
    def test_atr_based_trail_when_available(self, MockMDF, pipeline):
        """When ATR is available in supporting_data, use it for trail distance."""
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        # ATR of 2000 * 1.0 mult = 2000, vs flat 67000 * 0.015 = 1005 — ATR wins
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750,
                       supporting_data={"atr_14": 2000})

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        pos = pipeline._portfolio.open_positions[0]
        # Trail distance = max(2000 * 1.0, 67000 * 0.015) = max(2000, 1005) = 2000
        # new_stop = 67000 - 2000 = 65000
        assert pos["stop_loss_price"] == 65000.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_flat_pct_wins_when_larger_than_atr(self, MockMDF, pipeline):
        """When flat % trail is larger than ATR-based, use flat %."""
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        # ATR of 500 * 1.0 mult = 500, vs flat 67000 * 0.015 = 1005 — flat wins
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750,
                       supporting_data={"atr_14": 500})

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        pos = pipeline._portfolio.open_positions[0]
        # Trail distance = max(500, 1005) = 1005
        # new_stop = 67000 - 1005 = 65995
        assert pos["stop_loss_price"] == 65995.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_flat_pct_fallback_when_no_atr(self, MockMDF, pipeline):
        """When no ATR available, use flat % for trail distance."""
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)  # no supporting_data

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        pos = pipeline._portfolio.open_positions[0]
        # Trail distance = 67000 * 0.015 = 1005
        # new_stop = 67000 - 1005 = 65995
        assert pos["stop_loss_price"] == 65995.0

    @patch("core.pipeline.MarketDataFetcher")
    def test_zero_atr_uses_flat_pct(self, MockMDF, pipeline):
        """When ATR is 0, use flat % fallback."""
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750,
                       supporting_data={"atr_14": 0})

        result = pipeline.update_trailing_stops()

        pos = pipeline._portfolio.open_positions[0]
        assert pos["stop_loss_price"] == 65995.0  # flat % fallback


# ── Original Stop Preservation ────────────────────────────────────────


class TestOriginalStopPreservation:

    @patch("core.pipeline.MarketDataFetcher")
    def test_original_stop_preserved_on_first_activation(self, MockMDF, pipeline):
        """original_stop_loss_price should be set on first activation only."""
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        MockMDF.return_value.get_price.return_value = {"price": 67000}
        pipeline.update_trailing_stops()
        pos = pipeline._portfolio.open_positions[0]
        assert pos["original_stop_loss_price"] == 61750

        # Second update — original should NOT change
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        pipeline.update_trailing_stops()
        pos = pipeline._portfolio.open_positions[0]
        assert pos["original_stop_loss_price"] == 61750  # still the original


# ── Integration: Trailing + Stop-Loss Breach ──────────────────────────


class TestIntegration:

    @patch("core.pipeline.MarketDataFetcher")
    def test_trailing_update_then_breach(self, MockMDF, pipeline):
        """After trailing ratchets up, check_stop_losses should detect breach at the new level."""
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)

        # Step 1: Price rises to 69000 — trailing activates, stop moves to 67965
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        pipeline.update_trailing_stops()
        pos = pipeline._portfolio.open_positions[0]
        assert pos["stop_loss_price"] == 67965.0

        # Step 2: Price drops to 67500 — below the trailed stop of 67965
        MockMDF.return_value.get_price.return_value = {"price": 67500}
        pipeline._executor.execute.return_value = {
            "type": "order_confirmation",
            "order_id": 99, "asset": "BTC",
            "fill_price": 67500, "quantity": 0.01, "status": "Filled",
        }

        result = pipeline.check_stop_losses()

        assert len(result) == 1
        assert result[0]["trade_id"] == "t1"
        # PnL = (67500 - 65000) * 0.01 = +25 (profit locked in!)
        assert result[0]["pnl"] == 25.0
        assert len(pipeline._portfolio.open_positions) == 0


# ── Edge Cases / Error Handling ───────────────────────────────────────


class TestEdgeCases:

    def test_no_positions(self, pipeline):
        result = pipeline.update_trailing_stops()
        assert result == []

    def test_halted_system_skips(self, pipeline):
        pipeline._portfolio.halted = True
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)
        result = pipeline.update_trailing_stops()
        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_position_without_stop_loss_skipped(self, MockMDF, pipeline):
        pipeline._portfolio.add_position({
            "trade_id": "t1", "asset": "BTC", "direction": "long",
            "entry_price": 65000, "position_size_pct": 5.0,
        })
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        result = pipeline.update_trailing_stops()
        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_price_fetch_failure_skips(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.side_effect = Exception("API down")
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        result = pipeline.update_trailing_stops()
        assert result == []
        assert pipeline._portfolio.open_positions[0]["stop_loss_price"] == 61750

    @patch("core.pipeline.MarketDataFetcher")
    def test_zero_price_skips(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 0}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        result = pipeline.update_trailing_stops()
        assert result == []

    @patch("core.pipeline.MarketDataFetcher")
    def test_portfolio_persisted_after_update(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        with patch.object(pipeline._portfolio, "persist") as mock_persist:
            pipeline.update_trailing_stops()
            mock_persist.assert_called_once()

    @patch("core.pipeline.MarketDataFetcher")
    def test_telegram_called_on_first_activation(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        pipeline.update_trailing_stops()

        pipeline._telegram.send_alert.assert_called_once()
        alert_msg = pipeline._telegram.send_alert.call_args[0][0]
        assert "TRAILING STOP activated" in alert_msg

    @patch("core.pipeline.MarketDataFetcher")
    def test_telegram_not_called_on_subsequent_updates(self, MockMDF, pipeline):
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)

        # First activation
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        pipeline.update_trailing_stops()
        pipeline._telegram.send_alert.assert_called_once()

        pipeline._telegram.send_alert.reset_mock()

        # Second update — no Telegram alert
        MockMDF.return_value.get_price.return_value = {"price": 69000}
        pipeline.update_trailing_stops()
        pipeline._telegram.send_alert.assert_not_called()

    @patch("core.pipeline.MarketDataFetcher")
    def test_telegram_failure_still_updates_stop(self, MockMDF, pipeline):
        MockMDF.return_value.get_price.return_value = {"price": 67000}
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750)
        pipeline._telegram.send_alert.side_effect = Exception("Telegram down")

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        pos = pipeline._portfolio.open_positions[0]
        assert pos["stop_loss_price"] == 65995.0
        assert pos["trailing_stop_active"] is True

    @patch("core.pipeline.MarketDataFetcher")
    def test_multiple_positions_independent(self, MockMDF, pipeline):
        """Multiple positions should be updated independently."""
        def price_side_effect(symbol):
            return {"BTC": {"price": 67000}, "ETH": {"price": 3200}}.get(symbol, {"price": 0})
        MockMDF.return_value.get_price.side_effect = price_side_effect

        # BTC: 3.08% above entry — should trail
        _add_position(pipeline, "t1", "BTC", "long", 65000, 61750, qty=0.01)
        # ETH: 0.63% above entry — below threshold, should NOT trail
        _add_position(pipeline, "t2", "ETH", "long", 3180, 3021, qty=0.1)

        result = pipeline.update_trailing_stops()

        assert len(result) == 1
        assert result[0]["asset"] == "BTC"
        # BTC stop updated, ETH stop unchanged
        btc_pos = next(p for p in pipeline._portfolio.open_positions if p["asset"] == "BTC")
        eth_pos = next(p for p in pipeline._portfolio.open_positions if p["asset"] == "ETH")
        assert btc_pos["stop_loss_price"] == 65995.0
        assert eth_pos["stop_loss_price"] == 3021
