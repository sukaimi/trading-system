"""Tests for GA4Tracker — mock HTTP requests for unit testing."""

from unittest.mock import MagicMock, patch

import pytest


class TestGA4TrackerInit:
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_enabled_with_both_vars(self):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()
        assert tracker._enabled is True

    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "", "GA4_API_SECRET": ""}, clear=False)
    def test_disabled_without_vars(self):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()
        assert tracker._enabled is False

    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": ""}, clear=False)
    def test_disabled_without_secret(self):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()
        assert tracker._enabled is False


class TestGA4TrackerSend:
    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_send_event_posts_to_ga4(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        mock_post.return_value.status_code = 204
        tracker = GA4Tracker()
        tracker._send("test_event", {"asset": "BTC"})

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["params"]["measurement_id"] == "G-TEST123"
        assert call_kwargs.kwargs["params"]["api_secret"] == "secret123"
        payload = call_kwargs.kwargs["json"]
        assert payload["events"][0]["name"] == "test_event"
        assert payload["events"][0]["params"]["asset"] == "BTC"
        assert "session_id" in payload["events"][0]["params"]

    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "", "GA4_API_SECRET": ""}, clear=False)
    def test_send_event_noop_when_disabled(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()
        tracker.send_event("test_event")
        mock_post.assert_not_called()

    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_send_handles_network_error(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        mock_post.side_effect = Exception("Network error")
        tracker = GA4Tracker()
        # Should not raise
        tracker._send("test_event", {})


class TestGA4TrackerEventHandling:
    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_handle_mapped_event(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        mock_post.return_value.status_code = 204
        tracker = GA4Tracker()

        event = {
            "category": "pipeline",
            "event_type": "trade_executed",
            "data": {"asset": "BTC", "direction": "long", "quantity": "0.001"},
        }
        tracker.handle_event(event)
        # send_event spawns a thread — wait briefly for it to fire
        import time
        time.sleep(0.1)
        assert mock_post.called

    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_unmapped_event_ignored(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()

        event = {
            "category": "scheduler",
            "event_type": "task_run",
            "data": {"task_name": "heartbeat"},
        }
        tracker.handle_event(event)
        import time
        time.sleep(0.1)
        mock_post.assert_not_called()

    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "", "GA4_API_SECRET": ""}, clear=False)
    def test_handle_event_noop_when_disabled(self):
        from core.ga4_tracker import GA4Tracker
        tracker = GA4Tracker()
        # Should not raise
        tracker.handle_event({
            "category": "pipeline",
            "event_type": "trade_executed",
            "data": {},
        })

    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_stop_loss_event_mapped(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        mock_post.return_value.status_code = 204
        tracker = GA4Tracker()
        event = {
            "category": "stop_loss",
            "event_type": "triggered",
            "data": {"asset": "BTC", "price": "65000"},
        }
        tracker.handle_event(event)
        import time
        time.sleep(0.1)
        assert mock_post.called

    @patch("core.ga4_tracker.requests.post")
    @patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST123", "GA4_API_SECRET": "secret123"})
    def test_circuit_breaker_event_mapped(self, mock_post):
        from core.ga4_tracker import GA4Tracker
        mock_post.return_value.status_code = 204
        tracker = GA4Tracker()
        event = {
            "category": "circuit_breaker",
            "event_type": "triggered",
            "data": {"reason": "daily loss exceeded"},
        }
        tracker.handle_event(event)
        import time
        time.sleep(0.1)
        assert mock_post.called
