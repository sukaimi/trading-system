"""Tests for Heartbeat — health checks with mocked psutil."""

from unittest.mock import MagicMock, patch

import pytest

from core.heartbeat import Heartbeat


@pytest.fixture
def mock_psutil():
    """Mock psutil to return healthy system metrics."""
    with patch("core.heartbeat.psutil") as mock:
        mock.cpu_percent.return_value = 30.0
        mock.virtual_memory.return_value = MagicMock(percent=50.0)
        mock.disk_usage.return_value = MagicMock(percent=60.0)
        yield mock


@pytest.fixture
def mock_requests():
    """Mock requests to simulate API availability."""
    with patch("core.heartbeat.requests") as mock:
        response = MagicMock()
        response.status_code = 200
        mock.get.return_value = response
        yield mock


class TestHeartbeat:
    def test_all_healthy(self, mock_psutil, mock_requests):
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.all_healthy is True
            assert len(status.failures) == 0
            assert status.checks["cpu"] is True
            assert status.checks["ram"] is True
            assert status.checks["disk"] is True

    def test_high_cpu_fails(self, mock_psutil, mock_requests):
        mock_psutil.cpu_percent.return_value = 95.0
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["cpu"] is False
            assert "cpu" in status.failures
            assert status.all_healthy is False

    def test_high_ram_fails(self, mock_psutil, mock_requests):
        mock_psutil.virtual_memory.return_value = MagicMock(percent=90.0)
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["ram"] is False
            assert "ram" in status.failures

    def test_high_disk_fails(self, mock_psutil, mock_requests):
        mock_psutil.disk_usage.return_value = MagicMock(percent=95.0)
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["disk"] is False

    def test_ibkr_down(self, mock_psutil, mock_requests):
        with patch.object(Heartbeat, "_ping_ibkr", return_value=False), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["ibkr"] is False
            assert "ibkr" in status.failures

    def test_api_keys_missing(self, mock_psutil, mock_requests):
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=False), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["api_keys_present"] is False

    def test_telegram_alert_on_failure(self, mock_psutil, mock_requests):
        mock_psutil.cpu_percent.return_value = 95.0
        mock_telegram = MagicMock()
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat(telegram_notifier=mock_telegram)
            hb.check()
            mock_telegram.send_alert.assert_called_once()

    def test_no_telegram_alert_when_healthy(self, mock_psutil, mock_requests):
        mock_telegram = MagicMock()
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=False):
            hb = Heartbeat(telegram_notifier=mock_telegram)
            hb.check()
            mock_telegram.send_alert.assert_not_called()

    def test_halted_state(self, mock_psutil, mock_requests):
        with patch.object(Heartbeat, "_ping_ibkr", return_value=True), \
             patch.object(Heartbeat, "_check_env_keys", return_value=True), \
             patch.object(Heartbeat, "_is_halted", return_value=True):
            hb = Heartbeat()
            status = hb.check()
            assert status.checks["not_halted"] is False


class TestPingApi:
    def test_successful_ping(self):
        with patch("core.heartbeat.requests") as mock_req:
            mock_req.get.return_value = MagicMock(status_code=200)
            assert Heartbeat._ping_api("https://example.com") is True

    def test_server_error(self):
        with patch("core.heartbeat.requests") as mock_req:
            mock_req.get.return_value = MagicMock(status_code=500)
            assert Heartbeat._ping_api("https://example.com") is False

    def test_connection_error(self):
        with patch("core.heartbeat.requests") as mock_req:
            mock_req.get.side_effect = ConnectionError("unreachable")
            assert Heartbeat._ping_api("https://example.com") is False
