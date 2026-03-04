"""GA4 Measurement Protocol tracker — sends server-side events to Google Analytics.

Subscribes to the event_bus and forwards selected trading events to GA4.
Reads GA4_MEASUREMENT_ID and GA4_API_SECRET from environment variables.
If either is missing, the tracker silently disables itself.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

import requests

from core.event_bus import event_bus
from core.logger import setup_logger

log = setup_logger("trading.ga4")

GA4_ENDPOINT = "https://www.google-analytics.com/mp/collect"
GA4_DEBUG_ENDPOINT = "https://www.google-analytics.com/debug/mp/collect"

# Events we forward to GA4 (event_bus category.event_type → GA4 event name)
# Keys must match exact strings in event_bus.emit() calls across the codebase.
_EVENT_MAP: dict[str, str] = {
    "pipeline.thesis_generated": "trade_signal",
    "pipeline.trade_executed": "trade_executed",
    "pipeline.trade_killed": "trade_killed",
    "stop_loss.triggered": "stop_loss_triggered",
    "take_profit.triggered": "take_profit_triggered",
    "circuit_breaker.triggered": "circuit_breaker_triggered",
}


class GA4Tracker:
    """Sends server-side events to GA4 via Measurement Protocol."""

    def __init__(self) -> None:
        self._measurement_id = os.getenv("GA4_MEASUREMENT_ID", "")
        self._api_secret = os.getenv("GA4_API_SECRET", "")
        self._client_id = self._get_or_create_client_id()
        self._session_id = str(int(time.time()))
        self._enabled = bool(self._measurement_id and self._api_secret)

        if not self._enabled:
            log.info("GA4 tracker disabled (missing GA4_MEASUREMENT_ID or GA4_API_SECRET)")

    def _get_or_create_client_id(self) -> str:
        """Persistent client ID stored in data/ga4_client_id."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        path = os.path.join(data_dir, "ga4_client_id")
        try:
            if os.path.exists(path):
                with open(path) as f:
                    cid = f.read().strip()
                    if cid:
                        return cid
        except OSError:
            pass
        cid = str(uuid.uuid4())
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(cid)
        except OSError:
            pass
        return cid

    def send_event(self, name: str, params: dict[str, Any] | None = None) -> None:
        """Send a single event to GA4. Non-blocking (fires in background thread)."""
        if not self._enabled:
            return
        threading.Thread(
            target=self._send,
            args=(name, params or {}),
            daemon=True,
        ).start()

    def _send(self, name: str, params: dict[str, Any]) -> None:
        """Actual HTTP POST to GA4 Measurement Protocol."""
        params = {**params, "session_id": self._session_id, "engagement_time_msec": "1"}
        payload = {
            "client_id": self._client_id,
            "events": [{"name": name, "params": params}],
        }
        try:
            resp = requests.post(
                GA4_ENDPOINT,
                params={
                    "measurement_id": self._measurement_id,
                    "api_secret": self._api_secret,
                },
                json=payload,
                timeout=5,
            )
            if resp.status_code != 204:
                log.debug("GA4 response %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            log.debug("GA4 send failed: %s", e)

    def start(self) -> None:
        """Subscribe to event_bus and forward matching events to GA4."""
        if not self._enabled:
            return
        event_bus.add_listener(self.handle_event)
        log.info("GA4 tracker started (measurement_id=%s)", self._measurement_id)
        self.send_event("system_startup")

    def handle_event(self, event: dict[str, Any]) -> None:
        """Process an event_bus event — forward to GA4 if mapped."""
        if not self._enabled:
            return
        key = f"{event.get('category', '')}.{event.get('event_type', '')}"
        ga4_name = _EVENT_MAP.get(key)
        if ga4_name:
            params: dict[str, Any] = {}
            data = event.get("data", {})
            # Extract useful params (GA4 has 25 param limit, keep it lean)
            for field in ("asset", "direction", "quantity", "price", "reason", "agent"):
                if field in data:
                    params[field] = str(data[field])[:100]
            self.send_event(ga4_name, params)


# Module-level singleton
ga4_tracker = GA4Tracker()
