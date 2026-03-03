"""In-process event bus for dashboard live streaming.

Thread-safe: emit() is called from the scheduler thread,
queues are consumed by async WebSocket handlers in the FastAPI thread.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any


class EventBus:
    """Singleton in-process pub/sub for dashboard events."""

    _instance: EventBus | None = None
    _lock = threading.Lock()

    def __new__(cls) -> EventBus:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._subscribers: list[asyncio.Queue] = []
                inst._sub_lock = threading.Lock()
                inst._loop: asyncio.AbstractEventLoop | None = None
                inst._recent: list[dict[str, Any]] = []
                inst._max_recent = 200
                cls._instance = inst
            return cls._instance

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio loop used by the FastAPI thread."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket client. Returns a queue to await on."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a disconnected WebSocket client queue."""
        with self._sub_lock:
            self._subscribers = [s for s in self._subscribers if s is not q]

    def emit(self, category: str, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event from any thread. Thread-safe, non-blocking."""
        event = {
            "category": category,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Store in ring buffer
        self._recent.append(event)
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]

        # Push to all subscriber queues
        if self._loop is None:
            return

        with self._sub_lock:
            for q in self._subscribers:
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, event)
                except (asyncio.QueueFull, RuntimeError):
                    pass  # Drop for slow clients or closed loops

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent events for initial page load."""
        return self._recent[-limit:]


# Module-level singleton
event_bus = EventBus()
