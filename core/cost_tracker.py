"""LLM cost tracking — instruments LLMClient without changing its interface.

Uses actual token counts from API responses when available, falls back to
character-length estimation (~4 chars/token). Applies per-provider pricing
to track spend per call, per provider, and per agent.
State persists to data/cost_state.json for crash recovery.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from core.event_bus import event_bus

# Pricing per 1M tokens (input / output)
PRICING: dict[str, dict[str, float]] = {
    "deepseek": {"input": 0.14, "output": 0.28},
    "kimi": {"input": 0.50, "output": 1.00},
    "anthropic": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 0.075, "output": 0.30},
}

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cost_state.json")


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


class CostTracker:
    """Thread-safe LLM API cost tracker with JSON persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[dict[str, Any]] = []
        self._total_usd: float = 0.0
        self._by_provider: dict[str, float] = {}
        self._by_agent: dict[str, float] = {}
        self._total_call_count: int = 0
        self._load()

    def record(
        self,
        provider: str,
        agent: str,
        input_text: str,
        output_text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Record one LLM call and emit a cost event.

        Uses actual token counts from API response when provided,
        falls back to character-length estimation otherwise.
        """
        in_tok = input_tokens if input_tokens is not None else _estimate_tokens(input_text)
        out_tok = output_tokens if output_tokens is not None else _estimate_tokens(output_text)

        prices = PRICING.get(provider, {"input": 0, "output": 0})
        cost = (
            in_tok * prices["input"] + out_tok * prices["output"]
        ) / 1_000_000

        actual = input_tokens is not None

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "agent": agent,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "tokens_actual": actual,
            "cost_usd": round(cost, 6),
        }

        with self._lock:
            self._calls.append(record)
            self._total_usd += cost
            self._total_call_count += 1
            self._by_provider[provider] = self._by_provider.get(provider, 0) + cost
            self._by_agent[agent] = self._by_agent.get(agent, 0) + cost
            if len(self._calls) > 1000:
                self._calls = self._calls[-1000:]

        event_bus.emit("cost", "llm_call", record)
        self._persist()
        return record

    # Daily spend limits per provider (USD)
    DAILY_LIMITS: dict[str, float] = {
        "anthropic": 0.15,  # ~$4.50/month max
        "kimi": 0.05,
        "deepseek": 0.50,  # raised from 0.03 — actual usage is ~$0.66/day
    }

    def check_budget(self, provider: str) -> bool:
        """Return True if daily spend for provider is under its daily limit."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            today_spend = sum(
                r["cost_usd"]
                for r in self._calls
                if r.get("provider") == provider
                and r.get("timestamp", "").startswith(today)
            )
        limit = self.DAILY_LIMITS.get(provider, 1.0)
        if today_spend >= limit:
            from core.logger import setup_logger
            _log = setup_logger("trading.cost_tracker")
            _log.warning(
                "%s daily budget exhausted: $%.4f / $%.2f",
                provider, today_spend, limit,
            )
        return today_spend < limit

    def summary(self) -> dict[str, Any]:
        """Return cost summary for the dashboard."""
        with self._lock:
            return {
                "total_usd": round(self._total_usd, 4),
                "by_provider": {k: round(v, 4) for k, v in self._by_provider.items()},
                "by_agent": {k: round(v, 4) for k, v in self._by_agent.items()},
                "call_count": self._total_call_count,
                "recent_calls": self._calls[-20:],
            }

    def _persist(self) -> None:
        """Save cumulative totals to disk."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with self._lock:
                state = {
                    "total_usd": self._total_usd,
                    "by_provider": dict(self._by_provider),
                    "by_agent": dict(self._by_agent),
                    "call_count": self._total_call_count,
                    "recent_calls": self._calls[-50:],
                }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass  # non-critical — don't crash on persist failure

    def _load(self) -> None:
        """Restore totals from disk on startup."""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self._total_usd = state.get("total_usd", 0.0)
            self._by_provider = state.get("by_provider", {})
            self._by_agent = state.get("by_agent", {})
            self._calls = state.get("recent_calls", [])
            self._total_call_count = state.get("call_count", len(self._calls))
        except Exception:
            pass  # corrupted file — start fresh
