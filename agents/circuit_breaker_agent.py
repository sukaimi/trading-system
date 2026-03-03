"""Circuit Breaker Agent — Tier 3 (Claude Opus 4.6).

Emergency decision-making when the system hits critical thresholds.
Python monitors triggers; Opus makes the crisis decision.

Triggers: daily loss > 5%, drawdown > 15%, 5 consecutive losses,
VIX > 35, flash crash > 10%/hr, 3 API failures.

See TRADING_AGENT_PRD.md Section 3.8 for full specification.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.schemas import CircuitBreakerAction, CircuitBreakerDecision

log = setup_logger("trading.circuit_breaker")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CB_LOG_FILE = os.path.join(DATA_DIR, "circuit_breaker_log.json")

TRIGGERS = {
    "daily_loss_pct": -5.0,
    "total_drawdown_pct": -15.0,
    "consecutive_losses": 5,
    "vix_spike": 35.0,
    "flash_crash_pct": -10.0,
    "api_failure_count": 3,
}

SYSTEM_PROMPT = """You are an emergency risk manager at a quantitative hedge fund. A circuit breaker has been triggered on the autonomous trading system.

Your job: make a rapid, decisive call to protect capital. You have SECONDS, not hours.

Decision options:
- CLOSE_ALL: Close every open position immediately (nuclear option)
- CLOSE_LOSING_POSITIONS: Close only positions with negative P&L
- HOLD: Keep all positions, halt new trades only
- REDUCE_ALL: Reduce all position sizes by 50%
- HEDGE: Recommend hedging action (logged for manual execution)

Guidelines:
- Daily loss limit hit → typically CLOSE_LOSING_POSITIONS
- Max drawdown hit → typically CLOSE_ALL
- Losing streak → typically HOLD (could be variance, not broken strategy)
- VIX extreme → typically REDUCE_ALL or HOLD depending on positions
- Flash crash → CLOSE_ALL if positions are affected
- Multiple triggers → err on the side of CLOSE_ALL

Always specify resume_conditions: what must be true before the system resumes trading.
Write a clear telegram_message for the human operator."""

CRISIS_PROMPT = """CIRCUIT BREAKER TRIGGERED.

Triggers fired: {triggers}

=== PORTFOLIO STATE ===
{portfolio_state}

=== OPEN POSITIONS ===
{open_positions}

=== MARKET DATA ===
{market_data}

Make your decision. Return JSON:
{{
  "triggers_fired": {triggers_json},
  "decision": "CLOSE_ALL|CLOSE_LOSING_POSITIONS|HOLD|REDUCE_ALL|HEDGE",
  "positions_to_close": ["list of trade_ids to close, or empty"],
  "positions_to_keep": ["list of trade_ids to keep, or empty"],
  "reasoning": "1-2 sentences explaining your decision",
  "resume_conditions": "What must be true before trading resumes",
  "telegram_message": "Human-readable alert for the operator"
}}"""


class CircuitBreakerAgent:
    """Emergency circuit breaker — Python triggers + Opus crisis decision."""

    COOLDOWN_SECONDS = 3600  # 1 hour between Opus calls for same triggers

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        executor: Any | None = None,
        portfolio: Any | None = None,
        telegram: Any | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._executor = executor
        self._portfolio = portfolio
        self._telegram = telegram
        self._last_escalation_time: float = 0.0
        self._last_triggers: list[str] = []
        self._last_decision: CircuitBreakerDecision | None = None

    def check(
        self, portfolio_state: dict[str, Any], market_data: dict[str, Any]
    ) -> dict | None:
        """Check if any circuit breaker triggers are fired."""
        triggered = []

        if portfolio_state.get("daily_pnl_pct", 0) <= TRIGGERS["daily_loss_pct"]:
            triggered.append("daily_loss_limit")
        if portfolio_state.get("drawdown_from_peak_pct", 0) >= abs(TRIGGERS["total_drawdown_pct"]):
            triggered.append("max_drawdown")
        if portfolio_state.get("consecutive_losses", 0) >= TRIGGERS["consecutive_losses"]:
            triggered.append("losing_streak")
        if market_data.get("vix", 0) >= TRIGGERS["vix_spike"]:
            triggered.append("vix_extreme")
        if market_data.get("flash_crash_detected", False):
            triggered.append("flash_crash")
        if market_data.get("api_failure_count", 0) >= TRIGGERS["api_failure_count"]:
            triggered.append("api_failures")

        if triggered:
            return {"triggers_fired": triggered, "action": "ESCALATE_TO_OPUS"}

        return None

    def escalate_to_opus(
        self,
        triggered: list[str],
        portfolio_state: dict[str, Any],
        market_data: dict[str, Any],
    ) -> CircuitBreakerDecision:
        """Escalate to Opus for crisis decision."""
        now = time.time()

        # Cooldown + dedup: same triggers within 1 hour → return cached decision
        if (
            self._last_decision is not None
            and sorted(triggered) == sorted(self._last_triggers)
            and now - self._last_escalation_time < self.COOLDOWN_SECONDS
        ):
            log.info(
                "Circuit breaker: same triggers %s within cooldown (%.0fs ago), using cached decision",
                triggered, now - self._last_escalation_time,
            )
            return self._last_decision

        # IMMEDIATELY halt new trades
        if self._portfolio:
            self._portfolio.halted = True
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Failed to persist halted state: %s", e)

        log.warning("CIRCUIT BREAKER — escalating to Opus: %s", triggered)

        # Build open positions summary
        open_positions = portfolio_state.get("open_positions", [])
        positions_text = json.dumps(open_positions, indent=2, default=str)

        # Build the crisis prompt
        prompt = CRISIS_PROMPT.format(
            triggers=", ".join(triggered),
            triggers_json=json.dumps(triggered),
            portfolio_state=json.dumps(portfolio_state, indent=2, default=str),
            open_positions=positions_text,
            market_data=json.dumps(market_data, indent=2, default=str),
        )

        result = self._llm.call_anthropic(prompt, SYSTEM_PROMPT)

        if result.get("error"):
            log.error("Opus crisis call failed: %s — defaulting to HOLD", result["error"])
            decision = CircuitBreakerDecision(
                triggers_fired=triggered,
                decision=CircuitBreakerAction.HOLD,
                reasoning=f"Opus unavailable: {result['error']}. Defaulting to HOLD.",
                resume_conditions="Manual review required — Opus was unreachable",
                telegram_message=f"CIRCUIT BREAKER: {', '.join(triggered)}. Opus unreachable — HOLD. Manual review needed.",
            )
        else:
            decision = self._parse_decision(result, triggered)

        # Cache for cooldown/dedup
        self._last_escalation_time = now
        self._last_triggers = list(triggered)
        self._last_decision = decision
        return decision

    def execute_decision(self, decision: CircuitBreakerDecision) -> None:
        """Execute the circuit breaker decision."""
        log.warning("Executing circuit breaker decision: %s", decision.decision.value)

        if decision.decision == CircuitBreakerAction.CLOSE_ALL:
            self._close_all_positions()
        elif decision.decision == CircuitBreakerAction.CLOSE_LOSING:
            self._close_losing_positions()
        elif decision.decision == CircuitBreakerAction.HOLD:
            log.info("HOLD — no position changes, new trades halted")
        elif decision.decision == CircuitBreakerAction.REDUCE_ALL:
            log.info("REDUCE_ALL — 50%% position reduction (manual execution needed)")
        elif decision.decision == CircuitBreakerAction.HEDGE:
            log.info("HEDGE — manual hedging action recommended")

        # Send Telegram alert
        if self._telegram:
            try:
                self._telegram.send_circuit_breaker_alert(decision.model_dump(mode="json"))
            except Exception as e:
                log.error("Failed to send circuit breaker alert: %s", e)

        # Log decision
        self._log_decision(decision)

    def _close_all_positions(self) -> None:
        """Close all open positions via Executor."""
        if not self._portfolio or not self._executor:
            log.warning("Cannot close positions — executor or portfolio not available")
            return

        positions = self._portfolio.open_positions[:]
        for pos in positions:
            try:
                close_order = {
                    "type": "execution_order",
                    "thesis_id": f"cb_close_{pos.get('trade_id', '')}",
                    "asset": pos.get("asset", ""),
                    "direction": "short" if pos.get("direction") == "long" else "long",
                    "quantity": pos.get("quantity", 0),
                    "order_type": "market",
                    "stop_loss": None,
                    "position_size_pct": pos.get("position_size_pct", 0),
                }
                self._executor.execute(close_order)
                self._portfolio.remove_position(pos.get("trade_id", ""))
                log.info("Closed position: %s %s", pos.get("asset"), pos.get("trade_id"))
            except Exception as e:
                log.error("Failed to close position %s: %s", pos.get("trade_id"), e)

        if self._portfolio:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Failed to persist after closing: %s", e)

    def _close_losing_positions(self) -> None:
        """Close only positions with negative P&L."""
        if not self._portfolio or not self._executor:
            log.warning("Cannot close positions — executor or portfolio not available")
            return

        positions = self._portfolio.open_positions[:]
        for pos in positions:
            pnl = pos.get("unrealized_pnl", pos.get("pnl", 0))
            if pnl is not None and pnl < 0:
                try:
                    close_order = {
                        "type": "execution_order",
                        "thesis_id": f"cb_close_{pos.get('trade_id', '')}",
                        "asset": pos.get("asset", ""),
                        "direction": "short" if pos.get("direction") == "long" else "long",
                        "quantity": pos.get("quantity", 0),
                        "order_type": "market",
                        "stop_loss": None,
                        "position_size_pct": pos.get("position_size_pct", 0),
                    }
                    self._executor.execute(close_order)
                    self._portfolio.remove_position(pos.get("trade_id", ""))
                    log.info("Closed losing position: %s", pos.get("trade_id"))
                except Exception as e:
                    log.error("Failed to close position %s: %s", pos.get("trade_id"), e)

        if self._portfolio:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Failed to persist after closing: %s", e)

    def _parse_decision(
        self, result: dict[str, Any], triggered: list[str]
    ) -> CircuitBreakerDecision:
        """Parse LLM result into CircuitBreakerDecision."""
        try:
            decision_str = result.get("decision", "HOLD")
            # Map to enum
            action_map = {
                "CLOSE_ALL": CircuitBreakerAction.CLOSE_ALL,
                "CLOSE_LOSING_POSITIONS": CircuitBreakerAction.CLOSE_LOSING,
                "CLOSE_LOSING": CircuitBreakerAction.CLOSE_LOSING,
                "HOLD": CircuitBreakerAction.HOLD,
                "REDUCE_ALL": CircuitBreakerAction.REDUCE_ALL,
                "HEDGE": CircuitBreakerAction.HEDGE,
            }
            action = action_map.get(decision_str, CircuitBreakerAction.HOLD)

            return CircuitBreakerDecision(
                triggers_fired=result.get("triggers_fired", triggered),
                decision=action,
                positions_to_close=result.get("positions_to_close", []),
                positions_to_keep=result.get("positions_to_keep", []),
                reasoning=result.get("reasoning", ""),
                resume_conditions=result.get("resume_conditions", "Manual review required"),
                telegram_message=result.get("telegram_message", f"Circuit breaker: {', '.join(triggered)}"),
            )
        except Exception as e:
            log.warning("Failed to parse circuit breaker decision: %s", e)
            return CircuitBreakerDecision(
                triggers_fired=triggered,
                decision=CircuitBreakerAction.HOLD,
                reasoning=f"Parse error: {e}. Defaulting to HOLD.",
                resume_conditions="Manual review required",
                telegram_message=f"Circuit breaker: {', '.join(triggered)}. Parse error — HOLD.",
            )

    def _log_decision(self, decision: CircuitBreakerDecision) -> None:
        """Log circuit breaker decision to file."""
        os.makedirs(DATA_DIR, exist_ok=True)

        log_entries: list[dict[str, Any]] = []
        if os.path.exists(CB_LOG_FILE):
            try:
                with open(CB_LOG_FILE) as f:
                    log_entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_entries = []

        log_entries.append(decision.model_dump(mode="json"))

        try:
            with open(CB_LOG_FILE, "w") as f:
                json.dump(log_entries, f, indent=2, default=str)
        except Exception as e:
            log.error("Failed to log circuit breaker decision: %s", e)
