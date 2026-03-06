"""Trading Pipeline — orchestrates the full decision flow.

NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal

Every step is wrapped in try/except — failures log + alert, never crash.
See TRADING_AGENT_PRD.md Section 4 for the communication map.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.chart_analyst import ChartAnalyst
from agents.circuit_breaker_agent import CircuitBreakerAgent
from agents.devils_advocate import DevilsAdvocate
from agents.market_analyst import MarketAnalyst
from agents.news_scout import NewsScout
from agents.trade_journal import TradeJournal
from core.event_bus import event_bus
from core.executor import Executor
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.phantom_tracker import PhantomTracker
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
from core.schemas import (
    ConfirmingSignal,
    ExecutionOrder,
    OrderConfirmation,
    OrderStatus,
    SignalAlert,
    TradeThesis,
    Verdict,
)
from tools.market_data import MarketDataFetcher
from tools.telegram_bot import TelegramNotifier

log = setup_logger("trading.pipeline")


class TradingPipeline:
    """Orchestrates the full signal-to-execution flow."""

    def __init__(
        self,
        portfolio: PortfolioState,
        risk_manager: RiskManager,
        executor: Executor,
        telegram: TelegramNotifier,
        llm_client: LLMClient | None = None,
    ):
        self._portfolio = portfolio
        self._risk = risk_manager
        self._executor = executor
        self._telegram = telegram
        self._llm = llm_client or LLMClient()

        # Create agents with shared LLM client
        self._news_scout = NewsScout(llm_client=self._llm)
        self._analyst = MarketAnalyst(llm_client=self._llm)
        self._chart = ChartAnalyst(llm_client=self._llm)
        self._devil = DevilsAdvocate(llm_client=self._llm)
        self._journal = TradeJournal(llm_client=self._llm)
        self._circuit_breaker = CircuitBreakerAgent(
            llm_client=self._llm,
            executor=self._executor,
            portfolio=self._portfolio,
            telegram=self._telegram,
        )
        self._phantom = PhantomTracker()

        # Load risk params for auto-recovery cooldown + default stop-loss
        self._risk_params: dict[str, Any] = {}
        try:
            params_path = Path(__file__).resolve().parent.parent / "config" / "risk_params.json"
            if params_path.exists():
                self._risk_params = json.loads(params_path.read_text())
        except Exception as e:
            log.warning("Could not load risk_params.json: %s", e)

    def run_news_scan(self) -> list[SignalAlert]:
        """Run a news scan and process any signals found."""
        if self._portfolio.halted:
            log.warning("System halted — skipping news scan")
            return []

        try:
            log.info("Starting news scan...")
            event_bus.emit("pipeline", "news_scan_start", {})
            signals = self._news_scout.scan()
            log.info("News scan produced %d signals", len(signals))
            event_bus.emit("pipeline", "news_scan_complete", {
                "signal_count": len(signals),
                "signals": [{"headline": s.headline, "asset": s.asset, "strength": s.signal_strength} for s in signals],
            })

            if signals:
                self.process_signals(signals)

            return signals
        except Exception as e:
            log.error("News scan failed: %s", e)
            self._telegram.send_alert(f"News scan error: {e}")
            return []

    def process_signals(self, signals: list[SignalAlert]) -> list[dict[str, Any]]:
        """Process a list of signals through the full pipeline.

        The analysis phase (MarketAnalyst + ChartAnalyst + DevilsAdvocate) runs
        in parallel across signals using a thread pool. The execution phase
        (RiskManager + Executor + Portfolio update) runs serially to protect
        thread-unsafe portfolio state and executor.
        """
        # Phase 1: Run analysis in parallel (Analyst → Chart → Devil)
        analysis_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._analyze_signal, signal): signal
                for signal in signals
            }
            for future in as_completed(futures):
                signal = futures[future]
                try:
                    analysis = future.result()
                except Exception as e:
                    log.error("Parallel analysis failed for %s: %s", signal.asset, e)
                    analysis = {
                        "signal": signal,
                        "result": {"signal": signal.headline, "asset": signal.asset, "outcome": "analyst_error", "error": str(e)},
                        "thesis": None,
                        "verdict": None,
                    }
                analysis_results.append(analysis)

        # Phase 2: Execute serially (Risk → Execute → Portfolio)
        results = []
        for analysis in analysis_results:
            result = analysis["result"]
            thesis = analysis.get("thesis")
            verdict = analysis.get("verdict")
            signal = analysis["signal"]

            # If analysis phase already determined final outcome, skip execution
            if result["outcome"] != "pending_execution":
                results.append(result)
                continue

            # Serial execution phase
            result = self._execute_trade(signal, thesis, verdict, result)
            results.append(result)

        return results

    def _analyze_signal(self, signal: SignalAlert) -> dict[str, Any]:
        """Phase 1: Analyst → Chart → Devil (thread-safe, no portfolio mutation)."""
        result: dict[str, Any] = {
            "signal": signal.headline,
            "asset": signal.asset,
            "outcome": "no_trade",
        }

        # Step 1: Market Analyst generates thesis
        try:
            thesis = self._analyst.analyze_signal(signal)
        except Exception as e:
            log.error("Market Analyst failed for %s: %s", signal.asset, e)
            result["outcome"] = "analyst_error"
            result["error"] = str(e)
            return {"signal": signal, "result": result, "thesis": None, "verdict": None}

        if thesis is None:
            log.info("No trade thesis for %s — recording no-trade", signal.asset)
            event_bus.emit("pipeline", "no_thesis", {"asset": signal.asset, "signal_headline": signal.headline})
            try:
                self._journal.record_no_trade(signal, "No thesis generated")
            except Exception as e:
                log.error("Journal record_no_trade failed: %s", e)
            result["outcome"] = "no_trade"
            return {"signal": signal, "result": result, "thesis": None, "verdict": None}

        result["thesis"] = thesis.thesis
        result["confidence"] = thesis.confidence
        event_bus.emit("pipeline", "thesis_generated", {
            "asset": thesis.asset, "direction": thesis.direction.value,
            "confidence": thesis.confidence, "thesis": thesis.thesis[:200],
        })

        # Step 1b: Chart Analyst — independent price action analysis
        thesis = self._enrich_with_chart(thesis)

        # Step 2: Devil's Advocate challenges
        try:
            portfolio_state = self._portfolio.snapshot()
            verdict = self._devil.challenge(thesis, portfolio_state)
        except Exception as e:
            log.error("Devil's Advocate failed for %s: %s", signal.asset, e)
            result["outcome"] = "devil_error"
            result["error"] = str(e)
            return {"signal": signal, "result": result, "thesis": thesis, "verdict": None}

        result["verdict"] = verdict.verdict.value
        event_bus.emit("pipeline", "devil_verdict", {
            "asset": signal.asset, "verdict": verdict.verdict.value,
            "flags_raised": verdict.flags_raised, "reasoning": verdict.final_reasoning[:200],
        })

        if verdict.verdict == Verdict.KILLED:
            log.info("Trade KILLED by Devil's Advocate: %s", verdict.final_reasoning)
            event_bus.emit("pipeline", "trade_killed", {
                "asset": signal.asset, "killed_by": "devils_advocate",
                "reasoning": verdict.final_reasoning[:200],
            })
            try:
                self._journal.record_killed_trade(thesis, verdict)
            except Exception as e:
                log.error("Journal record_killed_trade failed: %s", e)
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="devils_advocate",
                reason=verdict.final_reasoning[:200],
                entry_price=thesis.supporting_data.get("current_price", 0),
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "killed"
            return {"signal": signal, "result": result, "thesis": thesis, "verdict": verdict}

        # Analysis passed — mark for serial execution phase
        result["outcome"] = "pending_execution"
        return {"signal": signal, "result": result, "thesis": thesis, "verdict": verdict}

    def _execute_trade(
        self, signal: SignalAlert, thesis: TradeThesis, verdict: Any, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Phase 2: Risk → Execute → Portfolio (serial, thread-unsafe operations)."""
        # Step 3: Build execution order and run risk checks
        try:
            exec_order = self._build_execution_order(thesis, verdict)
            approved, reason, adjusted = self._risk.validate_order(
                exec_order, self._portfolio.snapshot()
            )
        except Exception as e:
            log.error("Risk validation failed for %s: %s", signal.asset, e)
            result["outcome"] = "risk_error"
            result["error"] = str(e)
            return result

        event_bus.emit("pipeline", "risk_check", {
            "asset": signal.asset, "approved": approved, "reason": reason,
        })

        if not approved:
            log.info("Trade rejected by Risk Manager: %s", reason)
            try:
                self._journal.record_killed_trade(thesis, verdict)
            except Exception as e:
                log.error("Journal record_killed_trade failed: %s", e)
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="risk_manager",
                reason=reason,
                entry_price=thesis.supporting_data.get("current_price", 0),
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "risk_rejected"
            result["risk_reason"] = reason
            return result

        order_to_execute = adjusted or exec_order

        # Step 4: Execute
        try:
            confirmation = self._executor.execute(order_to_execute)
        except Exception as e:
            log.error("Execution failed for %s: %s", signal.asset, e)
            result["outcome"] = "execution_error"
            result["error"] = str(e)
            return result

        if confirmation.get("type") == "order_error":
            log.warning("Order error: %s", confirmation.get("error"))
            result["outcome"] = "execution_error"
            result["error"] = confirmation.get("error", "")
            return result

        # Step 5: Record in journal
        try:
            order_conf = OrderConfirmation(
                order_id=confirmation.get("order_id", 0),
                asset=confirmation.get("asset", thesis.asset),
                direction=thesis.direction,
                quantity=confirmation.get("quantity", 0),
                fill_price=confirmation.get("fill_price", 0.0),
                status=OrderStatus(confirmation.get("status", "Submitted")),
                thesis_id=exec_order.get("thesis_id", ""),
            )
            self._journal.record_entry(thesis, verdict, order_conf)
        except Exception as e:
            log.error("Journal record_entry failed: %s", e)

        # Step 6: Update portfolio
        try:
            self._portfolio.add_position({
                "trade_id": confirmation.get("order_id", ""),
                "asset": thesis.asset,
                "direction": thesis.direction.value,
                "entry_price": confirmation.get("fill_price", 0.0),
                "position_size_pct": thesis.suggested_position_pct,
                "quantity": confirmation.get("quantity", order_to_execute.get("quantity", 0)),
                "stop_loss_price": order_to_execute.get("stop_loss"),
                "take_profit_price": order_to_execute.get("take_profit"),
            })
            self._portfolio.persist()
        except Exception as e:
            log.error("Portfolio update failed: %s", e)

        # Step 7: Telegram notification
        try:
            self._telegram.send_alert(
                f"Trade executed: {thesis.direction.value.upper()} "
                f"{thesis.asset} @ {confirmation.get('fill_price', 0)}"
            )
        except Exception as e:
            log.error("Telegram notification failed: %s", e)

        result["outcome"] = "executed"
        result["fill_price"] = confirmation.get("fill_price", 0.0)
        event_bus.emit("pipeline", "trade_executed", {
            "asset": thesis.asset, "direction": thesis.direction.value,
            "fill_price": confirmation.get("fill_price", 0), "quantity": confirmation.get("quantity", 0),
        })
        log.info(
            "Pipeline complete: %s %s — executed",
            thesis.direction.value,
            thesis.asset,
        )
        return result

    def process_single_signal(self, signal: SignalAlert) -> dict[str, Any]:
        """Process one signal through the full pipeline (sequential fallback).

        Used for single-signal processing. For batch processing, use
        ``process_signals`` which parallelizes the analysis phase.
        """
        analysis = self._analyze_signal(signal)
        result = analysis["result"]
        thesis = analysis.get("thesis")
        verdict = analysis.get("verdict")

        if result["outcome"] != "pending_execution":
            return result

        return self._execute_trade(signal, thesis, verdict, result)

    def check_stop_losses(self) -> list[dict[str, Any]]:
        """Check all open positions against their stop-loss levels.

        Called every 5 minutes from the heartbeat cycle.
        Tier 0: pure Python, no LLM, deterministic.
        """
        if self._portfolio.halted:
            log.info("System halted — skipping stop-loss check")
            return []

        positions = self._portfolio.open_positions[:]
        closed: list[dict[str, Any]] = []
        market_data = MarketDataFetcher()

        for pos in positions:
            stop_price = pos.get("stop_loss_price")
            if stop_price is None:
                continue

            asset = pos.get("asset", "")
            trade_id = pos.get("trade_id", "")
            direction = pos.get("direction", "long")

            # Fetch current price
            try:
                price_data = market_data.get_price(asset)
                current_price = price_data.get("price", 0)
            except Exception as e:
                log.error("Stop-loss check: price fetch failed for %s: %s", asset, e)
                continue

            if current_price <= 0:
                log.warning("Stop-loss check: no valid price for %s, skipping", asset)
                continue

            # Check breach
            breached = False
            if direction == "long" and current_price <= stop_price:
                breached = True
            elif direction == "short" and current_price >= stop_price:
                breached = True

            if not breached:
                continue

            # --- STOP-LOSS BREACHED ---
            log.warning(
                "STOP-LOSS BREACHED: %s %s (trade %s) — price $%.2f hit stop $%.2f",
                direction.upper(), asset, trade_id, current_price, stop_price,
            )

            quantity = pos.get("quantity", 0)
            close_direction = "short" if direction == "long" else "long"
            close_order = {
                "type": "execution_order",
                "thesis_id": f"sl_close_{trade_id}",
                "asset": asset,
                "direction": close_direction,
                "quantity": quantity,
                "order_type": "market",
                "stop_loss": None,
                "position_size_pct": pos.get("position_size_pct", 0),
            }

            try:
                confirmation = self._executor.execute(close_order)
            except Exception as e:
                log.error("Stop-loss close failed for %s: %s", trade_id, e)
                continue

            if confirmation.get("type") == "order_error":
                log.error(
                    "Stop-loss close order error for %s: %s",
                    trade_id, confirmation.get("error", ""),
                )
                continue

            # Remove position and record P&L
            self._portfolio.remove_position(trade_id)

            entry_price = pos.get("entry_price", 0)
            exit_price = confirmation.get("fill_price", current_price)
            if direction == "long":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            self._portfolio.record_trade(pnl)

            event_bus.emit("stop_loss", "triggered", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
            })

            try:
                self._telegram.send_alert(
                    f"STOP-LOSS HIT: {direction.upper()} {asset}\n"
                    f"Entry: ${entry_price:.2f} | Stop: ${stop_price:.2f} | "
                    f"Exit: ${exit_price:.2f}\n"
                    f"P&L: ${pnl:.2f}"
                )
            except Exception as e:
                log.error("Stop-loss Telegram alert failed: %s", e)

            closed.append({
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "pnl": round(pnl, 2),
            })

            log.info("Stop-loss close complete: %s %s — P&L: $%.2f", asset, trade_id, pnl)

        if closed:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Portfolio persist after stop-loss closures failed: %s", e)

        return closed

    def check_take_profits(self) -> list[dict[str, Any]]:
        """Check all open positions against their take-profit levels.

        Called every 5 minutes from the heartbeat cycle.
        Tier 0: pure Python, no LLM, deterministic.
        """
        if self._portfolio.halted:
            log.info("System halted — skipping take-profit check")
            return []

        positions = self._portfolio.open_positions[:]
        closed: list[dict[str, Any]] = []
        market_data = MarketDataFetcher()

        for pos in positions:
            tp_price = pos.get("take_profit_price")
            if tp_price is None:
                continue

            asset = pos.get("asset", "")
            trade_id = pos.get("trade_id", "")
            direction = pos.get("direction", "long")

            # Fetch current price
            try:
                price_data = market_data.get_price(asset)
                current_price = price_data.get("price", 0)
            except Exception as e:
                log.error("Take-profit check: price fetch failed for %s: %s", asset, e)
                continue

            if current_price <= 0:
                log.warning("Take-profit check: no valid price for %s, skipping", asset)
                continue

            # Check breach (inverted from stop-loss)
            breached = False
            if direction == "long" and current_price >= tp_price:
                breached = True
            elif direction == "short" and current_price <= tp_price:
                breached = True

            if not breached:
                continue

            # --- TAKE-PROFIT HIT ---
            log.info(
                "TAKE-PROFIT HIT: %s %s (trade %s) — price $%.2f hit target $%.2f",
                direction.upper(), asset, trade_id, current_price, tp_price,
            )

            quantity = pos.get("quantity", 0)
            close_direction = "short" if direction == "long" else "long"
            close_order = {
                "type": "execution_order",
                "thesis_id": f"tp_close_{trade_id}",
                "asset": asset,
                "direction": close_direction,
                "quantity": quantity,
                "order_type": "market",
                "stop_loss": None,
                "position_size_pct": pos.get("position_size_pct", 0),
            }

            try:
                confirmation = self._executor.execute(close_order)
            except Exception as e:
                log.error("Take-profit close failed for %s: %s", trade_id, e)
                continue

            if confirmation.get("type") == "order_error":
                log.error(
                    "Take-profit close order error for %s: %s",
                    trade_id, confirmation.get("error", ""),
                )
                continue

            # Remove position and record P&L
            self._portfolio.remove_position(trade_id)

            entry_price = pos.get("entry_price", 0)
            exit_price = confirmation.get("fill_price", current_price)
            if direction == "long":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            self._portfolio.record_trade(pnl)

            event_bus.emit("take_profit", "triggered", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "target_price": tp_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
            })

            try:
                self._telegram.send_alert(
                    f"TAKE-PROFIT HIT: {direction.upper()} {asset}\n"
                    f"Entry: ${entry_price:.2f} | Target: ${tp_price:.2f} | "
                    f"Exit: ${exit_price:.2f}\n"
                    f"P&L: ${pnl:.2f}"
                )
            except Exception as e:
                log.error("Take-profit Telegram alert failed: %s", e)

            closed.append({
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "target_price": tp_price,
                "pnl": round(pnl, 2),
            })

            log.info("Take-profit close complete: %s %s — P&L: $%.2f", asset, trade_id, pnl)

        if closed:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Portfolio persist after take-profit closures failed: %s", e)

        return closed

    def sync_portfolio_with_broker(self) -> None:
        """Log broker account status for monitoring (does NOT override internal equity).

        Internal equity tracks from $100 allocated capital + trade P&L.
        Broker balance is logged for health monitoring only.
        """
        from core.alpaca_executor import AlpacaExecutor
        if not isinstance(self._executor, AlpacaExecutor):
            return
        try:
            account = self._executor.get_account_info()
            if not account:
                return
            broker_equity = float(account.get("portfolio_value", 0))
            broker_cash = float(account.get("cash", 0))
            log.info(
                "Broker health: equity=$%.2f, cash=$%.2f | Internal: equity=$%.2f",
                broker_equity, broker_cash, self._portfolio.equity,
            )
        except Exception as e:
            log.error("Broker health check failed: %s", e)

    def recalculate_equity(self) -> None:
        """Recalculate portfolio equity using live market prices."""
        if self._portfolio.halted:
            return
        try:
            self.sync_portfolio_with_broker()
        except Exception:
            pass
        try:
            mdf = MarketDataFetcher()
            self._portfolio.calculate_equity(mdf)
        except Exception as e:
            log.error("Equity recalculation failed: %s", e)

    def run_circuit_breaker_check(self, market_data: dict[str, Any] | None = None) -> None:
        """Run circuit breaker check against current portfolio state."""
        if self._portfolio.halted:
            self._try_auto_recovery()
            return

        try:
            portfolio_state = self._portfolio.snapshot()
            if market_data is None:
                market_data = {}

            result = self._circuit_breaker.check(portfolio_state, market_data)
            if result:
                log.warning("Circuit breaker triggered: %s", result["triggers_fired"])
                event_bus.emit("circuit_breaker", "triggered", {"triggers_fired": result["triggers_fired"]})
                decision = self._circuit_breaker.escalate_to_opus(
                    result["triggers_fired"], portfolio_state, market_data
                )
                event_bus.emit("circuit_breaker", "decision", {
                    "decision": getattr(decision, "decision", str(decision)),
                    "reasoning": getattr(decision, "reasoning", ""),
                })
                self._circuit_breaker.execute_decision(decision)
        except Exception as e:
            log.error("Circuit breaker check failed: %s", e)

    _AUTO_RECOVERY_COOLDOWN_HOURS = 6

    def _try_auto_recovery(self) -> None:
        """Auto-recover from circuit breaker halt after cooldown period."""
        try:
            cooldown_hours = self._risk_params.get(
                "auto_recovery_cooldown_hours", self._AUTO_RECOVERY_COOLDOWN_HOURS
            )
            last_updated = self._portfolio.last_updated
            if not last_updated:
                return

            halted_at = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if halted_at.tzinfo is None:
                halted_at = halted_at.replace(tzinfo=timezone.utc)

            hours_since = (now - halted_at).total_seconds() / 3600

            if hours_since < cooldown_hours:
                log.info(
                    "System halted — auto-recovery in %.1f hours (%.1f/%d hours elapsed)",
                    cooldown_hours - hours_since, hours_since, cooldown_hours,
                )
                return

            # Reset: unhalt, reset peak to current equity, clear drawdown
            with self._portfolio._lock:
                self._portfolio.halted = False
                self._portfolio.peak_equity = self._portfolio.equity
                self._portfolio.drawdown_from_peak_pct = 0.0
            self._portfolio.persist()

            log.warning(
                "AUTO-RECOVERY: System unhalted after %d-hour cooldown. "
                "Peak equity reset to $%.2f",
                cooldown_hours, self._portfolio.equity,
            )

            event_bus.emit("circuit_breaker", "auto_recovery", {
                "cooldown_hours": cooldown_hours,
                "equity": self._portfolio.equity,
            })

            try:
                self._telegram.send_alert(
                    f"AUTO-RECOVERY: System unhalted after {cooldown_hours}h cooldown.\n"
                    f"Peak equity reset to ${self._portfolio.equity:.2f}\n"
                    f"Trading will resume on next scan."
                )
            except Exception:
                pass

        except Exception as e:
            log.error("Auto-recovery check failed: %s", e)

    def run_scheduled_analysis(self, session: str) -> list[dict[str, Any]]:
        """Run scheduled analysis for all assets and process results."""
        if self._portfolio.halted:
            log.warning("System halted — skipping scheduled analysis")
            return []

        try:
            log.info("Running scheduled analysis: %s", session)
            event_bus.emit("pipeline", "scheduled_analysis_start", {"session": session})
            theses = self._analyst.scheduled_analysis(session)
            log.info("Scheduled analysis produced %d theses", len(theses))

            results = []
            for thesis in theses:
                result = self._process_thesis(thesis)
                results.append(result)

            event_bus.emit("pipeline", "scheduled_analysis_complete", {
                "session": session, "thesis_count": len(theses),
            })
            return results
        except Exception as e:
            log.error("Scheduled analysis failed: %s", e)
            self._telegram.send_alert(f"Scheduled analysis error: {e}")
            return []

    def _process_thesis(self, thesis: TradeThesis) -> dict[str, Any]:
        """Process a thesis through Chart → Devil → Risk → Execute → Journal."""
        result: dict[str, Any] = {
            "asset": thesis.asset,
            "thesis": thesis.thesis,
            "confidence": thesis.confidence,
            "outcome": "pending",
        }

        # Chart Analyst — enrich thesis with price action signal
        thesis = self._enrich_with_chart(thesis)

        # Devil's Advocate
        try:
            portfolio_state = self._portfolio.snapshot()
            verdict = self._devil.challenge(thesis, portfolio_state)
        except Exception as e:
            log.error("Devil's Advocate failed: %s", e)
            result["outcome"] = "devil_error"
            return result

        result["verdict"] = verdict.verdict.value

        if verdict.verdict == Verdict.KILLED:
            try:
                self._journal.record_killed_trade(thesis, verdict)
            except Exception as e:
                log.error("Journal failed: %s", e)
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="devils_advocate",
                reason=verdict.final_reasoning[:200],
                entry_price=thesis.supporting_data.get("current_price", 0),
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "killed"
            return result

        # Risk check
        try:
            exec_order = self._build_execution_order(thesis, verdict)
            approved, reason, adjusted = self._risk.validate_order(
                exec_order, self._portfolio.snapshot()
            )
        except Exception as e:
            log.error("Risk validation failed: %s", e)
            result["outcome"] = "risk_error"
            return result

        if not approved:
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="risk_manager",
                reason=reason,
                entry_price=thesis.supporting_data.get("current_price", 0),
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "risk_rejected"
            result["risk_reason"] = reason
            return result

        order_to_execute = adjusted or exec_order

        # Execute
        try:
            confirmation = self._executor.execute(order_to_execute)
        except Exception as e:
            log.error("Execution failed: %s", e)
            result["outcome"] = "execution_error"
            return result

        if confirmation.get("type") == "order_error":
            result["outcome"] = "execution_error"
            result["error"] = confirmation.get("error", "")
            return result

        # Journal
        try:
            order_conf = OrderConfirmation(
                order_id=confirmation.get("order_id", 0),
                asset=confirmation.get("asset", thesis.asset),
                direction=thesis.direction,
                quantity=confirmation.get("quantity", 0),
                fill_price=confirmation.get("fill_price", 0.0),
                status=OrderStatus(confirmation.get("status", "Submitted")),
            )
            self._journal.record_entry(thesis, verdict, order_conf)
        except Exception as e:
            log.error("Journal failed: %s", e)

        # Portfolio update
        try:
            self._portfolio.add_position({
                "trade_id": confirmation.get("order_id", ""),
                "asset": thesis.asset,
                "direction": thesis.direction.value,
                "entry_price": confirmation.get("fill_price", 0.0),
                "position_size_pct": thesis.suggested_position_pct,
                "quantity": confirmation.get("quantity", order_to_execute.get("quantity", 0)),
                "stop_loss_price": order_to_execute.get("stop_loss"),
                "take_profit_price": order_to_execute.get("take_profit"),
            })
            self._portfolio.persist()
        except Exception as e:
            log.error("Portfolio update failed: %s", e)

        result["outcome"] = "executed"
        result["fill_price"] = confirmation.get("fill_price", 0.0)
        return result

    def _enrich_with_chart(self, thesis: TradeThesis) -> TradeThesis:
        """Run chart analysis and merge result into thesis confirming signals."""
        try:
            chart = self._chart.analyze(thesis.asset)
            if chart.get("pattern_found"):
                # Chart agrees with thesis direction?
                chart_dir = chart.get("direction", "neutral")
                thesis_dir = thesis.direction.value

                agrees = (
                    chart_dir == thesis_dir
                    or chart_dir == "neutral"
                )

                thesis.confirming_signals.chart_pattern = ConfirmingSignal(
                    present=agrees and chart.get("confidence", 0) >= 0.3,
                    description=f"{chart.get('pattern_name', 'pattern')}: {chart.get('description', '')}",
                )

                # Boost confidence if chart strongly agrees
                if agrees and chart.get("confidence", 0) >= 0.5:
                    boost = min(0.1, chart["confidence"] * 0.15)
                    thesis.confidence = min(1.0, thesis.confidence + boost)
                    log.info(
                        "Chart confirms %s %s — confidence boosted by %.2f to %.2f",
                        thesis_dir, thesis.asset, boost, thesis.confidence,
                    )

                # Store chart data in supporting_data
                thesis.supporting_data["chart_analysis"] = {
                    "pattern": chart.get("pattern_name"),
                    "direction": chart_dir,
                    "confidence": chart.get("confidence"),
                    "trend": chart.get("trend"),
                    "support": chart.get("support_levels", []),
                    "resistance": chart.get("resistance_levels", []),
                }

                event_bus.emit("pipeline", "chart_analysis", {
                    "asset": thesis.asset,
                    "pattern": chart.get("pattern_name"),
                    "direction": chart_dir,
                    "confidence": chart.get("confidence"),
                    "agrees_with_thesis": agrees,
                })
            else:
                thesis.confirming_signals.chart_pattern = ConfirmingSignal(
                    present=False,
                    description="No actionable chart pattern found",
                )
        except Exception as e:
            log.warning("Chart analysis skipped for %s: %s", thesis.asset, e)
            thesis.confirming_signals.chart_pattern = ConfirmingSignal(
                present=False, description=f"Chart analysis error: {e}"
            )
        return thesis

    @staticmethod
    def _calculate_ev(
        confidence: float, risk_reward_ratio: float
    ) -> dict[str, float]:
        """Calculate Expected Value for a trade.

        EV = P(win) × R(win) - P(loss) × R(loss)
        Where R(win) = risk_reward_ratio, R(loss) = 1.0 (risk unit).
        Returns EV and Kelly-optimal fraction.
        """
        p_win = max(0.01, min(0.99, confidence))
        p_loss = 1.0 - p_win
        rr = max(0.1, risk_reward_ratio)

        ev = p_win * rr - p_loss * 1.0
        # Kelly fraction: f* = (p * b - q) / b  where b=rr, p=p_win, q=p_loss
        kelly = max(0.0, (p_win * rr - p_loss) / rr)

        return {
            "ev": round(ev, 4),
            "kelly_fraction": round(kelly, 4),
            "positive_ev": ev > 0,
        }

    def _build_execution_order(
        self, thesis: TradeThesis, verdict: Any
    ) -> dict[str, Any]:
        """Build an execution order dict from thesis + verdict."""
        # Use confidence-adjusted position size
        position_pct = thesis.suggested_position_pct
        if hasattr(verdict, "confidence_adjusted") and verdict.confidence_adjusted > 0:
            # Scale position by confidence adjustment ratio
            ratio = verdict.confidence_adjusted / max(thesis.confidence, 0.01)
            position_pct = min(position_pct * ratio, 7.0)

        # EV-based sizing: scale position by Kelly fraction
        try:
            rr_str = thesis.risk_reward_ratio or "1.5"
            rr = float(str(rr_str).split(":")[0]) if ":" in str(rr_str) else float(rr_str)
        except (ValueError, TypeError):
            rr = 1.5

        confidence = verdict.confidence_adjusted if hasattr(verdict, "confidence_adjusted") and verdict.confidence_adjusted > 0 else thesis.confidence
        ev_result = self._calculate_ev(confidence, rr)

        if ev_result["positive_ev"]:
            # Scale position by Kelly fraction (capped at suggested_position_pct)
            kelly_pct = ev_result["kelly_fraction"] * 100
            position_pct = min(position_pct, kelly_pct, 7.0)
            position_pct = max(position_pct, 2.0)  # Minimum 2% micro position
            log.info("EV: %.4f | Kelly: %.1f%% | Position: %.1f%%", ev_result["ev"], kelly_pct, position_pct)
        else:
            # Negative EV — still take micro position during paper trading for data
            position_pct = 2.0
            log.info("Negative EV (%.4f) — micro position 2%% for data", ev_result["ev"])

        # Calculate stop-loss from invalidation level, or default to 5% from price
        stop_loss = None
        if thesis.invalidation_level:
            try:
                stop_loss = float(thesis.invalidation_level)
            except (ValueError, TypeError):
                pass

        # Calculate approximate quantity
        equity = self._portfolio.equity
        current_price = thesis.supporting_data.get("current_price", 0)

        # Fetch live price if not in supporting_data
        if not current_price:
            try:
                mdf = MarketDataFetcher()
                price_data = mdf.get_price(thesis.asset)
                current_price = price_data.get("price", 0)
            except Exception:
                pass

        # Default stop-loss if none provided: use config or 3% adverse move
        if stop_loss is None and current_price > 0:
            sl_pct = self._risk_params.get("default_stop_loss_pct", 3.0) / 100.0
            if thesis.direction.value == "long":
                stop_loss = round(current_price * (1 - sl_pct), 2)
            else:
                stop_loss = round(current_price * (1 + sl_pct), 2)
            log.info("Default stop-loss set at $%.2f (%.0f%% from $%.2f)", stop_loss, sl_pct * 100, current_price)
        # Calculate take-profit from risk/reward ratio + stop-loss distance
        take_profit = None
        if stop_loss and current_price > 0:
            risk_distance = abs(current_price - stop_loss)
            reward_distance = risk_distance * rr
            if thesis.direction.value == "long":
                take_profit = round(current_price + reward_distance, 2)
            else:
                take_profit = round(current_price - reward_distance, 2)
            log.info("Take-profit set at $%.2f (%.1f:1 R:R from $%.2f)", take_profit, rr, current_price)

        if current_price > 0 and equity > 0:
            position_value = equity * (position_pct / 100.0)
            quantity = position_value / current_price
        else:
            quantity = 0.0

        return {
            "type": "execution_order",
            "thesis_id": str(id(thesis)),
            "asset": thesis.asset,
            "direction": thesis.direction.value,
            "quantity": quantity,
            "order_type": "market",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_pct": position_pct,
        }
