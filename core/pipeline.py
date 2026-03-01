"""Trading Pipeline — orchestrates the full decision flow.

NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal

Every step is wrapped in try/except — failures log + alert, never crash.
See TRADING_AGENT_PRD.md Section 4 for the communication map.
"""

from __future__ import annotations

from typing import Any

from agents.devils_advocate import DevilsAdvocate
from agents.market_analyst import MarketAnalyst
from agents.news_scout import NewsScout
from agents.trade_journal import TradeJournal
from core.executor import Executor
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
from core.schemas import (
    Asset,
    ExecutionOrder,
    OrderConfirmation,
    OrderStatus,
    SignalAlert,
    TradeThesis,
    Verdict,
)
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
        self._devil = DevilsAdvocate(llm_client=self._llm)
        self._journal = TradeJournal(llm_client=self._llm)

    def run_news_scan(self) -> list[SignalAlert]:
        """Run a news scan and process any signals found."""
        try:
            log.info("Starting news scan...")
            signals = self._news_scout.scan()
            log.info("News scan produced %d signals", len(signals))

            if signals:
                self.process_signals(signals)

            return signals
        except Exception as e:
            log.error("News scan failed: %s", e)
            self._telegram.send_alert(f"News scan error: {e}")
            return []

    def process_signals(self, signals: list[SignalAlert]) -> list[dict[str, Any]]:
        """Process a list of signals through the full pipeline."""
        results = []
        for signal in signals:
            result = self.process_single_signal(signal)
            results.append(result)
        return results

    def process_single_signal(self, signal: SignalAlert) -> dict[str, Any]:
        """Process one signal through: Analyst → Devil → Risk → Execute → Journal."""
        result: dict[str, Any] = {
            "signal": signal.headline,
            "asset": signal.asset.value,
            "outcome": "no_trade",
        }

        # Step 1: Market Analyst generates thesis
        try:
            thesis = self._analyst.analyze_signal(signal)
        except Exception as e:
            log.error("Market Analyst failed for %s: %s", signal.asset.value, e)
            result["outcome"] = "analyst_error"
            result["error"] = str(e)
            return result

        if thesis is None:
            log.info("No trade thesis for %s — recording no-trade", signal.asset.value)
            try:
                self._journal.record_no_trade(signal, "No thesis generated")
            except Exception as e:
                log.error("Journal record_no_trade failed: %s", e)
            result["outcome"] = "no_trade"
            return result

        result["thesis"] = thesis.thesis
        result["confidence"] = thesis.confidence

        # Step 2: Devil's Advocate challenges
        try:
            portfolio_state = self._portfolio.snapshot()
            verdict = self._devil.challenge(thesis, portfolio_state)
        except Exception as e:
            log.error("Devil's Advocate failed for %s: %s", signal.asset.value, e)
            result["outcome"] = "devil_error"
            result["error"] = str(e)
            return result

        result["verdict"] = verdict.verdict.value

        if verdict.verdict == Verdict.KILLED:
            log.info("Trade KILLED by Devil's Advocate: %s", verdict.final_reasoning)
            try:
                self._journal.record_killed_trade(thesis, verdict)
            except Exception as e:
                log.error("Journal record_killed_trade failed: %s", e)
            result["outcome"] = "killed"
            return result

        # Step 3: Build execution order and run risk checks
        try:
            exec_order = self._build_execution_order(thesis, verdict)
            approved, reason, adjusted = self._risk.validate_order(
                exec_order, self._portfolio.snapshot()
            )
        except Exception as e:
            log.error("Risk validation failed for %s: %s", signal.asset.value, e)
            result["outcome"] = "risk_error"
            result["error"] = str(e)
            return result

        if not approved:
            log.info("Trade rejected by Risk Manager: %s", reason)
            try:
                self._journal.record_killed_trade(thesis, verdict)
            except Exception as e:
                log.error("Journal record_killed_trade failed: %s", e)
            result["outcome"] = "risk_rejected"
            result["risk_reason"] = reason
            return result

        order_to_execute = adjusted or exec_order

        # Step 4: Execute
        try:
            confirmation = self._executor.execute(order_to_execute)
        except Exception as e:
            log.error("Execution failed for %s: %s", signal.asset.value, e)
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
                asset=Asset(confirmation.get("asset", thesis.asset.value)),
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
                "asset": thesis.asset.value,
                "direction": thesis.direction.value,
                "entry_price": confirmation.get("fill_price", 0.0),
                "position_size_pct": thesis.suggested_position_pct,
            })
            self._portfolio.persist()
        except Exception as e:
            log.error("Portfolio update failed: %s", e)

        # Step 7: Telegram notification
        try:
            self._telegram.send_alert(
                f"Trade executed: {thesis.direction.value.upper()} "
                f"{thesis.asset.value} @ {confirmation.get('fill_price', 0)}"
            )
        except Exception as e:
            log.error("Telegram notification failed: %s", e)

        result["outcome"] = "executed"
        result["fill_price"] = confirmation.get("fill_price", 0.0)
        log.info(
            "Pipeline complete: %s %s — executed",
            thesis.direction.value,
            thesis.asset.value,
        )
        return result

    def run_scheduled_analysis(self, session: str) -> list[dict[str, Any]]:
        """Run scheduled analysis for all assets and process results."""
        try:
            log.info("Running scheduled analysis: %s", session)
            theses = self._analyst.scheduled_analysis(session)
            log.info("Scheduled analysis produced %d theses", len(theses))

            results = []
            for thesis in theses:
                result = self._process_thesis(thesis)
                results.append(result)
            return results
        except Exception as e:
            log.error("Scheduled analysis failed: %s", e)
            self._telegram.send_alert(f"Scheduled analysis error: {e}")
            return []

    def _process_thesis(self, thesis: TradeThesis) -> dict[str, Any]:
        """Process a thesis through Devil → Risk → Execute → Journal."""
        result: dict[str, Any] = {
            "asset": thesis.asset.value,
            "thesis": thesis.thesis,
            "confidence": thesis.confidence,
            "outcome": "pending",
        }

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
                asset=Asset(confirmation.get("asset", thesis.asset.value)),
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
                "asset": thesis.asset.value,
                "direction": thesis.direction.value,
                "entry_price": confirmation.get("fill_price", 0.0),
                "position_size_pct": thesis.suggested_position_pct,
            })
            self._portfolio.persist()
        except Exception as e:
            log.error("Portfolio update failed: %s", e)

        result["outcome"] = "executed"
        result["fill_price"] = confirmation.get("fill_price", 0.0)
        return result

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

        # Calculate stop-loss from invalidation or ATR
        stop_loss = None
        if thesis.invalidation_level:
            try:
                stop_loss = float(thesis.invalidation_level)
            except (ValueError, TypeError):
                pass

        # Calculate approximate quantity
        equity = self._portfolio.equity
        current_price = thesis.supporting_data.get("current_price", 0)
        if current_price > 0 and equity > 0:
            position_value = equity * (position_pct / 100.0)
            quantity = position_value / current_price
        else:
            quantity = 0.0

        return {
            "type": "execution_order",
            "thesis_id": str(id(thesis)),
            "asset": thesis.asset.value,
            "direction": thesis.direction.value,
            "quantity": quantity,
            "order_type": "market",
            "stop_loss": stop_loss,
            "take_profit": None,
            "position_size_pct": position_pct,
        }
