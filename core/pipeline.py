"""Trading Pipeline — orchestrates the full decision flow.

NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal

Every step is wrapped in try/except — failures log + alert, never crash.
See TRADING_AGENT_PRD.md Section 4 for the communication map.
"""

from __future__ import annotations

import json
import uuid
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
from core.signal_tracker import SignalAccuracyTracker
from core.portfolio import PortfolioState
from core import vault_writer
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
from core.self_optimizer import SelfOptimizer
from core.confidence_calibrator import ConfidenceCalibrator
from core.earnings_calendar import EarningsCalendar
from core.regime_classifier import RegimeClassifier
from core.regime_strategy import RegimeStrategySelector
from core.session_analyzer import SessionAnalyzer
from core.trading_friction import TradingFriction
from tools.correlation import CorrelationAnalyzer
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
        optimizer: SelfOptimizer | None = None,
    ):
        self._portfolio = portfolio
        self._risk = risk_manager
        self._executor = executor
        self._telegram = telegram
        self._llm = llm_client or LLMClient()
        self._optimizer = optimizer

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
        self._signal_tracker = SignalAccuracyTracker()
        self._regime_classifier = RegimeClassifier(market_data_fetcher=MarketDataFetcher())
        self._earnings_cal = EarningsCalendar()
        self._session_analyzer = SessionAnalyzer()
        self._confidence_cal = ConfidenceCalibrator()
        self._regime_strategy = RegimeStrategySelector()

        # Trading friction simulator (active in paper mode only)
        self._friction = TradingFriction(paper_mode=getattr(executor, "paper_mode", True))
        self._portfolio.set_friction(self._friction)

        # Signal funnel stats — in-memory counters, reset daily
        self._funnel_stats: dict[str, Any] = {
            "signals_generated": 0,
            "pre_filtered": 0,
            "analyst_no_trade": 0,
            "analyst_errors": 0,
            "devil_killed": 0,
            "risk_rejected": 0,
            "rr_rejected": 0,
            "executed": 0,
            "last_reset": datetime.now(timezone.utc).isoformat(),
        }

        # Load risk params for auto-recovery cooldown + default stop-loss
        self._risk_params: dict[str, Any] = {}
        try:
            params_path = Path(__file__).resolve().parent.parent / "config" / "risk_params.json"
            if params_path.exists():
                self._risk_params = json.loads(params_path.read_text())
        except Exception as e:
            log.warning("Could not load risk_params.json: %s", e)

        # Track deferred close attempts (log once per asset, not every heartbeat)
        self._deferred_closes: set[str] = set()

        # Backfill any legacy positions missing fields
        self._backfill_legacy_positions()

    def _maybe_reset_funnel_stats(self) -> None:
        """Reset funnel stats if the day has changed (UTC)."""
        last = self._funnel_stats.get("last_reset", "")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not last.startswith(today):
            self._funnel_stats = {
                "signals_generated": 0,
                "pre_filtered": 0,
                "analyst_no_trade": 0,
                "analyst_errors": 0,
                "devil_killed": 0,
                "risk_rejected": 0,
                "rr_rejected": 0,
                "executed": 0,
                "last_reset": datetime.now(timezone.utc).isoformat(),
            }

    def get_funnel_stats(self) -> dict[str, Any]:
        """Return funnel stats with computed pass-through rates."""
        self._maybe_reset_funnel_stats()
        s = dict(self._funnel_stats)
        generated = s["signals_generated"]

        past_prefilter = generated - s["pre_filtered"]
        past_analyst = past_prefilter - s["analyst_no_trade"] - s["analyst_errors"]
        past_devil = past_analyst - s["devil_killed"]
        past_risk = past_devil - s["risk_rejected"] - s["rr_rejected"]

        s["past_prefilter"] = max(past_prefilter, 0)
        s["past_analyst"] = max(past_analyst, 0)
        s["past_devil"] = max(past_devil, 0)
        s["past_risk"] = max(past_risk, 0)

        s["prefilter_pass_rate"] = round(past_prefilter / generated * 100, 1) if generated > 0 else 0.0
        s["analyst_pass_rate"] = round(past_analyst / past_prefilter * 100, 1) if past_prefilter > 0 else 0.0
        s["devil_pass_rate"] = round(past_devil / past_analyst * 100, 1) if past_analyst > 0 else 0.0
        s["risk_pass_rate"] = round(past_risk / past_devil * 100, 1) if past_devil > 0 else 0.0
        s["execution_rate"] = round(s["executed"] / generated * 100, 1) if generated > 0 else 0.0

        return s

    def run_chart_scan(self) -> list[SignalAlert]:
        """Run a technical chart scan and process any signals found.

        Tier 0: pure Python, no LLM calls. Generates SignalAlerts from
        technical indicator triggers (RSI, MACD, Bollinger, EMA crossovers).
        """
        if self._portfolio.halted:
            log.warning("System halted — skipping chart scan")
            return []

        try:
            from tools.chart_scanner import ChartScanner

            log.info("Starting chart scan...")
            event_bus.emit("pipeline", "chart_scan_start", {})
            scanner = ChartScanner()
            signals = scanner.scan_all()
            log.info("Chart scan produced %d signals", len(signals))
            event_bus.emit("pipeline", "chart_scan_complete", {
                "signal_count": len(signals),
                "signals": [{"headline": s.headline, "asset": s.asset, "strength": s.signal_strength} for s in signals],
            })

            # Assign signal_id and record in signal tracker
            for sig in signals:
                sig.signal_id = f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{sig.asset}_{uuid.uuid4().hex[:4]}"
                self._signal_tracker.record_signal(sig.signal_id, sig, source_type="chart_scan")

            if signals:
                self.process_signals(signals)

            return signals
        except Exception as e:
            log.error("Chart scan failed: %s", e)
            self._telegram.send_alert(f"Chart scan error: {e}")
            return []

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

            # Assign signal_id and record in signal tracker
            for sig in signals:
                sig.signal_id = f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{sig.asset}_{uuid.uuid4().hex[:4]}"
                self._signal_tracker.record_signal(sig.signal_id, sig, source_type="news_scan")

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
        self._maybe_reset_funnel_stats()
        self._funnel_stats["signals_generated"] += len(signals)

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

        # ── Pre-LLM filter (Tier 0, free) ─────────────────────────────────
        # Note: duplicate asset is no longer a hard pre-filter.
        # The Devil's Advocate will flag it as a regular challenge instead,
        # allowing high-confidence add-to-position or reversal signals through.
        held_assets = {
            pos.get("asset") for pos in self._portfolio.open_positions
        }
        if signal.asset in held_assets:
            log.info(
                "PRE-FILTER: %s already held — passing through to DA for review",
                signal.asset,
            )

        # Log earnings context (informational only — don't block)
        earnings_days = self._earnings_cal.days_until_earnings(signal.asset)
        if earnings_days is not None:
            log.info(
                "EARNINGS CONTEXT: %s has earnings in %d days",
                signal.asset, earnings_days,
            )

        # Step 1: Market Analyst generates thesis
        try:
            thesis = self._analyst.analyze_signal(signal)
        except Exception as e:
            log.error("Market Analyst failed for %s: %s", signal.asset, e)
            result["outcome"] = "analyst_error"
            result["error"] = str(e)
            self._funnel_stats["analyst_errors"] += 1
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "analyst_error")
            return {"signal": signal, "result": result, "thesis": None, "verdict": None}

        if thesis is None:
            log.info("No trade thesis for %s — recording no-trade", signal.asset)
            event_bus.emit("pipeline", "no_thesis", {"asset": signal.asset, "signal_headline": signal.headline})
            try:
                self._journal.record_no_trade(signal, "No thesis generated")
            except Exception as e:
                log.error("Journal record_no_trade failed: %s", e)
            result["outcome"] = "no_trade"
            self._funnel_stats["analyst_no_trade"] += 1
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "no_trade")
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

            # Compute actual correlations between candidate asset and open positions
            open_positions = portfolio_state.get("open_positions", [])
            if open_positions:
                try:
                    mdf = MarketDataFetcher()
                    corr_analyzer = CorrelationAnalyzer(market_data_fetcher=mdf)
                    candidate_asset = thesis.asset
                    candidate_ohlcv = mdf.get_ohlcv(candidate_asset, period="1mo", interval="1d")
                    candidate_prices = [bar["close"] for bar in candidate_ohlcv if "close" in bar]

                    corr_pairs = []
                    for pos in open_positions:
                        pos_asset = pos.get("asset", "")
                        if not pos_asset or pos_asset == candidate_asset:
                            continue
                        try:
                            pos_ohlcv = mdf.get_ohlcv(pos_asset, period="1mo", interval="1d")
                            pos_prices = [bar["close"] for bar in pos_ohlcv if "close" in bar]
                            corr = corr_analyzer.pairwise_correlation(candidate_prices, pos_prices)
                            corr_pairs.append(f"{candidate_asset}-{pos_asset}: {corr:.2f}")
                        except Exception:
                            continue

                    if corr_pairs:
                        portfolio_state["actual_correlations"] = (
                            f"Actual 30d correlations with open positions: {', '.join(corr_pairs)}"
                        )
                except Exception as e:
                    log.debug("Correlation computation skipped: %s", e)

            verdict = self._devil.challenge(thesis, portfolio_state)
        except Exception as e:
            log.error("Devil's Advocate failed for %s: %s", signal.asset, e)
            result["outcome"] = "devil_error"
            result["error"] = str(e)
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "devil_error")
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
                self._journal.record_killed_trade(thesis, verdict, signal=signal, killed_by="devils_advocate")
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
            self._funnel_stats["devil_killed"] += 1
            if signal.signal_id:
                self._signal_tracker.record_outcome(
                    signal.signal_id, "killed",
                    killed_by="devils_advocate",
                    kill_reason=verdict.final_reasoning[:200],
                )
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
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "risk_error")
            return result

        event_bus.emit("pipeline", "risk_check", {
            "asset": signal.asset, "approved": approved, "reason": reason,
        })

        if not approved:
            log.info("Trade rejected by Risk Manager: %s", reason)
            try:
                self._journal.record_killed_trade(thesis, verdict, signal=signal, killed_by="risk_manager")
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
            self._funnel_stats["risk_rejected"] += 1
            if signal.signal_id:
                self._signal_tracker.record_outcome(
                    signal.signal_id, "risk_rejected",
                    killed_by="risk_manager", kill_reason=reason,
                )
            return result

        order_to_execute = adjusted or exec_order

        # Step 3b: Reward:Risk ratio check
        current_price = thesis.supporting_data.get("current_price", 0)
        rr_passes, rr_ratio, rr_reason = self._check_reward_risk_ratio(
            order_to_execute, current_price, thesis.direction.value
        )
        if not rr_passes:
            log.warning(
                "Trade rejected by R:R check for %s: %s", signal.asset, rr_reason
            )
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="rr_check",
                reason=rr_reason,
                entry_price=current_price,
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "rr_rejected"
            result["rr_reason"] = rr_reason
            result["rr_ratio"] = rr_ratio
            self._funnel_stats["rr_rejected"] += 1
            if signal.signal_id:
                self._signal_tracker.record_outcome(
                    signal.signal_id, "rr_rejected",
                    killed_by="rr_check", kill_reason=rr_reason,
                )
            return result

        # Step 4: Execute
        try:
            confirmation = self._executor.execute(order_to_execute)
        except Exception as e:
            log.error("Execution failed for %s: %s", signal.asset, e)
            result["outcome"] = "execution_error"
            result["error"] = str(e)
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "execution_error")
            return result

        if confirmation.get("type") == "order_error":
            log.warning("Order error: %s", confirmation.get("error"))
            result["outcome"] = "execution_error"
            result["error"] = confirmation.get("error", "")
            if signal.signal_id:
                self._signal_tracker.record_outcome(signal.signal_id, "execution_error")
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
            fill_price = confirmation.get("fill_price", 0.0)
            fill_qty = confirmation.get("quantity", order_to_execute.get("quantity", 0))
            self._portfolio.add_position({
                "trade_id": confirmation.get("order_id", ""),
                "asset": thesis.asset,
                "direction": thesis.direction.value,
                "entry_price": fill_price,
                "position_size_pct": thesis.suggested_position_pct,
                "quantity": fill_qty,
                "stop_loss_price": order_to_execute.get("stop_loss"),
                "take_profit_price": order_to_execute.get("take_profit"),
                "timestamp_open": datetime.now(timezone.utc).isoformat(),
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
            })
            # Deduct entry friction from equity
            entry_friction = self._friction.total_entry_cost(
                thesis.asset, fill_price, fill_qty, thesis.direction.value
            )
            if entry_friction > 0:
                self._portfolio.update_equity(self._portfolio.equity - entry_friction)
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
        self._funnel_stats["executed"] += 1
        if signal.signal_id:
            self._signal_tracker.record_outcome(
                signal.signal_id, "executed",
                trade_id=str(confirmation.get("order_id", "")),
                direction=thesis.direction.value,
                entry_price=confirmation.get("fill_price", 0.0),
            )
        event_bus.emit("pipeline", "trade_executed", {
            "asset": thesis.asset, "direction": thesis.direction.value,
            "fill_price": confirmation.get("fill_price", 0), "quantity": confirmation.get("quantity", 0),
        })

        # Write trade to memory vault
        vault_writer.write_trade(
            asset=thesis.asset,
            direction=thesis.direction.value,
            entry_price=confirmation.get("fill_price", 0),
            thesis=thesis.thesis if hasattr(thesis, "thesis") else signal.headline,
            verdict=result.get("devil_verdict", ""),
            outcome="executed",
        )

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

    def update_trailing_stops(self) -> list[dict[str, Any]]:
        """Update trailing stop-loss levels for open positions.

        Called every 5 minutes from the heartbeat cycle, BEFORE check_stop_losses().
        Tier 0: pure Python, no LLM, deterministic.

        For each position that has moved favorably beyond the activation threshold,
        ratchet the stop-loss price upward (long) or downward (short) to lock in profits.
        The stop never moves backward (against profit protection).
        """
        if self._portfolio.halted:
            log.info("System halted — skipping trailing stop update")
            return []

        activation_pct = self._risk_params.get("trailing_stop_activation_pct", 2.0) / 100.0
        distance_pct = self._risk_params.get("trailing_stop_distance_pct", 1.5) / 100.0
        atr_mult = self._risk_params.get("trailing_stop_atr_mult", 1.0)

        positions = self._portfolio.open_positions
        updated: list[dict[str, Any]] = []
        market_data = MarketDataFetcher()

        for pos in positions:
            stop_price = pos.get("stop_loss_price")
            if stop_price is None:
                continue

            asset = pos.get("asset", "")
            trade_id = pos.get("trade_id", "")
            direction = pos.get("direction", "long")
            entry_price = pos.get("entry_price", 0)
            if entry_price <= 0:
                continue

            # Fetch current price
            try:
                price_data = market_data.get_price(asset)
                current_price = price_data.get("price", 0)
            except Exception as e:
                log.error("Trailing stop: price fetch failed for %s: %s", asset, e)
                continue

            if current_price <= 0:
                continue

            # Check if price has moved favorably beyond activation threshold
            if direction == "long":
                favorable_move_pct = (current_price - entry_price) / entry_price
            else:
                favorable_move_pct = (entry_price - current_price) / entry_price

            if favorable_move_pct < activation_pct:
                continue

            # Classify regime for adaptive trail distance
            try:
                regime_info = self._regime_classifier.classify(asset)
                regime = regime_info.get("regime", "RANGING")
                pos["regime"] = regime
            except Exception as e:
                log.debug("Regime classification failed for %s: %s", asset, e)
                regime = "RANGING"

            regime_mult = self._regime_classifier.get_trailing_multiplier(
                regime, self._risk_params
            )

            # Calculate trail distance: max(ATR-based, flat %) * regime multiplier
            atr = pos.get("supporting_data", {}).get("atr_14") if isinstance(pos.get("supporting_data"), dict) else None
            if atr and atr > 0:
                trail_distance = max(atr * atr_mult, current_price * distance_pct) * regime_mult
            else:
                trail_distance = current_price * distance_pct * regime_mult

            # Calculate new stop price
            if direction == "long":
                new_stop = current_price - trail_distance
                # Only move stop upward (protective direction)
                if new_stop <= stop_price:
                    continue
            else:
                new_stop = current_price + trail_distance
                # Only move stop downward (protective direction)
                if new_stop >= stop_price:
                    continue

            new_stop = round(new_stop, 2)

            # Preserve original stop-loss price on first activation
            was_active = pos.get("trailing_stop_active", False)
            if not was_active:
                pos["original_stop_loss_price"] = stop_price
                pos["trailing_stop_active"] = True

            old_stop = stop_price
            pos["stop_loss_price"] = new_stop

            log.info(
                "TRAILING STOP [%s] %s %s (trade %s) stop $%.2f->$%.2f (price $%.2f trail $%.2f mult=%.1f)",
                regime, direction.upper(), asset, trade_id, old_stop, new_stop, current_price, trail_distance, regime_mult,
            )

            event_bus.emit("trailing_stop", "updated", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "current_price": current_price,
                "trail_distance": round(trail_distance, 2),
                "trailing_stop_active": True,
                "regime": regime,
                "regime_multiplier": regime_mult,
            })

            # Send Telegram alert on first activation
            if not was_active:
                try:
                    self._telegram.send_alert(
                        f"TRAILING STOP activated: {direction.upper()} {asset}\n"
                        f"Entry: ${entry_price:.2f} | Price: ${current_price:.2f}\n"
                        f"Stop moved: ${old_stop:.2f} → ${new_stop:.2f}"
                    )
                except Exception as e:
                    log.error("Trailing stop Telegram alert failed: %s", e)

            updated.append({
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "current_price": current_price,
                "first_activation": not was_active,
            })

        if updated:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Portfolio persist after trailing stop updates failed: %s", e)

        return updated

    def _resolve_close_quantity(self, asset: str, internal_qty: float, trade_id: str) -> float | None:
        """Resolve the actual quantity to close by checking the broker.

        Returns the broker's qty if the position exists, or None if the position
        doesn't exist on the broker (meaning we should just clean up internal state).
        """
        from core.alpaca_executor import AlpacaExecutor
        if not isinstance(self._executor, AlpacaExecutor):
            return internal_qty

        broker_pos = self._executor.get_broker_position(asset)
        if broker_pos is None or broker_pos["qty"] == 0:
            log.warning(
                "STALE POSITION: %s (trade %s) — internal qty=%.4f but broker has no position. "
                "Removing from internal portfolio.",
                asset, trade_id, internal_qty,
            )
            return None

        broker_qty = broker_pos["qty"]
        if abs(broker_qty - internal_qty) > 0.0001:
            log.warning(
                "QTY MISMATCH: %s (trade %s) — internal=%.4f, broker=%.4f. Using broker qty.",
                asset, trade_id, internal_qty, broker_qty,
            )
        return broker_qty

    @staticmethod
    def _is_crypto(asset: str) -> bool:
        """Check if asset is crypto (trades 24/7, no market hours gate)."""
        return asset in ("BTC", "ETH")

    def _close_broker_position(self, asset: str, trade_id: str, quantity: float,
                                direction: str, thesis_prefix: str) -> dict[str, Any]:
        """Close a position on the broker, using DELETE endpoint for Alpaca to avoid wash trades.

        Falls back to counter-direction order for non-Alpaca executors.
        Skips stock/ETF closes when US market is closed (crypto trades 24/7).
        """
        from core.alpaca_executor import AlpacaExecutor
        if isinstance(self._executor, AlpacaExecutor):
            # Gate: skip stock/ETF closes when market is closed
            if not self._is_crypto(asset) and not self._executor.is_market_open():
                if asset not in self._deferred_closes:
                    self._deferred_closes.add(asset)
                    log.info(
                        "DEFERRED CLOSE: %s (trade %s) — US market closed, will retry at market open",
                        asset, trade_id,
                    )
                return {
                    "type": "order_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"Market closed — deferring close for {asset}",
                }
            # Market is open — clear deferred flag
            self._deferred_closes.discard(asset)
            return self._executor.close_position(asset)

        # Non-Alpaca: use counter-direction order
        close_direction = "short" if direction == "long" else "long"
        close_order = {
            "type": "execution_order",
            "thesis_id": f"{thesis_prefix}_{trade_id}",
            "asset": asset,
            "direction": close_direction,
            "quantity": quantity,
            "order_type": "market",
            "stop_loss": None,
        }
        return self._executor.execute(close_order)

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

            internal_qty = pos.get("quantity", 0)
            quantity = self._resolve_close_quantity(asset, internal_qty, trade_id)

            if quantity is None:
                # Position doesn't exist on broker — clean up internal state
                self._portfolio.remove_position(trade_id)
                self._portfolio.record_trade(0)  # no P&L — already closed at broker
                try:
                    self._journal.record_exit(trade_id, {
                        "asset": asset, "exit_price": current_price, "pnl_usd": 0,
                        "pnl_pct": 0, "exit_reason": "stale_cleanup_stop_loss",
                    })
                except Exception:
                    pass
                closed.append({"trade_id": trade_id, "asset": asset, "reason": "stale_position_cleanup"})
                continue

            try:
                confirmation = self._close_broker_position(asset, trade_id, quantity, direction, "sl_close")
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

            # Deduct exit friction from equity
            exit_friction = self._friction.total_exit_cost(asset, exit_price, quantity, direction)
            if exit_friction > 0:
                self._portfolio.update_equity(self._portfolio.equity - exit_friction)

            event_bus.emit("stop_loss", "triggered", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
            })

            # Record exit in journal with exit_reason + MAE/MFE
            try:
                pnl_pct = ((pnl) / (entry_price * quantity) * 100) if entry_price and quantity else 0
                ts_open = pos.get("timestamp_open")
                hold_hours = 0.0
                if ts_open:
                    try:
                        opened_at = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
                        if opened_at.tzinfo is None:
                            opened_at = opened_at.replace(tzinfo=timezone.utc)
                        hold_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                    except (ValueError, TypeError):
                        pass
                self._journal.record_exit(trade_id, {
                    "asset": asset,
                    "exit_price": exit_price,
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": "stop_loss",
                    "hold_duration_hours": round(hold_hours, 1),
                    "mae_pct": pos.get("mae_pct", 0.0),
                    "mfe_pct": pos.get("mfe_pct", 0.0),
                })
            except Exception as e:
                log.error("Stop-loss journal record failed: %s", e)

            # Extract principles from this closed trade
            self._extract_trade_principles(trade_id)

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

            internal_qty = pos.get("quantity", 0)
            quantity = self._resolve_close_quantity(asset, internal_qty, trade_id)

            if quantity is None:
                # Position doesn't exist on broker — clean up internal state
                self._portfolio.remove_position(trade_id)
                self._portfolio.record_trade(0)
                try:
                    self._journal.record_exit(trade_id, {
                        "asset": asset, "exit_price": current_price, "pnl_usd": 0,
                        "pnl_pct": 0, "exit_reason": "stale_cleanup_take_profit",
                    })
                except Exception:
                    pass
                closed.append({"trade_id": trade_id, "asset": asset, "reason": "stale_position_cleanup"})
                continue

            try:
                confirmation = self._close_broker_position(asset, trade_id, quantity, direction, "tp_close")
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

            # Deduct exit friction from equity
            exit_friction = self._friction.total_exit_cost(asset, exit_price, quantity, direction)
            if exit_friction > 0:
                self._portfolio.update_equity(self._portfolio.equity - exit_friction)

            event_bus.emit("take_profit", "triggered", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "target_price": tp_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
            })

            # Record exit in journal with exit_reason + MAE/MFE
            try:
                pnl_pct = ((pnl) / (entry_price * quantity) * 100) if entry_price and quantity else 0
                ts_open = pos.get("timestamp_open")
                hold_hours = 0.0
                if ts_open:
                    try:
                        opened_at = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
                        if opened_at.tzinfo is None:
                            opened_at = opened_at.replace(tzinfo=timezone.utc)
                        hold_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                    except (ValueError, TypeError):
                        pass
                self._journal.record_exit(trade_id, {
                    "asset": asset,
                    "exit_price": exit_price,
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": "take_profit",
                    "hold_duration_hours": round(hold_hours, 1),
                    "mae_pct": pos.get("mae_pct", 0.0),
                    "mfe_pct": pos.get("mfe_pct", 0.0),
                })
            except Exception as e:
                log.error("Take-profit journal record failed: %s", e)

            # Extract principles from this closed trade
            self._extract_trade_principles(trade_id)

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

    _MAX_HOLDING_HOURS = 72

    def _get_holding_period_for_asset(self, asset: str) -> float:
        """Get regime-dependent holding period for an asset.

        Uses the regime classifier to determine market regime, then looks up
        holding_period_hours from regime_strategy config. Falls back to 72 hours.
        """
        try:
            regime_info = self._regime_classifier.classify(asset)
            regime = regime_info.get("regime", "")
        except Exception:
            return float(self._MAX_HOLDING_HOURS)

        preset = self._regime_strategy._presets.get(regime, {})
        base_hours = preset.get("holding_period_hours", self._MAX_HOLDING_HOURS)
        mult = preset.get("max_hold_hours_mult", 1.0)
        return float(base_hours) * float(mult)

    def check_holding_periods(self) -> list[dict[str, Any]]:
        """Force-close positions exceeding regime-dependent holding periods.

        Called every 5 minutes from the heartbeat cycle.
        Tier 0: pure Python, no LLM, deterministic.
        Holding periods vary by regime:
          TRENDING_UP/DOWN: 168h, RANGING: 48h, HIGH_VOL: 24h, LOW_VOL: 96h
        Falls back to 72h if regime data unavailable.
        """
        if self._portfolio.halted:
            log.info("System halted — skipping holding period check")
            return []

        positions = self._portfolio.open_positions[:]
        closed: list[dict[str, Any]] = []
        market_data = MarketDataFetcher()
        now = datetime.now(timezone.utc)

        for pos in positions:
            ts_open = pos.get("timestamp_open")
            if not ts_open:
                continue

            try:
                opened_at = datetime.fromisoformat(ts_open.replace("Z", "+00:00"))
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            asset = pos.get("asset", "")
            max_hours = self._get_holding_period_for_asset(asset)

            holding_hours = (now - opened_at).total_seconds() / 3600
            if holding_hours < max_hours:
                continue

            trade_id = pos.get("trade_id", "")
            direction = pos.get("direction", "long")

            log.warning(
                "TIME EXIT: %s %s (trade %s) — held %.1f hours (max %.0f)",
                direction.upper(), asset, trade_id, holding_hours, max_hours,
            )

            # Fetch current price for P&L calculation
            try:
                price_data = market_data.get_price(asset)
                current_price = price_data.get("price", 0)
            except Exception as e:
                log.error("Time exit: price fetch failed for %s: %s", asset, e)
                continue

            if current_price <= 0:
                log.warning("Time exit: no valid price for %s, skipping", asset)
                continue

            internal_qty = pos.get("quantity", 0)
            quantity = self._resolve_close_quantity(asset, internal_qty, trade_id)

            if quantity is None:
                # Position doesn't exist on broker — clean up internal state
                self._portfolio.remove_position(trade_id)
                self._portfolio.record_trade(0)
                try:
                    self._journal.record_exit(trade_id, {
                        "asset": asset, "exit_price": current_price, "pnl_usd": 0,
                        "pnl_pct": 0, "exit_reason": "stale_cleanup_time_exit",
                    })
                except Exception:
                    pass
                closed.append({"trade_id": trade_id, "asset": asset, "reason": "stale_position_cleanup"})
                continue

            try:
                confirmation = self._close_broker_position(asset, trade_id, quantity, direction, "time_close")
            except Exception as e:
                log.error("Time exit close failed for %s: %s", trade_id, e)
                continue

            if confirmation.get("type") == "order_error":
                log.error(
                    "Time exit close order error for %s: %s",
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

            # Deduct exit friction from equity
            exit_friction = self._friction.total_exit_cost(asset, exit_price, quantity, direction)
            if exit_friction > 0:
                self._portfolio.update_equity(self._portfolio.equity - exit_friction)

            event_bus.emit("pipeline", "time_exit", {
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "holding_hours": round(holding_hours, 1),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
            })

            # Record exit in journal with exit_reason
            try:
                pnl_pct = ((pnl) / (entry_price * quantity) * 100) if entry_price and quantity else 0
                self._journal.record_exit(trade_id, {
                    "asset": asset,
                    "exit_price": exit_price,
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": "time_exit",
                    "hold_duration_hours": round(holding_hours, 1),
                    "mae_pct": pos.get("mae_pct", 0.0),
                    "mfe_pct": pos.get("mfe_pct", 0.0),
                })
            except Exception as e:
                log.error("Time exit journal record failed: %s", e)

            # Extract principles from this closed trade
            self._extract_trade_principles(trade_id)

            try:
                self._telegram.send_alert(
                    f"TIME EXIT ({max_hours:.0f}h): {direction.upper()} {asset}\n"
                    f"Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}\n"
                    f"Held: {holding_hours:.1f}h | P&L: ${pnl:.2f}"
                )
            except Exception as e:
                log.error("Time exit Telegram alert failed: %s", e)

            closed.append({
                "trade_id": trade_id,
                "asset": asset,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_hours": round(holding_hours, 1),
                "pnl": round(pnl, 2),
            })

            log.info("Time exit close complete: %s %s — P&L: $%.2f (held %.1fh)", asset, trade_id, pnl, holding_hours)

        if closed:
            try:
                self._portfolio.persist()
            except Exception as e:
                log.error("Portfolio persist after time exit closures failed: %s", e)

        return closed

    def sync_portfolio_with_broker(self) -> None:
        """Sync internal equity with broker's reported equity.

        Broker is the source of truth — it knows actual fills, borrow costs,
        and available balance. Internal equity is overridden to match.
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
            old_equity = self._portfolio.equity
            if broker_equity > 0:
                self._portfolio.equity = broker_equity
                self._portfolio.persist()
            log.info(
                "Broker sync: equity=$%.2f (was $%.2f, diff=$%.2f), cash=$%.2f",
                broker_equity, old_equity, broker_equity - old_equity, broker_cash,
            )
        except Exception as e:
            log.error("Broker sync failed: %s", e)

    def _extract_trade_principles(self, trade_id: str) -> None:
        """Extract and save principles from a closed trade via the SelfOptimizer.

        Non-blocking: failures are logged but do not affect trade flow.
        """
        if not self._optimizer:
            return

        try:
            # Load the closed trade record from the journal
            journal_path = Path(__file__).resolve().parent.parent / "data" / "trade_journal.json"
            if not journal_path.exists():
                return
            with open(journal_path) as f:
                entries = json.load(f)

            closed_trade = None
            for entry in entries:
                if isinstance(entry, dict) and entry.get("trade_id") == trade_id and entry.get("outcome"):
                    closed_trade = entry
                    break

            if not closed_trade:
                return

            principles = self._optimizer.extract_principles(closed_trade)
            if principles:
                self._optimizer.save_principles(principles, closed_trade)
        except Exception as e:
            log.error("Principle extraction failed for %s: %s", trade_id, e)

    def run_proactive_scan(self) -> list[dict[str, Any]]:
        """Proactively evaluate all assets for trade setups based on technicals + regime.

        Runs independently of news — constructs synthetic signals from current
        regime, technicals, and price action for each unheld asset. This solves
        the undertrading problem by generating trade candidates 3x daily.
        """
        if self._portfolio.halted:
            log.warning("System halted — skipping proactive scan")
            return []

        try:
            from core.asset_registry import get_tradeable_assets

            log.info("Starting proactive scan...")
            event_bus.emit("pipeline", "proactive_scan_start", {})

            assets = get_tradeable_assets()
            held_assets = {
                pos.get("asset") for pos in self._portfolio.open_positions
            }

            # Filter out assets we already hold
            scan_assets = [a for a in assets if a not in held_assets]
            log.info(
                "Proactive scan: %d assets to scan (%d held, skipped)",
                len(scan_assets), len(held_assets),
            )

            if not scan_assets:
                log.info("Proactive scan: no assets to scan (all held)")
                event_bus.emit("pipeline", "proactive_scan_complete", {
                    "scanned": 0, "signals": 0, "theses": 0,
                })
                return []

            mdf = MarketDataFetcher()
            ti = self._analyst._ti
            signals: list[SignalAlert] = []

            for asset in scan_assets:
                try:
                    signal = self._build_proactive_signal(asset, mdf, ti)
                    if signal is not None:
                        signal.signal_id = (
                            f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                            f"_{asset}_{uuid.uuid4().hex[:4]}"
                        )
                        self._signal_tracker.record_signal(
                            signal.signal_id, signal, source_type="proactive_scan"
                        )
                        signals.append(signal)
                except Exception as e:
                    log.warning("Proactive scan: failed for %s: %s", asset, e)

            log.info(
                "Proactive scan produced %d signals from %d assets",
                len(signals), len(scan_assets),
            )

            results: list[dict[str, Any]] = []
            if signals:
                results = self.process_signals(signals)

            theses_count = sum(
                1 for r in results if r.get("outcome") not in ("no_trade", "pre_filtered", "analyst_error")
            )
            executed_count = sum(1 for r in results if r.get("outcome") == "executed")

            event_bus.emit("pipeline", "proactive_scan_complete", {
                "scanned": len(scan_assets),
                "signals": len(signals),
                "theses": theses_count,
                "executed": executed_count,
            })

            log.info(
                "Proactive scan complete: %d scanned, %d signals, %d theses, %d executed",
                len(scan_assets), len(signals), theses_count, executed_count,
            )

            return results
        except Exception as e:
            log.error("Proactive scan failed: %s", e)
            self._telegram.send_alert(f"Proactive scan error: {e}")
            return []

    def _build_proactive_signal(
        self, asset: str, mdf: MarketDataFetcher, ti: "TechnicalIndicators"
    ) -> SignalAlert | None:
        """Build a synthetic SignalAlert from technicals + regime for one asset.

        Returns None if insufficient data to form a meaningful signal.
        """
        # 1. Fetch OHLCV data
        ohlcv = mdf.get_ohlcv(asset, period="3mo", interval="1d")
        if not ohlcv or len(ohlcv) < 30:
            log.debug("Proactive scan: insufficient OHLCV for %s (%d bars)", asset, len(ohlcv) if ohlcv else 0)
            return None

        closes = [bar["close"] for bar in ohlcv]
        highs = [bar["high"] for bar in ohlcv]
        lows = [bar["low"] for bar in ohlcv]
        volumes = [bar["volume"] for bar in ohlcv]

        current_price = closes[-1] if closes else 0
        if current_price <= 0:
            return None

        # 2. Compute technicals
        rsi = ti.rsi(closes)
        macd = ti.macd(closes)
        bollinger = ti.bollinger_bands(closes)
        atr = ti.atr(highs, lows, closes)
        sma_50 = ti.sma(closes, 50)
        sma_200 = ti.sma(closes, 200)

        vol_avg_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0
        vol_ratio = round(volumes[-1] / vol_avg_20, 2) if vol_avg_20 > 0 else 0

        # 2b. Liquidity sweep detection + volume anomaly scoring
        sweep = ti.liquidity_sweep(highs, lows, closes)
        vol_anomaly = ti.volume_anomaly(volumes)

        # 3. Compute price action (% changes)
        def pct_change(n: int) -> float:
            if len(closes) > n and closes[-n - 1] > 0:
                return round((closes[-1] - closes[-n - 1]) / closes[-n - 1] * 100, 2)
            return 0.0

        change_1d = pct_change(1)
        change_5d = pct_change(5)
        change_20d = pct_change(20)

        # 4. Classify regime
        regime_info = self._regime_classifier.classify_from_ohlcv(ohlcv)
        regime = regime_info.get("regime", "RANGING")

        # 5. Get regime strategy recommendation
        regime_preset = self._regime_strategy.presets.get(regime, {})
        preferred_dir = regime_preset.get("prefer_direction")
        min_confidence = regime_preset.get("min_confidence", 0.5)

        # 6. Build MACD crossover description
        macd_desc = "neutral"
        if macd["histogram"] > 0:
            macd_desc = "bullish (MACD above signal)"
        elif macd["histogram"] < 0:
            macd_desc = "bearish (MACD below signal)"

        # 7. Bollinger position
        bb_pos = "middle"
        if bollinger["upper"] > 0 and bollinger["lower"] > 0:
            if current_price > bollinger["upper"]:
                bb_pos = "above upper band (overbought)"
            elif current_price < bollinger["lower"]:
                bb_pos = "below lower band (oversold)"
            elif bollinger["middle"] > 0:
                bb_pct = (current_price - bollinger["lower"]) / (bollinger["upper"] - bollinger["lower"]) if (bollinger["upper"] - bollinger["lower"]) > 0 else 0.5
                if bb_pct > 0.7:
                    bb_pos = "near upper band"
                elif bb_pct < 0.3:
                    bb_pos = "near lower band"

        # 8. Determine signal sentiment from technicals
        bullish_count = 0
        bearish_count = 0

        if rsi < 30:
            bullish_count += 1  # Oversold = potential reversal up
        elif rsi > 70:
            bearish_count += 1  # Overbought = potential reversal down
        elif rsi < 45:
            bearish_count += 1
        elif rsi > 55:
            bullish_count += 1

        if macd["histogram"] > 0:
            bullish_count += 1
        elif macd["histogram"] < 0:
            bearish_count += 1

        if sma_50 > 0 and sma_200 > 0:
            if sma_50 > sma_200:
                bullish_count += 1  # Golden cross territory
            else:
                bearish_count += 1  # Death cross territory

        if change_5d > 1:
            bullish_count += 1
        elif change_5d < -1:
            bearish_count += 1

        # Liquidity sweep signals
        if sweep["detected"]:
            if sweep["type"] == "bullish":
                bullish_count += 1
            elif sweep["type"] == "bearish":
                bearish_count += 1

        # Volume spike confirms direction of 1d move
        if vol_anomaly["level"] == "spike":
            if change_1d > 0:
                bullish_count += 1
            elif change_1d < 0:
                bearish_count += 1

        if bullish_count > bearish_count:
            sentiment = "bullish"
        elif bearish_count > bullish_count:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # 9. Calculate signal strength from regime confidence + technical clarity
        tech_clarity = abs(bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
        regime_conf = regime_info.get("confidence", 0.5)
        signal_strength = round(min(1.0, (tech_clarity * 0.6 + regime_conf * 0.4)), 2)
        signal_strength = max(signal_strength, 0.3)  # Floor at 0.3

        # 9b. Boost signal strength from sweep/volume anomaly
        if sweep["detected"]:
            sweep_matches = (
                (sweep["type"] == "bullish" and sentiment == "bullish")
                or (sweep["type"] == "bearish" and sentiment == "bearish")
            )
            if sweep_matches:
                signal_strength = round(min(1.0, signal_strength + 0.15), 2)
        if vol_anomaly["is_anomaly"]:
            signal_strength = round(min(1.0, signal_strength + 0.10), 2)

        # 10. Build the headline with key information
        direction_hint = preferred_dir or ("long" if sentiment == "bullish" else "short" if sentiment == "bearish" else "neutral")
        headline = (
            f"Proactive: {asset} {regime} RSI={rsi:.0f} "
            f"{change_5d:+.1f}% 5d"
        )[:100]

        # 11. Build rich new_information text for MarketAnalyst
        sma_cross = ""
        if sma_50 > 0 and sma_200 > 0:
            if sma_50 > sma_200:
                sma_cross = "SMA50 above SMA200 (bullish structure)"
            else:
                sma_cross = "SMA50 below SMA200 (bearish structure)"

        new_info = (
            f"Proactive technical scan for {asset}. "
            f"Regime: {regime} (confidence {regime_conf:.0%}). "
            f"RSI(14): {rsi:.1f}. MACD: {macd_desc}. "
            f"Bollinger: {bb_pos}, bandwidth {bollinger['bandwidth']:.4f}. "
            f"ATR(14): {atr:.2f}. Volume vs 20d avg: {vol_ratio:.2f}x. "
            f"Price action: 1d {change_1d:+.1f}%, 5d {change_5d:+.1f}%, 20d {change_20d:+.1f}%. "
            f"{sma_cross}. "
            f"Regime strategy: prefer {preferred_dir or 'any'} direction, min confidence {min_confidence:.0%}."
        )
        if sweep["detected"]:
            new_info += (
                f" LIQUIDITY SWEEP: {sweep['type']} sweep at {sweep['sweep_level']:.2f}, "
                f"{sweep['reclaim_pct']:.0%} reclaimed."
            )
        if vol_anomaly["is_anomaly"]:
            new_info += (
                f" VOLUME ANOMALY: {vol_anomaly['ratio']:.1f}x average ({vol_anomaly['level']})."
            )

        return SignalAlert(
            asset=asset,
            signal_strength=signal_strength,
            headline=headline,
            sentiment=sentiment,
            category="technical",
            source="proactive_scan",
            new_information=new_info,
            urgency="medium",
            confidence_in_classification=signal_strength,
        )

    def _backfill_legacy_positions(self) -> None:
        """Patch legacy positions missing timestamp_open, take_profit_price, or MAE/MFE.

        Positions opened before the acceleration commit (100e1f4) lack these
        fields, which breaks take-profit checks, 72h time exits, and MAE/MFE
        tracking. This runs once at startup and persists the patched state.
        """
        positions = self._portfolio.open_positions
        patched = 0

        for pos in positions:
            changed = False

            # Backfill timestamp_open — use current time (conservative: starts 72h clock now)
            if not pos.get("timestamp_open"):
                pos["timestamp_open"] = datetime.now(timezone.utc).isoformat()
                changed = True

            # Backfill take_profit_price from entry price + config default
            if pos.get("take_profit_price") is None:
                entry = pos.get("entry_price", 0)
                direction = pos.get("direction", "long")
                tp_pct = self._risk_params.get("default_take_profit_pct", 5.0) / 100.0
                if entry > 0 and tp_pct > 0:
                    if direction == "long":
                        pos["take_profit_price"] = round(entry * (1 + tp_pct), 2)
                    else:
                        pos["take_profit_price"] = round(entry * (1 - tp_pct), 2)
                    changed = True

            # Backfill MAE/MFE
            if "mae_pct" not in pos:
                pnl_pct = pos.get("unrealized_pnl_pct", 0.0)
                pos["mae_pct"] = min(pnl_pct, 0.0)
                changed = True
            if "mfe_pct" not in pos:
                pnl_pct = pos.get("unrealized_pnl_pct", 0.0)
                pos["mfe_pct"] = max(pnl_pct, 0.0)
                changed = True

            if changed:
                patched += 1
                log.info(
                    "Backfilled legacy position: %s (TP=$%s, timestamp=%s)",
                    pos.get("asset"), pos.get("take_profit_price"), pos.get("timestamp_open"),
                )

        if patched:
            log.info("Backfilled %d legacy positions — persisting", patched)
            self._portfolio.persist()

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
            # If permanent halt is set, never auto-recover
            if getattr(self._portfolio, "permanent_halt", False):
                log.info("PERMANENT HALT active — manual restart required")
                return
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

                # Permanent kill switch: in live mode, if drawdown exceeds threshold, halt permanently
                is_paper = getattr(self._executor, "paper_mode", True)
                perm_threshold = self._risk_params.get("permanent_halt_drawdown_pct", 25.0)
                drawdown = portfolio_state.get("drawdown_from_peak_pct", 0)
                if not is_paper and drawdown >= perm_threshold:
                    self._portfolio.halted = True
                    self._portfolio.permanent_halt = True
                    self._portfolio.persist()
                    log.critical(
                        "PERMANENT HALT: Portfolio drawdown %.1f%% exceeded %.1f%%. "
                        "Manual review required. Auto-recovery DISABLED.",
                        drawdown, perm_threshold,
                    )
                    event_bus.emit("circuit_breaker", "permanent_halt", {
                        "drawdown_pct": drawdown,
                        "threshold_pct": perm_threshold,
                    })
                    try:
                        self._telegram.send_alert(
                            f"PERMANENT HALT: Portfolio drawdown exceeded {perm_threshold:.0f}%.\n"
                            f"Current drawdown: {drawdown:.1f}%\n"
                            f"Manual review required. Auto-recovery DISABLED."
                        )
                    except Exception:
                        pass
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
        self._maybe_reset_funnel_stats()
        self._funnel_stats["signals_generated"] += 1
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
                self._journal.record_killed_trade(thesis, verdict, killed_by="devils_advocate")
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
            self._funnel_stats["devil_killed"] += 1
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
            try:
                self._journal.record_killed_trade(thesis, verdict, killed_by="risk_manager")
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
            self._funnel_stats["risk_rejected"] += 1
            return result

        order_to_execute = adjusted or exec_order

        # Reward:Risk ratio check
        current_price = thesis.supporting_data.get("current_price", 0)
        rr_passes, rr_ratio, rr_reason = self._check_reward_risk_ratio(
            order_to_execute, current_price, thesis.direction.value
        )
        if not rr_passes:
            log.warning(
                "Trade rejected by R:R check for %s: %s", thesis.asset, rr_reason
            )
            self._phantom.record_missed(
                asset=thesis.asset,
                direction=thesis.direction.value,
                confidence=thesis.confidence,
                killed_by="rr_check",
                reason=rr_reason,
                entry_price=current_price,
                suggested_position_pct=thesis.suggested_position_pct,
                thesis=thesis.thesis,
            )
            result["outcome"] = "rr_rejected"
            result["rr_reason"] = rr_reason
            result["rr_ratio"] = rr_ratio
            self._funnel_stats["rr_rejected"] += 1
            return result

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
            fill_price = confirmation.get("fill_price", 0.0)
            fill_qty = confirmation.get("quantity", order_to_execute.get("quantity", 0))
            self._portfolio.add_position({
                "trade_id": confirmation.get("order_id", ""),
                "asset": thesis.asset,
                "direction": thesis.direction.value,
                "entry_price": fill_price,
                "position_size_pct": thesis.suggested_position_pct,
                "quantity": fill_qty,
                "stop_loss_price": order_to_execute.get("stop_loss"),
                "take_profit_price": order_to_execute.get("take_profit"),
                "timestamp_open": datetime.now(timezone.utc).isoformat(),
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
            })
            # Deduct entry friction from equity
            entry_friction = self._friction.total_entry_cost(
                thesis.asset, fill_price, fill_qty, thesis.direction.value
            )
            if entry_friction > 0:
                self._portfolio.update_equity(self._portfolio.equity - entry_friction)
            self._portfolio.persist()
        except Exception as e:
            log.error("Portfolio update failed: %s", e)

        result["outcome"] = "executed"
        result["fill_price"] = confirmation.get("fill_price", 0.0)
        self._funnel_stats["executed"] += 1
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

    def _check_reward_risk_ratio(
        self, order: dict[str, Any], current_price: float, direction: str
    ) -> tuple[bool, float, str]:
        """Check if trade meets minimum reward:risk ratio.

        Returns (passes, ratio, reason).
        If stop_loss or take_profit or current_price is missing/zero, skip the check.
        """
        stop_loss = order.get("stop_loss")
        take_profit = order.get("take_profit")
        min_rr = self._risk_params.get("min_reward_risk_ratio", 2.0)

        if not stop_loss or not take_profit or not current_price:
            return True, 0.0, "skipped (missing stop_loss, take_profit, or entry_price)"

        if direction == "long":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        else:
            risk = stop_loss - current_price
            reward = current_price - take_profit

        if risk <= 0:
            return False, 0.0, f"invalid risk: stop_loss ({stop_loss}) at or beyond entry ({current_price})"

        ratio = reward / risk

        if ratio < min_rr:
            return (
                False,
                round(ratio, 2),
                f"R:R {ratio:.2f}:1 below minimum {min_rr:.1f}:1 "
                f"(risk=${risk:.2f}, reward=${reward:.2f})",
            )

        return True, round(ratio, 2), "passed"

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
        raw_confidence = confidence
        confidence = self._confidence_cal.calibrate_confidence(confidence)
        if confidence != raw_confidence:
            log.info("CONFIDENCE CALIBRATION: %.2f -> %.2f", raw_confidence, confidence)
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

        # ── Consecutive loss throttling ────────────────────────────────────
        consecutive_losses = self._portfolio.consecutive_losses
        if consecutive_losses >= 5:
            position_pct *= 0.25
            log.warning(
                "LOSS THROTTLE: %d consecutive losses — position reduced to %.1f%% (25%% of normal)",
                consecutive_losses, position_pct,
            )
        elif consecutive_losses >= 3:
            position_pct *= 0.5
            log.warning(
                "LOSS THROTTLE: %d consecutive losses — position reduced to %.1f%% (50%% of normal)",
                consecutive_losses, position_pct,
            )

        # ── Earnings proximity reduction ──────────────────────────────────
        earnings_days = self._risk_params.get("earnings_proximity_days", 3)
        earnings_reduction = self._risk_params.get("earnings_position_reduction", 0.5)
        if self._earnings_cal.has_earnings_soon(thesis.asset, days=earnings_days):
            position_pct *= earnings_reduction
            days_until = self._earnings_cal.days_until_earnings(thesis.asset)
            log.warning(
                "EARNINGS PROXIMITY: %s has earnings in %s days — position halved to %.1f%%",
                thesis.asset, days_until, position_pct,
            )

        # ── Session weight adjustment ─────────────────────────────────────
        try:
            current_hour = datetime.now(timezone.utc).hour
            session = self._session_analyzer.classify_session(current_hour)
            session_weight = self._session_analyzer.get_session_weight(thesis.asset, session)
            if session_weight != 1.0:
                position_pct *= session_weight
                log.info(
                    "SESSION WEIGHT: %s %s session=%.2f → position %.1f%%",
                    thesis.asset, session, session_weight, position_pct,
                )
        except Exception as e:
            log.debug("Session weight failed for %s: %s", thesis.asset, e)

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

        # Classify regime for adaptive stop sizing
        try:
            regime_info = self._regime_classifier.classify(thesis.asset)
            regime = regime_info.get("regime", "RANGING")
        except Exception:
            regime = "RANGING"
            regime_info = {"regime": regime, "confidence": 0.5}

        regime_stop_mult = self._regime_classifier.get_initial_stop_multiplier(
            regime, self._risk_params
        )

        # ── Regime strategy adjustments ───────────────────────────────────
        regime_adj = self._regime_strategy.get_adjustments(regime, thesis.direction.value)
        position_pct *= regime_adj["position_size_mult"]
        confidence += regime_adj.get("confidence_adjustment", 0.0)
        confidence = min(confidence, 1.0)

        should_trade, regime_reason = self._regime_strategy.should_trade(
            regime, confidence, thesis.direction.value
        )
        if not should_trade:
            log.info("REGIME FILTER: %s -- %s", thesis.asset, regime_reason)
            # Don't block -- reduce position to micro-size for data collection
            position_pct = min(position_pct, 2.0)

        # Default stop-loss if none provided: use ATR-based stop, fall back to flat %
        atr = thesis.supporting_data.get("atr_14")
        if stop_loss is None and current_price > 0:
            sl_atr_mult = self._risk_params.get("stop_loss_atr_mult", 2.0) * regime_stop_mult
            if atr and atr > 0:
                # ATR-based dynamic stop (regime-adjusted)
                if thesis.direction.value == "long":
                    stop_loss = round(current_price - (atr * sl_atr_mult), 2)
                else:
                    stop_loss = round(current_price + (atr * sl_atr_mult), 2)
                log.info("ATR stop [%s] at $%.2f (ATR=%.2f, mult=%.1f from $%.2f)", regime, stop_loss, atr, sl_atr_mult, current_price)
            else:
                # Flat % fallback when ATR is unavailable (regime-adjusted)
                sl_pct = self._risk_params.get("default_stop_loss_pct", 3.0) / 100.0 * regime_stop_mult
                if thesis.direction.value == "long":
                    stop_loss = round(current_price * (1 - sl_pct), 2)
                else:
                    stop_loss = round(current_price * (1 + sl_pct), 2)
                log.info("Default stop [%s] at $%.2f (%.1f%% from $%.2f)", regime, stop_loss, sl_pct * 100, current_price)

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

        # Ensure take-profit is at least the ATR-based or default percentage minimum
        tp_atr_mult = self._risk_params.get("take_profit_atr_mult", 3.0)
        if current_price > 0 and atr and atr > 0:
            # ATR-based minimum take-profit
            if thesis.direction.value == "long":
                min_tp = round(current_price + (atr * tp_atr_mult), 2)
                if take_profit is None or take_profit < min_tp:
                    take_profit = min_tp
                    log.info("Take-profit raised to ATR minimum $%.2f (ATR=%.2f, mult=%.1f)", take_profit, atr, tp_atr_mult)
            else:
                min_tp = round(current_price - (atr * tp_atr_mult), 2)
                if take_profit is None or take_profit > min_tp:
                    take_profit = min_tp
                    log.info("Take-profit raised to ATR minimum $%.2f (ATR=%.2f, mult=%.1f)", take_profit, atr, tp_atr_mult)
        else:
            # Flat % fallback when ATR is unavailable
            tp_pct = self._risk_params.get("default_take_profit_pct", 5.0) / 100.0
            if current_price > 0 and tp_pct > 0:
                if thesis.direction.value == "long":
                    min_tp = round(current_price * (1 + tp_pct), 2)
                    if take_profit is None or take_profit < min_tp:
                        take_profit = min_tp
                        log.info("Take-profit raised to default minimum $%.2f (%.0f%% from $%.2f)", take_profit, tp_pct * 100, current_price)
                else:
                    min_tp = round(current_price * (1 - tp_pct), 2)
                    if take_profit is None or take_profit > min_tp:
                        take_profit = min_tp
                        log.info("Take-profit raised to default minimum $%.2f (%.0f%% from $%.2f)", take_profit, tp_pct * 100, current_price)

        # ── Apply regime take-profit multiplier ──────────────────────────
        tp_mult = regime_adj.get("take_profit_mult", 1.0)
        if take_profit is not None and current_price > 0 and tp_mult != 1.0:
            tp_distance = abs(take_profit - current_price)
            tp_distance *= tp_mult
            if thesis.direction.value == "long":
                take_profit = round(current_price + tp_distance, 2)
            else:
                take_profit = round(current_price - tp_distance, 2)
            log.info("Regime TP adjusted [%s] to $%.2f (mult=%.1f)", regime, take_profit, tp_mult)

        # ── Ensure TP respects minimum R:R ratio after regime adjustment ──
        min_rr = self._risk_params.get("min_reward_risk_ratio", 2.0)
        if take_profit is not None and stop_loss and current_price > 0 and min_rr > 0:
            direction = thesis.direction.value
            if direction == "long":
                risk = current_price - stop_loss
                if risk > 0:
                    min_tp = current_price + (risk * min_rr)
                    if take_profit < min_tp:
                        log.info("TP floor applied: %.2f → %.2f (min R:R %.1f:1)", take_profit, min_tp, min_rr)
                        take_profit = round(min_tp, 2)
            elif direction == "short":
                risk = stop_loss - current_price
                if risk > 0:
                    min_tp = current_price - (risk * min_rr)
                    if take_profit > min_tp:
                        log.info("TP floor applied: %.2f → %.2f (min R:R %.1f:1)", take_profit, min_tp, min_rr)
                        take_profit = round(min_tp, 2)

        # ATR-based position sizing via RiskManager, fall back to %-based
        if current_price > 0 and equity > 0:
            if atr and atr > 0 and hasattr(self._risk, "calculate_position_size"):
                position_value = self._risk.calculate_position_size(confidence, atr, equity)
                # Update position_pct to reflect ATR-based sizing
                position_pct = min((position_value / equity) * 100.0, 7.0)
                position_pct = max(position_pct, 2.0)  # Minimum 2% micro position
                log.info("ATR position sizing: $%.2f (%.1f%% of $%.2f)", position_value, position_pct, equity)
            else:
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
            "regime": regime,
            "regime_confidence": regime_info.get("confidence", 0.5),
            "regime_stop_multiplier": regime_stop_mult,
            "regime_position_mult": regime_adj["position_size_mult"],
            "regime_tp_mult": regime_adj.get("take_profit_mult", 1.0),
        }
