"""Trading System — Main entry point.

Loads environment, initializes all Tier 0-3 modules and the intelligence
pipeline, registers all schedules (heartbeat, news scan, analysis sessions,
daily summary, weekly review, circuit breaker), and runs the event loop.
Exits cleanly on SIGINT/SIGTERM.
"""

import os
import signal
import sys
import threading
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv

from agents.trade_journal import TradeJournal
from agents.weekly_strategist import WeeklyStrategist
from core.cost_tracker import CostTracker
from core.event_bus import event_bus
from core.alpaca_executor import AlpacaExecutor
from core.executor import Executor
from core.paper_executor import PaperExecutor
from core.heartbeat import Heartbeat
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
from core.self_optimizer import SelfOptimizer
from dashboard.server import start_dashboard
from core.broker_sync import BrokerReconciler
from tools.telegram_bot import TelegramNotifier

log = setup_logger("trading.main")

# Global flag for graceful shutdown
_running = True

# Scheduler watchdog — updated every loop iteration
_last_scheduler_tick = time.monotonic()
_last_scheduler_lock = threading.Lock()

# Task timeouts (seconds)
_TASK_TIMEOUTS: dict[str, int] = {
    "heartbeat": 120,
    "news_scan": 300,
    "chart_scan": 300,
    "proactive_scan": 600,
    "asian_open": 600,
    "european_overlap": 600,
    "us_close": 600,
    "weekly_review": 900,
    "daily_summary": 120,
    "daily_reset": 120,
}


def _shutdown(signum, frame):
    global _running
    sig_name = signal.Signals(signum).name
    log.info("Received %s — shutting down gracefully", sig_name)
    _running = False
    # Force exit after 10s if threads don't cooperate
    threading.Timer(10.0, lambda: os._exit(0)).start()


def main():
    global _running, _last_scheduler_tick

    # 1. Load environment
    load_dotenv()
    log.info("Environment loaded")

    # 2. Initialize portfolio state (load persisted state if available)
    portfolio = PortfolioState()
    portfolio.load()
    log.info("Portfolio initialized — equity: $%.2f", portfolio.equity)

    # 3. Initialize Tier 0 modules
    risk_manager = RiskManager()
    log.info("Risk manager initialized")

    executor_mode = os.getenv("EXECUTOR_MODE", "paper").lower()
    if executor_mode == "live":
        from core.routing_executor import RoutingExecutor
        executor = RoutingExecutor()
    elif executor_mode == "alpaca":
        executor = AlpacaExecutor()
    elif executor_mode == "ibkr":
        executor = Executor()
    else:
        executor = PaperExecutor()
    log.info("Executor initialized (mode=%s, paper_mode=%s)", executor_mode, executor.paper_mode)

    telegram = TelegramNotifier()

    # 4. Initialize LLM client
    llm_client = LLMClient()
    cost_tracker = CostTracker()
    llm_client.set_cost_tracker(cost_tracker)
    log.info("LLM client initialized (mock_mode=%s)", llm_client.mock_mode)

    # 5. Initialize Tier 3 modules
    optimizer = SelfOptimizer(telegram=telegram, portfolio=portfolio, llm_client=llm_client)
    log.info("Self-optimizer initialized")

    journal = TradeJournal(llm_client=llm_client)
    strategist = WeeklyStrategist(
        llm_client=llm_client,
        optimizer=optimizer,
        telegram=telegram,
    )
    log.info("Weekly strategist initialized")

    # 6. Initialize trading pipeline
    pipeline = TradingPipeline(
        portfolio=portfolio,
        risk_manager=risk_manager,
        executor=executor,
        telegram=telegram,
        llm_client=llm_client,
        optimizer=optimizer,
    )
    log.info("Trading pipeline initialized")

    # Broker reconciliation on startup
    reconciler = BrokerReconciler(executor=executor, portfolio=portfolio, telegram=telegram)
    auto_fix = os.getenv("BROKER_SYNC_AUTO_FIX", "false").lower() == "true"
    report = reconciler.reconcile(auto_fix=auto_fix)
    if not report.is_clean:
        log.warning("Broker reconciliation found discrepancies:\n%s", report.summary)
    else:
        log.info("Broker reconciliation: all clean")

    # Log broker account status
    pipeline.sync_portfolio_with_broker()

    # 7. Initialize heartbeat (skip IBKR check in paper mode)
    heartbeat = Heartbeat(telegram_notifier=telegram, skip_ibkr=(executor_mode != "ibkr"))
    log.info("Heartbeat monitor initialized")

    # 8. Task wrapper with timeout + dashboard event emission
    def _emit_task(name, func):
        timeout_sec = _TASK_TIMEOUTS.get(name, 300)

        def wrapper():
            event_bus.emit("scheduler", "task_run", {"task_name": name})
            t0 = time.time()
            error_container: list = []

            def _target():
                try:
                    func()
                except Exception as e:
                    error_container.append(e)

            worker = threading.Thread(target=_target, daemon=True)
            worker.start()
            worker.join(timeout=timeout_sec)

            duration = round(time.time() - t0, 2)
            if worker.is_alive():
                log.error("TASK TIMEOUT: %s exceeded %ds limit (will be abandoned)", name, timeout_sec)
                event_bus.emit("scheduler", "task_timeout", {
                    "task_name": name, "timeout_sec": timeout_sec,
                })
                telegram.send_alert(f"Task timeout: {name} exceeded {timeout_sec}s — abandoned")
            elif error_container:
                log.error("Task %s failed: %s", name, error_container[0])
            event_bus.emit("scheduler", "task_complete", {
                "task_name": name, "duration_sec": duration,
                "timed_out": worker.is_alive(),
            })
        wrapper.__name__ = name
        return wrapper

    # 9. Define scheduled tasks
    def run_heartbeat():
        status = heartbeat.check()
        if not status.all_healthy:
            log.warning("Heartbeat reported failures: %s", status.failures)

        # Scheduler watchdog: check if main loop is still ticking
        with _last_scheduler_lock:
            idle_sec = time.monotonic() - _last_scheduler_tick
        if idle_sec > 1200:  # 20 minutes
            log.error("SCHEDULER WATCHDOG: main loop idle for %.0fs — forcing restart", idle_sec)
            telegram.send_alert(f"Scheduler watchdog: idle {idle_sec:.0f}s — restarting")
            event_bus.emit("watchdog", "scheduler_stuck", {"idle_sec": round(idle_sec)})
            from core import vault_writer
            vault_writer.write_incident(
                title=f"Scheduler stuck for {idle_sec:.0f}s",
                what=f"Scheduler watchdog detected main loop idle for {idle_sec:.0f}s. Forcing restart.",
                root_cause="Unknown — likely a hung LLM call or network timeout in a scheduled task.",
                fix="Auto-restart via os._exit(1) — systemd restarts the service.",
                tags=["system/scheduler", "system/watchdog"],
                severity="high",
            )
            time.sleep(2)  # Let Telegram/event bus flush
            os._exit(1)  # systemd will restart us

        # Periodic broker reconciliation (log-only unless BROKER_SYNC_AUTO_FIX=true)
        try:
            r = reconciler.reconcile(auto_fix=auto_fix)
            if not r.is_clean:
                log.warning("Broker drift detected: %s", r.summary)
        except Exception as e:
            log.error("Periodic reconciliation failed: %s", e)

        # Run trailing stop update + stop-loss + take-profit + holding period checks (Tier 0 — deterministic, every 5 min)
        pipeline.update_trailing_stops()
        pipeline.check_stop_losses()
        pipeline.check_take_profits()
        pipeline.check_holding_periods()
        # Recalculate equity with live market prices
        pipeline.recalculate_equity()
        # Run circuit breaker check alongside heartbeat
        pipeline.run_circuit_breaker_check()

    def run_chart_scan():
        log.info("Scheduled chart scan starting...")
        pipeline.run_chart_scan()

    def run_news_scan():
        log.info("Scheduled news scan starting...")
        pipeline.run_news_scan()

    def run_asian_open():
        log.info("Scheduled Asian Open analysis starting...")
        pipeline.run_scheduled_analysis("asian_open")

    def run_european_overlap():
        log.info("Scheduled European Overlap analysis starting...")
        pipeline.run_scheduled_analysis("european_overlap")

    def run_us_close():
        log.info("Scheduled US Close analysis starting...")
        pipeline.run_scheduled_analysis("us_close")

    def run_proactive_scan():
        log.info("Scheduled proactive scan starting...")
        pipeline.run_proactive_scan()

    def run_daily_summary():
        log.info("Running daily summary...")
        try:
            snapshot = portfolio.snapshot()
            telegram.send_daily_summary(snapshot)
        except Exception as e:
            log.error("Daily summary failed: %s", e)

    def run_daily_reset():
        log.info("Running daily portfolio reset...")
        try:
            portfolio.reset_daily()
            portfolio.persist()
        except Exception as e:
            log.error("Daily reset failed: %s", e)

    def run_weekly_review():
        log.info("Running weekly strategy review...")
        try:
            week_ending = datetime.utcnow().strftime("%Y-%m-%d")
            portfolio_state = portfolio.snapshot()
            weekly_package = journal.assemble_weekly_package(
                week_ending, portfolio_state=portfolio_state
            )
            strategist.review_week(weekly_package)
        except Exception as e:
            log.error("Weekly review failed: %s", e)
            telegram.send_alert(f"Weekly review error: {e}")

    # 10. Register all schedules (wrapped for dashboard events)
    schedule.every(5).minutes.do(_emit_task("heartbeat", run_heartbeat))
    schedule.every(15).minutes.do(_emit_task("news_scan", run_news_scan))
    schedule.every().day.at("00:55").do(_emit_task("chart_scan", run_chart_scan))            # 08:55 SGT — before Asian session
    schedule.every().day.at("06:55").do(_emit_task("chart_scan", run_chart_scan))            # 14:55 SGT — before European session
    schedule.every().day.at("12:55").do(_emit_task("chart_scan", run_chart_scan))            # 20:55 SGT — before US session
    schedule.every().day.at("18:55").do(_emit_task("chart_scan", run_chart_scan))            # 02:55 SGT — overnight scan
    schedule.every().day.at("00:00").do(_emit_task("asian_open", run_asian_open))            # 08:00 SGT
    schedule.every().day.at("08:00").do(_emit_task("european_overlap", run_european_overlap))  # 16:00 SGT
    schedule.every().day.at("14:00").do(_emit_task("us_close", run_us_close))               # 22:00 SGT
    schedule.every().day.at("01:00").do(_emit_task("proactive_scan", run_proactive_scan))   # 09:00 SGT — after Asian Open
    schedule.every().day.at("09:00").do(_emit_task("proactive_scan", run_proactive_scan))   # 17:00 SGT — after European Overlap
    schedule.every().day.at("15:00").do(_emit_task("proactive_scan", run_proactive_scan))   # 23:00 SGT — after US Close
    schedule.every().day.at("15:00").do(_emit_task("daily_summary", run_daily_summary))     # 23:00 SGT
    schedule.every().day.at("16:00").do(_emit_task("daily_reset", run_daily_reset))         # 00:00 SGT (next day)
    schedule.every().sunday.at("15:00").do(_emit_task("weekly_review", run_weekly_review))  # 23:00 SGT Sunday
    log.info("All schedules registered (15 total)")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 11. Start GA4 tracker (server-side analytics)
    from core.ga4_tracker import ga4_tracker
    ga4_tracker.start()

    # 12. Start dashboard server
    dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
    dashboard_host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    start_dashboard(portfolio, heartbeat, cost_tracker, host=dashboard_host, port=dashboard_port, pipeline=pipeline)

    # 13. Pre-flight API connectivity check
    if not llm_client.mock_mode:
        log.info("Pre-flight: testing LLM API keys...")
        for name, test_fn in [
            ("DeepSeek", lambda: llm_client.call_deepseek('Return {"ok":true}', "Respond with JSON.")),
            # ("Kimi", lambda: llm_client.call_kimi('Return {"ok":true}', "Respond with JSON.")),
            ("Anthropic", lambda: llm_client.call_anthropic("hi", "Reply with 1 word.", max_tokens=5)),
        ]:
            result = test_fn()
            if "error" in result:
                log.warning("PRE-FLIGHT FAIL: %s — %s", name, result["error"])
            else:
                log.info("PRE-FLIGHT OK: %s", name)

    # 14. Run initial checks
    log.info("Running initial heartbeat check...")
    run_heartbeat()

    log.info("Running initial news scan...")
    run_news_scan()

    # 15. Event loop
    log.info("Trading system started — entering main loop")
    try:
        while _running:
            schedule.run_pending()
            # Update watchdog tick
            with _last_scheduler_lock:
                _last_scheduler_tick = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 16. Cleanup
    log.info("Persisting portfolio state...")
    portfolio.persist()
    log.info("Trading system stopped")


if __name__ == "__main__":
    main()
