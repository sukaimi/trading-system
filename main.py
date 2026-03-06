"""Trading System — Main entry point.

Loads environment, initializes all Tier 0-3 modules and the intelligence
pipeline, registers all schedules (heartbeat, news scan, analysis sessions,
daily summary, weekly review, circuit breaker), and runs the event loop.
Exits cleanly on SIGINT/SIGTERM.
"""

import os
import signal
import sys
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
from tools.telegram_bot import TelegramNotifier

log = setup_logger("trading.main")

# Global flag for graceful shutdown
_running = True


def _shutdown(signum, frame):
    global _running
    sig_name = signal.Signals(signum).name
    log.info("Received %s — shutting down gracefully", sig_name)
    _running = False


def main():
    global _running

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
    optimizer = SelfOptimizer(telegram=telegram, portfolio=portfolio)
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
    )
    log.info("Trading pipeline initialized")

    # Log broker account status on startup (does not override internal equity)
    pipeline.sync_portfolio_with_broker()

    # 7. Initialize heartbeat (skip IBKR check in paper mode)
    heartbeat = Heartbeat(telegram_notifier=telegram, skip_ibkr=(executor_mode != "ibkr"))
    log.info("Heartbeat monitor initialized")

    # 8. Task wrapper for dashboard event emission
    def _emit_task(name, func):
        def wrapper():
            event_bus.emit("scheduler", "task_run", {"task_name": name})
            t0 = time.time()
            try:
                func()
            finally:
                event_bus.emit("scheduler", "task_complete", {
                    "task_name": name, "duration_sec": round(time.time() - t0, 2),
                })
        wrapper.__name__ = name
        return wrapper

    # 9. Define scheduled tasks
    def run_heartbeat():
        status = heartbeat.check()
        if not status.all_healthy:
            log.warning("Heartbeat reported failures: %s", status.failures)
        # Run stop-loss + take-profit checks (Tier 0 — deterministic, every 5 min)
        pipeline.check_stop_losses()
        pipeline.check_take_profits()
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
    schedule.every(30).minutes.do(_emit_task("news_scan", run_news_scan))
    schedule.every(2).hours.do(_emit_task("chart_scan", run_chart_scan))
    schedule.every().day.at("00:00").do(_emit_task("asian_open", run_asian_open))           # 08:00 SGT
    schedule.every().day.at("08:00").do(_emit_task("european_overlap", run_european_overlap))  # 16:00 SGT
    schedule.every().day.at("14:00").do(_emit_task("us_close", run_us_close))               # 22:00 SGT
    schedule.every().day.at("15:00").do(_emit_task("daily_summary", run_daily_summary))     # 23:00 SGT
    schedule.every().day.at("16:00").do(_emit_task("daily_reset", run_daily_reset))         # 00:00 SGT (next day)
    schedule.every().sunday.at("15:00").do(_emit_task("weekly_review", run_weekly_review))  # 23:00 SGT Sunday
    log.info("All schedules registered (9 total)")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 11. Start GA4 tracker (server-side analytics)
    from core.ga4_tracker import ga4_tracker
    ga4_tracker.start()

    # 12. Start dashboard server
    dashboard_port = int(os.getenv("DASHBOARD_PORT", "8080"))
    dashboard_host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    start_dashboard(portfolio, heartbeat, cost_tracker, host=dashboard_host, port=dashboard_port)

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
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 16. Cleanup
    log.info("Persisting portfolio state...")
    portfolio.persist()
    log.info("Trading system stopped")


if __name__ == "__main__":
    main()
