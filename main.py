"""Trading System — Main entry point.

Loads environment, initializes all Tier 0-3 modules and the intelligence
pipeline, registers all schedules (heartbeat, news scan, analysis sessions,
daily summary, weekly review, circuit breaker), and runs the event loop.
Exits cleanly on SIGINT/SIGTERM.
"""

import signal
import sys
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv

from agents.trade_journal import TradeJournal
from agents.weekly_strategist import WeeklyStrategist
from core.executor import Executor
from core.heartbeat import Heartbeat
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
from core.self_optimizer import SelfOptimizer
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

    executor = Executor()
    log.info("Executor initialized (paper_mode=%s)", executor.paper_mode)

    telegram = TelegramNotifier()

    # 4. Initialize LLM client
    llm_client = LLMClient()
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

    # 7. Initialize heartbeat
    heartbeat = Heartbeat(telegram_notifier=telegram)
    log.info("Heartbeat monitor initialized")

    # 8. Define scheduled tasks
    def run_heartbeat():
        status = heartbeat.check()
        if not status.all_healthy:
            log.warning("Heartbeat reported failures: %s", status.failures)
        # Run circuit breaker check alongside heartbeat
        pipeline.run_circuit_breaker_check()

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

    # 9. Register all schedules
    schedule.every(5).minutes.do(run_heartbeat)
    schedule.every(30).minutes.do(run_news_scan)
    schedule.every().day.at("00:00").do(run_asian_open)         # 08:00 SGT
    schedule.every().day.at("08:00").do(run_european_overlap)    # 16:00 SGT
    schedule.every().day.at("14:00").do(run_us_close)            # 22:00 SGT
    schedule.every().day.at("15:00").do(run_daily_summary)       # 23:00 SGT
    schedule.every().day.at("16:00").do(run_daily_reset)         # 00:00 SGT (next day)
    schedule.every().sunday.at("15:00").do(run_weekly_review)    # 23:00 SGT Sunday
    log.info("All schedules registered (8 total)")

    # 10. Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 11. Run initial checks
    log.info("Running initial heartbeat check...")
    run_heartbeat()

    log.info("Running initial news scan...")
    run_news_scan()

    # 12. Event loop
    log.info("Trading system started — entering main loop")
    try:
        while _running:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 13. Cleanup
    log.info("Persisting portfolio state...")
    portfolio.persist()
    log.info("Trading system stopped")


if __name__ == "__main__":
    main()
