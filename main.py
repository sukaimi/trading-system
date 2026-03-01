"""Trading System — Main entry point.

Loads environment, initializes all Tier 0 modules and the intelligence
pipeline, registers schedules, and runs the event loop.
Exits cleanly on SIGINT/SIGTERM.
"""

import signal
import sys
import time

import schedule
from dotenv import load_dotenv

from core.executor import Executor
from core.heartbeat import Heartbeat
from core.llm_client import LLMClient
from core.logger import setup_logger
from core.pipeline import TradingPipeline
from core.portfolio import PortfolioState
from core.risk_manager import RiskManager
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

    # 5. Initialize trading pipeline
    pipeline = TradingPipeline(
        portfolio=portfolio,
        risk_manager=risk_manager,
        executor=executor,
        telegram=telegram,
        llm_client=llm_client,
    )
    log.info("Trading pipeline initialized")

    # 6. Initialize heartbeat
    heartbeat = Heartbeat(telegram_notifier=telegram)
    log.info("Heartbeat monitor initialized")

    # 7. Register schedules
    def run_heartbeat():
        status = heartbeat.check()
        if not status.all_healthy:
            log.warning("Heartbeat reported failures: %s", status.failures)

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

    schedule.every(5).minutes.do(run_heartbeat)
    schedule.every(30).minutes.do(run_news_scan)
    schedule.every().day.at("00:00").do(run_asian_open)       # 08:00 SGT
    schedule.every().day.at("08:00").do(run_european_overlap)  # 16:00 SGT
    schedule.every().day.at("14:00").do(run_us_close)          # 22:00 SGT
    log.info("All schedules registered")

    # 8. Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 9. Run initial checks
    log.info("Running initial heartbeat check...")
    run_heartbeat()

    log.info("Running initial news scan...")
    run_news_scan()

    # 10. Event loop
    log.info("Trading system started — entering main loop")
    try:
        while _running:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 11. Cleanup
    log.info("Persisting portfolio state...")
    portfolio.persist()
    log.info("Trading system stopped")


if __name__ == "__main__":
    main()
