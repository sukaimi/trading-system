"""FastAPI dashboard server — runs in a background thread alongside the scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.event_bus import event_bus
from core.logger import setup_logger

log = setup_logger("trading.dashboard")

DASHBOARD_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(DASHBOARD_DIR), "data")
CONFIG_DIR = os.path.join(os.path.dirname(DASHBOARD_DIR), "config")

def _ga4_id() -> str:
    return os.getenv("GA4_MEASUREMENT_ID", "")

app = FastAPI(title="Trading Dashboard", docs_url=None, redoc_url=None)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(DASHBOARD_DIR, "static")),
    name="static",
)

# ── State references (set by start_dashboard) ─────────────────────────

_portfolio: Any = None
_heartbeat: Any = None
_cost_tracker: Any = None


def _set_refs(portfolio: Any, heartbeat: Any, cost_tracker: Any) -> None:
    global _portfolio, _heartbeat, _cost_tracker
    _portfolio = portfolio
    _heartbeat = heartbeat
    _cost_tracker = cost_tracker


# ── REST endpoints — initial data load ─────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = os.path.join(DASHBOARD_DIR, "static", "index.html")
    with open(html_path) as f:
        content = f.read().replace("{{GA4_MEASUREMENT_ID}}", _ga4_id())
    return HTMLResponse(content=content)


@app.get("/agents", response_class=HTMLResponse)
async def agent_floor() -> HTMLResponse:
    html_path = os.path.join(DASHBOARD_DIR, "static", "agent-floor.html")
    with open(html_path) as f:
        content = f.read().replace("{{GA4_MEASUREMENT_ID}}", _ga4_id())
    return HTMLResponse(content=content)


@app.get("/api/portfolio")
async def get_portfolio() -> dict[str, Any]:
    if _portfolio:
        return _portfolio.snapshot()
    return _read_json(os.path.join(DATA_DIR, "portfolio_state.json"), {})


@app.get("/api/journal")
async def get_journal() -> list[Any] | dict[str, Any]:
    return _read_json(os.path.join(DATA_DIR, "trade_journal.json"), [])


@app.get("/api/heartbeat")
async def get_heartbeat() -> dict[str, Any]:
    if _heartbeat:
        try:
            status = _heartbeat.check()
            return status.model_dump(mode="json") if hasattr(status, "model_dump") else status
        except Exception:
            pass
    return {"all_healthy": None, "checks": {}, "failures": []}


@app.get("/api/circuit-breaker")
async def get_circuit_breaker() -> list[Any] | dict[str, Any]:
    return _read_json(os.path.join(DATA_DIR, "circuit_breaker_log.json"), [])


@app.get("/api/costs")
async def get_costs() -> dict[str, Any]:
    if _cost_tracker:
        return _cost_tracker.summary()
    return {"total_usd": 0, "by_provider": {}, "by_agent": {}, "call_count": 0, "recent_calls": []}


@app.get("/api/scheduler")
async def get_scheduler() -> list[dict[str, Any]]:
    try:
        import schedule as sched_lib
        tasks = []
        for job in sched_lib.get_jobs():
            name = getattr(job.job_func, "__name__", str(job.job_func))
            at_str = None
            if job.at_time:
                at_str = job.at_time.strftime("%H:%M")
            tasks.append({
                "name": name,
                "interval": job.interval,
                "unit": job.unit,
                "at_time": at_str,
                "start_day": getattr(job, "start_day", None),
                "next_run": (job.next_run.isoformat() + "Z") if job.next_run else None,
            })
        return tasks
    except Exception:
        return []


@app.get("/api/market")
async def get_market() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_fetch_market_prices)
    except Exception as e:
        log.warning("Market data fetch failed: %s", e)
        return {}


@app.get("/api/price-history")
async def get_price_history() -> dict[str, Any]:
    """Return last 24h hourly close prices for sparklines."""
    try:
        return await asyncio.to_thread(_fetch_price_history)
    except Exception as e:
        log.warning("Price history fetch failed: %s", e)
        return {}


@app.get("/api/equity-history")
async def get_equity_history() -> list[Any] | dict[str, Any]:
    """Return equity snapshots over time for the equity curve chart."""
    return _read_json(os.path.join(DATA_DIR, "equity_history.json"), [])


@app.get("/api/phantom")
async def get_phantom() -> dict[str, Any]:
    return _read_json(os.path.join(DATA_DIR, "phantom_trades.json"), {
        "total_missed": 0, "recent": [],
    })


@app.get("/api/events/recent")
async def get_recent_events() -> list[dict[str, Any]]:
    return event_bus.get_recent(50)


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("risk_params", "assets", "news_scout_params", "market_analyst_params", "devils_advocate_params", "chart_analyst_params"):
        path = os.path.join(CONFIG_DIR, f"{name}.json")
        data = _read_json(path, None)
        if data is not None:
            result[name] = data
    return result


# ── WebSocket — live event streaming ───────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    queue = event_bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("WebSocket error: %s", e)
    finally:
        event_bus.unsubscribe(queue)


# ── Helpers ────────────────────────────────────────────────────────────

def _read_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _fetch_price_history() -> dict[str, Any]:
    from tools.market_data import MarketDataFetcher
    try:
        from core.asset_registry import get_tradeable_assets
        symbols = get_tradeable_assets()
    except Exception:
        symbols = ["BTC", "ETH", "GLDM", "SLV"]
    mdf = MarketDataFetcher()
    result: dict[str, Any] = {}
    for symbol in symbols:
        try:
            ohlcv = mdf.get_ohlcv(symbol, period="5d", interval="1h")
            recent = ohlcv[-24:] if len(ohlcv) > 24 else ohlcv
            result[symbol] = [bar["close"] for bar in recent]
        except Exception:
            result[symbol] = []
    return result


def _fetch_market_prices() -> dict[str, Any]:
    from tools.market_data import MarketDataFetcher
    try:
        from core.asset_registry import get_tradeable_assets
        symbols = get_tradeable_assets()
    except Exception:
        symbols = ["BTC", "ETH", "GLDM", "SLV"]
    mdf = MarketDataFetcher()
    prices: dict[str, Any] = {}
    for symbol in symbols:
        try:
            prices[symbol] = mdf.get_price(symbol)
        except Exception:
            prices[symbol] = None
    return prices


# ── Background thread launcher ─────────────────────────────────────────

def start_dashboard(
    portfolio: Any,
    heartbeat: Any,
    cost_tracker: Any,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> threading.Thread:
    """Start the dashboard server in a background daemon thread."""
    _set_refs(portfolio, heartbeat, cost_tracker)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        event_bus.set_loop(loop)

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, name="dashboard", daemon=True)
    thread.start()
    log.info("Dashboard started on http://%s:%d", host, port)
    return thread
