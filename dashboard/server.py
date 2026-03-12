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

app = FastAPI(title="Trading Dashboard", docs_url=None, redoc_url=None, openapi_url=None)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(DASHBOARD_DIR, "static")),
    name="static",
)

# ── State references (set by start_dashboard) ─────────────────────────

_portfolio: Any = None
_heartbeat: Any = None
_cost_tracker: Any = None
_pipeline: Any = None


def _set_refs(portfolio: Any, heartbeat: Any, cost_tracker: Any, pipeline: Any = None) -> None:
    global _portfolio, _heartbeat, _cost_tracker, _pipeline
    _portfolio = portfolio
    _heartbeat = heartbeat
    _cost_tracker = cost_tracker
    _pipeline = pipeline


# ── REST endpoints — initial data load ─────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = os.path.join(DASHBOARD_DIR, "static", "index.html")
    with open(html_path) as f:
        content = f.read().replace("{{GA4_MEASUREMENT_ID}}", _ga4_id())
    return HTMLResponse(content=content)


@app.get("/v3", response_class=HTMLResponse)
async def dashboard_v3() -> HTMLResponse:
    html_path = os.path.join(DASHBOARD_DIR, "static", "v3.html")
    with open(html_path) as f:
        content = f.read()
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


@app.get("/api/signal-accuracy")
async def get_signal_accuracy() -> dict[str, Any]:
    data = _read_json(os.path.join(DATA_DIR, "signal_accuracy.json"), [])
    if isinstance(data, list):
        # Compute summary from raw signal list
        total = len(data)
        executed = [s for s in data if s.get("pipeline_outcome") == "executed"]
        closed = [s for s in data if s.get("exit_price") is not None]
        wins = [s for s in closed if s.get("signal_correct") is True]
        losses = [s for s in closed if s.get("signal_correct") is False]
        closed_count = len(closed)
        return {
            "total_signals": total,
            "executed": len(executed),
            "closed": closed_count,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / closed_count * 100, 1) if closed_count > 0 else 0.0,
            "recent": data[-10:] if data else [],
        }
    return {"total_signals": 0, "executed": 0, "closed": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "recent": []}


@app.get("/api/signal-funnel")
async def get_signal_funnel() -> dict[str, Any]:
    """Return signal pipeline funnel stats."""
    if _pipeline and hasattr(_pipeline, "get_funnel_stats"):
        return _pipeline.get_funnel_stats()
    return {
        "signals_generated": 0, "pre_filtered": 0, "analyst_no_trade": 0,
        "analyst_errors": 0, "devil_killed": 0, "risk_rejected": 0, "executed": 0,
        "last_reset": "", "past_prefilter": 0, "past_analyst": 0, "past_devil": 0,
        "past_risk": 0, "prefilter_pass_rate": 0.0, "analyst_pass_rate": 0.0,
        "devil_pass_rate": 0.0, "risk_pass_rate": 0.0, "execution_rate": 0.0,
    }


@app.get("/api/earnings")
async def get_earnings() -> list[dict[str, Any]]:
    from core.earnings_calendar import EarningsCalendar
    cal = EarningsCalendar()
    return cal.upcoming_earnings(days=30)


@app.get("/api/regime")
async def get_regime() -> dict[str, Any]:
    """Return current market regime classification for all tradeable assets."""
    try:
        return await asyncio.to_thread(_fetch_regime)
    except Exception as e:
        log.warning("Regime fetch failed: %s", e)
        return {"per_asset": {}, "dominant_regime": "RANGING", "regime_agreement": 0.0}


@app.get("/api/stop-recommendations")
async def get_stop_recommendations() -> dict[str, Any]:
    """Return adaptive stop-loss/take-profit recommendations from trade history."""
    from core.adaptive_stops import AdaptiveStopOptimizer
    opt = AdaptiveStopOptimizer()
    return opt.analyze()


@app.get("/api/session-analysis")
async def get_session_analysis() -> dict[str, Any]:
    """Return trading session performance analysis."""
    from core.session_analyzer import SessionAnalyzer
    analyzer = SessionAnalyzer()
    return analyzer.analyze()


@app.get("/api/confidence-calibration")
async def get_confidence_calibration() -> dict[str, Any]:
    from core.confidence_calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    return cal.analyze()


@app.get("/api/phantom-analysis")
async def get_phantom_analysis() -> dict[str, Any]:
    from core.phantom_analyzer import PhantomAnalyzer
    analyzer = PhantomAnalyzer()
    return analyzer.analyze()


@app.get("/api/learning-insights")
async def get_learning_insights() -> dict[str, Any]:
    """Return learning system insights from the SelfOptimizer."""
    from core.self_optimizer import SelfOptimizer
    try:
        optimizer = SelfOptimizer()
        return optimizer.gather_learning_insights()
    except Exception as e:
        log.warning("Learning insights fetch failed: %s", e)
        return {
            "stop_recommendations": None,
            "confidence_calibration": None,
            "phantom_analysis": None,
            "session_analysis": None,
            "actionable_insights": [],
        }


@app.get("/api/regime-strategy")
async def get_regime_strategy() -> dict[str, Any]:
    """Return current regime strategy presets and active regime per asset."""
    from core.regime_strategy import RegimeStrategySelector
    try:
        selector = RegimeStrategySelector()
        result: dict[str, Any] = {"presets": selector.presets, "per_asset": {}}
        regime_data = await asyncio.to_thread(_fetch_regime)
        per_asset_regimes = regime_data.get("per_asset", {})
        for asset, info in per_asset_regimes.items():
            regime = info.get("regime", "LOW_VOLATILITY") if isinstance(info, dict) else str(info)
            result["per_asset"][asset] = {
                "regime": regime,
                "adjustments": selector.get_adjustments(regime),
            }
        return result
    except Exception as e:
        log.warning("Regime strategy fetch failed: %s", e)
        return {"presets": {}, "per_asset": {}}


@app.get("/api/kelly-stats")
async def kelly_stats() -> dict[str, Any]:
    """Return Kelly Criterion position sizing stats from trade journal."""
    from core.kelly_sizer import KellySizer
    try:
        journal_path = os.path.join(DATA_DIR, "trade_journal.json")
        risk_params = _read_json(os.path.join(CONFIG_DIR, "risk_params.json"), {})
        sizer = KellySizer(journal_file=journal_path, config=risk_params)
        return sizer.get_all_stats()
    except Exception as e:
        log.warning("Kelly stats fetch failed: %s", e)
        return {"global": {}, "per_asset": {}, "per_sector": {}, "config": {}}


@app.get("/api/postmortems")
async def get_postmortems(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent post-mortem findings for losing trades."""
    from core.postmortem import PostMortemEngine
    try:
        engine = PostMortemEngine()
        return engine.get_recent_findings(limit=limit)
    except Exception as e:
        log.warning("Post-mortem findings fetch failed: %s", e)
        return []


@app.get("/api/prevention-rules")
async def get_prevention_rules() -> list[dict[str, Any]]:
    """Return active prevention rules from post-mortem analysis."""
    rules_path = os.path.join(DATA_DIR, "prevention_rules.json")
    data = _read_json(rules_path, [])
    if isinstance(data, list):
        return [r for r in data if r.get("active", True)]
    return []


@app.get("/api/watchlist")
async def get_watchlist() -> dict[str, Any]:
    """Return dynamic watchlist: only assets we're actively trading (held positions).

    This powers the dynamic market panel — shows only what you're in.
    Empty when no positions, grows as trades are taken.
    """
    try:
        from core.asset_registry import get_registry
        registry = get_registry()

        # Only held positions — nothing else
        held: list[str] = []
        if _portfolio:
            snap = _portfolio.snapshot()
            held = list({pos.get("asset", "") for pos in snap.get("open_positions", []) if pos.get("asset")})

        # Sort: dynamic discovered assets last, otherwise alphabetical
        held.sort(key=lambda s: (not registry.is_core(s), s))

        # Annotate each asset
        watchlist: list[dict[str, Any]] = []
        for sym in held:
            config = registry.get_config(sym)
            watchlist.append({
                "symbol": sym,
                "is_core": registry.is_core(sym),
                "is_held": True,
                "is_dynamic": registry.is_dynamic(sym),
                "type": config.get("type", "stock"),
                "sector": config.get("sector", ""),
            })

        return {
            "assets": watchlist,
            "held_count": len(held),
            "dynamic_count": len([w for w in watchlist if w["is_dynamic"]]),
            "total": len(watchlist),
        }
    except Exception as e:
        log.warning("Watchlist fetch failed: %s", e)
        return {"assets": [], "held_count": 0, "dynamic_count": 0, "total": 0}


@app.get("/ping")
@app.head("/ping")
async def ping() -> dict[str, str]:
    """Lightweight health check for uptime monitors (UptimeRobot, etc.)."""
    return {"status": "ok"}


@app.get("/api/events/recent")
async def get_recent_events() -> list[dict[str, Any]]:
    return event_bus.get_recent(50)


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("risk_params", "assets", "news_scout_params", "market_analyst_params", "devils_advocate_params", "chart_analyst_params", "chart_scanner_params"):
        path = os.path.join(CONFIG_DIR, f"{name}.json")
        data = _read_json(path, None)
        if data is not None:
            result[name] = data
    # Include dynamic assets if any exist
    dynamic_data = _read_json(os.path.join(DATA_DIR, "dynamic_assets.json"), None)
    if dynamic_data:
        result["dynamic_assets"] = dynamic_data
    return result


# ── WebSocket — live event streaming ───────────────────────────────────

_MAX_WS_CONNECTIONS = 20
_ws_count = 0
_ws_lock = threading.Lock()

_ALLOWED_ORIGINS: set[str] = set()


def _allowed_ws_origin(origin: str | None) -> bool:
    """Check if WebSocket origin is allowed."""
    if not _ALLOWED_ORIGINS:
        return True  # No restriction configured
    if not origin:
        return False
    return any(origin.startswith(a) for a in _ALLOWED_ORIGINS)


def _init_allowed_origins() -> None:
    """Build allowed origins from DASHBOARD_ALLOWED_ORIGINS env var."""
    raw = os.getenv("DASHBOARD_ALLOWED_ORIGINS", "")
    if raw:
        _ALLOWED_ORIGINS.update(o.strip() for o in raw.split(",") if o.strip())


_init_allowed_origins()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global _ws_count

    # Origin check
    origin = ws.headers.get("origin")
    if not _allowed_ws_origin(origin):
        await ws.close(code=4003, reason="Origin not allowed")
        return

    # Connection limit
    with _ws_lock:
        if _ws_count >= _MAX_WS_CONNECTIONS:
            await ws.close(code=4029, reason="Too many connections")
            return
        _ws_count += 1

    await ws.accept()
    queue = event_bus.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                # No events for 30s — send ping to keep connection alive
                await ws.send_json({"category": "system", "event_type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("WebSocket error: %s", e)
    finally:
        event_bus.unsubscribe(queue)
        with _ws_lock:
            _ws_count -= 1


# ── Helpers ────────────────────────────────────────────────────────────

def _read_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _get_active_symbols() -> list[str]:
    """Get symbols to fetch data for: only held positions."""
    try:
        if _portfolio:
            snap = _portfolio.snapshot()
            symbols = list({pos.get("asset", "") for pos in snap.get("open_positions", []) if pos.get("asset")})
            if symbols:
                return sorted(symbols)
        # Fallback: core assets if no portfolio loaded yet
        from core.asset_registry import get_core_assets
        return list(get_core_assets())
    except Exception:
        return ["BTC", "ETH", "GLDM", "SLV"]


def _fetch_regime() -> dict[str, Any]:
    from core.regime_classifier import RegimeClassifier
    from tools.market_data import MarketDataFetcher
    symbols = _get_active_symbols()
    classifier = RegimeClassifier(market_data_fetcher=MarketDataFetcher())
    return classifier.classify_portfolio(symbols)


def _fetch_price_history() -> dict[str, Any]:
    from tools.market_data import MarketDataFetcher
    symbols = _get_active_symbols()
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
    symbols = _get_active_symbols()
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
    host: str = "127.0.0.1",
    port: int = 8080,
    pipeline: Any = None,
) -> threading.Thread:
    """Start the dashboard server in a background daemon thread."""
    _set_refs(portfolio, heartbeat, cost_tracker, pipeline=pipeline)

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
