# CLAUDE.md — Trading System Project Guide

## Project Overview

Autonomous multi-agent AI trading system with **open universe** trading — any US-listed stock/ETF with >$1B market cap is tradeable. Core 14 assets: BTC, ETH, GLDM (gold), SLV (silver), AAPL, NVDA, TSLA, AMZN, SPY, META, TLT (bonds), XLE (energy), EWS (Singapore), FXI (China). NewsScout can discover opportunities in ANY stock. Uses a 4-tier intelligence stack.

**Owner**: Sukaimi (Code&Canvas)
**Stack**: Python 3.12 / Direct LLM API calls / Ubuntu 24.04 VPS
**Executor**: Multi-broker — Alpaca, IBKR, Tiger Brokers, Coinbase (per-tenant via EXECUTOR_MODE)
**PRD**: `docs/TRADING_AGENT_PRD.md`
**Dashboard**: `https://tradebot.codeandcraft.ai` — Agent Trading Floor + trading dashboard
**Repo**: `https://github.com/sukaimi/trading-system`
**Memory Vault**: `/home/trader/trading-memory/` on VPS — Obsidian-style long-term memory (decisions, incidents, trades, learnings). Git-tracked. See vault's `CLAUDE.md` for structure and conventions.

## Architecture

### 4-Tier Intelligence Stack
| Tier | Model | Cost | Modules |
|------|-------|------|---------|
| 0 | Pure Python | $0 | RiskManager, Executor, Heartbeat, Portfolio, StopLossMonitor |
| 1 | DeepSeek V3.2 | ~$0.08/mo | NewsScout, TradeJournal, MarketAnalyst (escalated), DevilsAdvocate |
| 2 | Kimi K2.5 | (disabled) | — (kept as fallback, not actively used) |
| 3 | Claude Sonnet 4.6 | ~$0.50/mo | WeeklyStrategist, CircuitBreaker, SelfOptimizer |

### Decision Flow
```
NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal
Heartbeat (5 min): health check → stop-loss → take-profit → holding period → circuit breaker
News scan (15 min): full pipeline
Chart scan (4x daily): 00:55, 06:55, 12:55, 18:55 UTC
3 daily sessions: Asian Open (00:00 UTC), European Overlap (08:00), US Close (14:00)
Proactive scan (3x daily): 01:00, 09:00, 15:00 UTC (= 09:00, 17:00, 23:00 SGT)
Weekly: Strategist + SelfOptimizer → rewrite agent params
Emergency: CircuitBreaker → halt trading → auto-recovery after 6h cooldown
```

## Project Structure
```
trading-system/
├── agents/          # 6 agent modules
├── core/            # 29 core modules (schemas, llm_client, pipeline, portfolio, risk_manager,
│                    #   executor, routing_executor, coinbase_executor, alpaca_executor,
│                    #   tiger_executor, paper_executor, broker_sync, heartbeat, logger,
│                    #   self_optimizer, cost_tracker, event_bus, ga4_tracker, asset_registry,
│                    #   phantom_tracker, signal_tracker, regime_classifier, earnings_calendar,
│                    #   adaptive_stops, session_analyzer, confidence_calibrator,
│                    #   phantom_analyzer, regime_strategy)
├── tools/           # 5 tools (news_fetcher, market_data, technical_indicators, correlation, telegram_bot)
├── dashboard/       # FastAPI dashboard server + static files (index.html, agent-floor.html, lotus-creature.js)
├── config/          # Dynamic params JSON (updated weekly by SelfOptimizer)
├── data/            # Persisted state, trade journal, logs, weekly reviews (gitignored)
├── tests/           # 40 test files, 676 tests (pytest)
├── docs/            # PRD, Lotus spec
├── main.py          # Entry point — 12-task scheduler + dashboard
└── requirements.txt
```

## Key Files
- `main.py` — Entry point, scheduler, executor selection (paper/alpaca/ibkr/tiger/live)
- `core/pipeline.py` — Signal-to-execution orchestration + `update_trailing_stops()` + `check_stop_losses()` + `check_holding_periods()`
- `core/regime_classifier.py` — Market regime detection (trending/ranging/volatile) using ADX, ATR, Bollinger, RSI
- `core/earnings_calendar.py` — Earnings date tracking, position reduction near earnings
- `core/signal_tracker.py` — Signal accuracy tracking (signal → trade → outcome correlation)
- `core/alpaca_executor.py` — Alpaca paper/live trading executor
- `core/coinbase_executor.py` — Coinbase Advanced Trade API crypto executor (BTC, ETH)
- `core/routing_executor.py` — Smart routing: crypto→Coinbase, stocks/ETFs→Alpaca, SGX→IBKR (EXECUTOR_MODE=live)
- `core/tiger_executor.py` — Tiger Brokers executor for SGX/US/HK stocks (tigeropen SDK)
- `core/asset_registry.py` — Dynamic asset registry: core 14 from `config/assets.json` + open universe resolution via yfinance (quality gates: >$1B cap, >500K vol, no OTC/SPAC). Cache at `data/dynamic_assets.json`
- `core/broker_sync.py` — Broker reconciliation engine: detect ghosts, orphans, qty mismatches
- `core/cost_tracker.py` — LLM cost tracking with JSON persistence + daily budget limits (`check_budget()`)
- `core/event_bus.py` — Real-time pub/sub for dashboard WebSocket + sync listeners
- `core/ga4_tracker.py` — GA4 Measurement Protocol server-side event tracking
- `core/portfolio.py` — Thread-safe portfolio state with JSON persistence + MAE/MFE tracking + position weight drift monitoring
- `dashboard/server.py` — FastAPI server (REST + WebSocket)
- `dashboard/static/index.html` — Dashboard UI + Agent Trading Floor (two-view toggle, Lotus disabled)
- `dashboard/static/agent-floor.html` — Isometric Agent Trading Floor with live data (served at `/agents`)
- `dashboard/static/lotus-creature.js` — Animated canvas creature engine (6 stages, currently disabled)
- `core/risk_manager.py` — Position limits, daily loss caps, drawdown, sector concentration (max 3 per sector)
- `config/risk_params.json` — Risk limits (max position 7%, daily loss 5%, drawdown 15%, stop-loss 3%, take-profit 5%, correlation 0.65, min R:R 2:1)

## Development Commands
```bash
source venv/bin/activate
pytest tests/ -v                        # Run all 302 tests
pytest tests/test_stop_loss_monitor.py -v  # Run specific test file
python main.py                          # Start full system (scheduler + dashboard on :8080)
```

## VPS Deployment
```
VPS: Ubuntu 24.04 LTS
URL: https://tradebot.codeandcraft.ai
SSH: ssh trader@VPS_IP  (root login disabled, key-only auth)
Services:
  - trading-system.service — main trading system (runs as trader, port 8080)
  - nginx — reverse proxy (80/443 → localhost:8080, SSL via Let's Encrypt)
  - webhook.service — GitHub auto-deploy listener on port 9000
Auto-deploy: git push → GitHub webhook → VPS pulls + restarts
Deploy script: /opt/deploy.sh
SSL: Let's Encrypt, auto-renews via certbot timer
Firewall (UFW): ports 22 (SSH), 443 (HTTPS), 9000 (GitHub IPs only)
Fail2ban: active on SSH
```

### Security Hardening
- **Port 8080 closed**: Uvicorn binds to `127.0.0.1` only, UFW blocks external access. All traffic goes through nginx.
- **Port 9000 restricted**: Webhook port only accepts connections from GitHub IP ranges (`140.82.112.0/20`, `185.199.108.0/22`, `192.30.252.0/22`, `143.55.64.0/20`)
- **HTTP Basic Auth**: Dashboard and private API behind nginx Basic Auth (`/etc/nginx/.htpasswd`, username: `trader`). Public routes (no auth): `/agents`, `/static/`, `/ws`, `/api/(market|price-history|equity-history|portfolio|events/recent)`
- **Security headers**: HSTS, X-Frame-Options (SAMEORIGIN), X-Content-Type-Options, CSP, Referrer-Policy, Permissions-Policy
- **Server tokens**: `server_tokens off` — nginx version not disclosed
- **OpenAPI schema**: Disabled (`openapi_url=None`) — `/openapi.json` returns 404
- **WebSocket**: Max 20 concurrent connections, origin validation via `DASHBOARD_ALLOWED_ORIGINS` env var
- **HTTP/2**: Enabled in nginx (`listen 443 ssl http2`)
- **Gzip**: Enabled for JSON, JS, CSS, XML responses
- **Hardening script**: `scripts/harden-vps.sh` — run on VPS to apply all nginx/UFW changes

### Manual VPS commands
```bash
ssh trader@VPS_IP
sudo systemctl status trading-system    # Check status
sudo systemctl restart trading-system   # Restart
journalctl -u trading-system -f         # Live logs
bash scripts/harden-vps.sh             # Apply security hardening
sudo htpasswd /etc/nginx/.htpasswd trader  # Reset dashboard password
```

## Environment Variables
All API keys in `.env` (never committed). See `.env.example` for template.
Required: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Required: EXECUTOR_MODE (paper/alpaca/ibkr/tiger/live)
Required (alpaca): ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
Required (ibkr): IBKR_HOST, IBKR_PAPER_PORT, IBKR_CLIENT_ID, IBKR_PAPER_MODE
Required (tiger): TIGER_ID, TIGER_PRIVATE_KEY, TIGER_ACCOUNT, TIGER_PAPER_MODE
Required (live mode): COINBASE_API_KEY, COINBASE_API_SECRET (+ Alpaca keys)
Optional: BROKER_SYNC_AUTO_FIX (true/false — auto-fix portfolio-broker discrepancies)
Optional: KIMI_API_KEY (disabled), GOOGLE_API_KEY (fallback), ALPHA_VANTAGE_API_KEY, CRYPTOCOMPARE_API_KEY
Optional: DASHBOARD_PORT (default 8080), DASHBOARD_ALLOWED_ORIGINS, GA4_MEASUREMENT_ID, GA4_API_SECRET

## Conventions
- All inter-agent communication uses strict JSON (Pydantic v2 validated)
- Tier 0 modules are pure Python — no LLM calls
- LLM client supports mock mode for testing (`MOCK_LLM=true`)
- Config files in `config/` are dynamic — updated weekly by SelfOptimizer
- Data files in `data/` are gitignored (private trade data)
- All state persisted to JSON for crash recovery
- Tests use pytest with mocked LLM responses
- Dashboard serves index.html from `/` route (not static mount), so script paths must be absolute (`/static/file.js`)

## Implementation Status

### Completed
- [x] **Phase 1: Foundation** — Project scaffolding, Tier 0 modules, schemas, tests
- [x] **Phase 2: Intelligence Layer** — LLM client, all agents, pipeline, tools
- [x] **Phase 3: Tier 3 Intelligence** — Weekly Strategist, Circuit Breaker, Self-Optimizer
- [x] **Phase 4: Paper Trading** — Alpaca executor, stop-loss monitor, dashboard, Lotus creature, VPS deployment
- [x] **Agent Trading Floor** — Isometric visualization with 9 agents + 4 NPC interns, live data from APIs + WebSocket

### Current Phase: Paper Trading + Agent Upskilling
- Running on VPS with Alpaca paper account
- Monitoring trades, agent decisions, costs
- Homepage: Agent Trading Floor (isometric, live market data, real-time agent animations)
- Acceleration features: 15-min news scan, 4x chart scan, 72h forced exit, MAE/MFE tracking, sector concentration, take-profit floor (5%), position weight drift monitoring
- **Tier 1 upgrades**: ATR dynamic stops/TP, ATR position sizing, pre-LLM filter, loss throttling, phantom→DA feedback, signal accuracy→NewsScout feedback, correlation→DA feedback
- **Tier 2 upgrades**: Trailing stops, regime classifier (5 regimes), adaptive stops, correlation-based position limits, earnings calendar
- **Tier 3 learning systems**: Adaptive stop optimizer, session analyzer, confidence calibrator, phantom analyzer, regime strategy selector, enhanced SelfOptimizer with data-driven directives
- **Roadmap features**: Proactive scan (3x/day), regime holding periods, reflexivity detection, principle extraction, kill switch, signal funnel, R:R enforcement (min 2:1), TP floor
- **Gridlock fix (v52)**: Duplicate asset soft flag, pre-filter removed, correlation 0.50→0.65
- **Open Universe (2026-03-11)**: Any US-listed stock with >$1B cap tradeable. NewsScout LLM prompt unlocked. Dynamic asset resolution via yfinance with quality gates. Disk cache at `data/dynamic_assets.json`. Risk manager uses dynamic sector lookup. Dashboard market panel shows only held positions (dynamic, scrollable). Agent floor ticker board auto-scrolls with pagination.
- **Broker equity sync (2026-03-11)**: Internal equity now syncs to Alpaca's reported number every 5 min via `sync_portfolio_with_broker()`. Broker is source of truth.
- **New endpoint**: `/api/watchlist` — returns held positions only for dynamic market panel

### Pending
- [ ] **Phase 5: Optimization** — Analyze paper trading data, tune parameters (needs 10-20+ trades)
- [ ] **Phase 6: Micro Live** — $100 Alpaca (stocks/ETFs) + $100 Coinbase (BTC/ETH), EXECUTOR_MODE=live with RoutingExecutor
- [ ] **Phase 7: Scale** — Increase capital if profitable
- [x] Set up GitHub webhook for auto-deploy (payload URL: http://187.77.132.195:9000, secret: trading-system-deploy)
- [x] HTTPS via nginx + Let's Encrypt on tradebot.codeandcraft.ai
- [x] Add equity/macro RSS feeds (Yahoo Finance, MarketWatch, CNBC, Seeking Alpha, Investing.com)
- [x] Swap Kimi→DeepSeek and Opus→Sonnet for cost savings

## Known Issues & Lessons Learned

### Alpaca Limitations
- **Crypto stop orders not supported**: Alpaca returns `{"code":40010001,"message":"invalid order type for crypto order"}` for stop/stop_limit orders on crypto. This is why we built the software stop-loss monitor.
- **Minimum $10 for crypto**: Crypto orders must use notional ordering with minimum $10. Tiny qty orders fail.
- **Price API returns $0 for crypto**: Alpaca's data API (`/v2/assets/{symbol}/bars`) often returns 0 for crypto prices. Use CoinGecko fallback via `MarketDataFetcher`.
- **Crypto close orders must use qty, not notional**: When closing a crypto position, using `notional` (dollar amount) causes Alpaca to reject with "insufficient balance" because the notional→qty conversion at a different price produces a slightly different quantity than what's held. Fix: crypto sells use `qty` param (exact base quantity), crypto buys use `notional`.

### Kimi API
- **International vs Chinese endpoint**: Keys from `platform.moonshot.ai` (international) use `api.moonshot.ai/v1`. Keys from Chinese platform use `api.moonshot.cn/v1`. They are NOT interchangeable — wrong endpoint gives 401.

### LLM Response Parsing
- **TimeHorizon 'N/A'**: DeepSeek sometimes returns "N/A" for `time_horizon` field. Fixed with `_parse_time_horizon()` fallback in `agents/market_analyst.py`.
- Always validate/fallback LLM enum responses — models don't always respect the schema.

### Portfolio State
- **Equity syncs with broker**: `sync_portfolio_with_broker()` overrides internal equity with Alpaca's reported number every 5 min. Tiny drift (<0.05%) between syncs is normal due to price movement. Broker equity is the source of truth.
- **Phantom PnL from test orders**: Early Alpaca test orders created fake -40% daily PnL entries that persisted in `portfolio_state.json` and triggered the circuit breaker repeatedly. When resetting, clear BOTH `portfolio_state.json` AND `circuit_breaker_log.json`.
- **Stop-loss prices must be persisted**: Positions need `quantity` and `stop_loss_price` fields saved to survive restarts. Added in Phase 4.

### Dashboard
- **Script paths must be absolute**: `index.html` is served from `/` route (not `/static/`), so `<script src="lotus-creature.js">` resolves to `/lotus-creature.js` (404). Must use `<script src="/static/lotus-creature.js">`.
- **WebSocket null guard**: When creating `LotusCreature` with `wsUrl: null`, the `_connectWS()` method must guard against `new WebSocket(null)` which crashes. Fixed with `if (!this.opts.wsUrl) return;`.
- **Palette glow is an array**: `getState().palette.glow` returns `[r,g,b]` array, not a CSS color string. Must convert to `rgb(r,g,b)` before using in `style.color`.
- **Canvas background mismatch**: When canvas is smaller than viewport, the page background color doesn't match the canvas radial gradient. Fix: stretch canvas to fill viewport with `position: absolute; width: 100%; height: 100%`.
- **Agent Trading Floor**: Homepage is now an isometric canvas visualization (`agent-floor.html`) embedded in `index.html` via iframe. Lotus creature is disabled but code preserved.
- **Market API returns nested objects**: `/api/market` returns `{BTC: {price: 71529, ...}}` not flat numbers. Agent floor flattens to `{BTC: 71529}` in `fetchLiveData()`.
- **Agent floor live data**: Polls `/api/market`, `/api/price-history`, `/api/equity-history`, `/api/portfolio` every 30s. WebSocket `/ws` events trigger real-time agent walking animations mapped via `EVENT_TO_ANIMATION`.
- **Isometric left wall text direction**: Left wall (gx=0) parallelogram follows wall slope, but text content uses `-ux,-uy` transform and reversed `sPos()` so text faces room interior (readable by agents inside the office).

### Cost Tracking
- **CostTracker was in-memory only**: Costs reset on every restart. Fixed by adding JSON persistence to `data/cost_state.json`. Now survives restarts.
- **Standalone dashboard loses costs**: Starting dashboard independently (not through `main.py`) means `_cost_tracker` is None. Always run via `main.py`.
- **Circuit breaker burned $0.44 in 2 days**: No cooldown/dedup on `escalate_to_opus()` — every 5-min heartbeat re-fired identical Opus calls at $0.04 each when stale portfolio data persisted. Fixed with 4 guards:
  - Pipeline: `run_circuit_breaker_check()` triggers auto-recovery after 6h cooldown (configurable via `auto_recovery_cooldown_hours` in `risk_params.json`). Resets peak equity, unhalts, sends Telegram alert.
  - Circuit breaker: 1-hour cooldown + trigger dedup on Opus calls (cached decision reuse)
  - Pre-flight: minimal `max_tokens=5` ping instead of full JSON prompt
  - CostTracker: `check_budget()` with daily per-provider limits ($0.15/day Anthropic hard cap), enforced in `LLMClient.call_anthropic()` before every API call

### VPS Security
- **Never run as root**: Trading system and webhook should run as a dedicated `trader` user.
- **Disable SSH root + password**: Use key-only auth for `trader` user. Harden in `/etc/ssh/sshd_config.d/01-hardening.conf`.
- **Always enable UFW**: Default deny incoming, allow only needed ports (22, 443, 9000).
- **Install fail2ban**: Protects SSH from brute-force attacks.

### Coinbase Limitations
- **No paper trading**: Coinbase Advanced Trade API has no sandbox. For paper trading, use `EXECUTOR_MODE=paper` or `EXECUTOR_MODE=alpaca` with paper URL.
- **API key format**: Coinbase uses CDP API keys (format: `organizations/{org_id}/apiKeys/{key_id}`) with EC private key secrets. NOT the legacy API keys.
- **Market orders only**: CoinbaseExecutor supports market orders only. No stop/limit orders — the software stop-loss monitor handles stops.
- **Singapore crypto restriction**: Alpaca cannot legally execute crypto for Singapore residents. Use `EXECUTOR_MODE=live` to route crypto→Coinbase, stocks/ETFs→Alpaca.

### Tiger Brokers
- **SDK**: `pip install tigeropen` — RSA keypair auth (PKCS#1 format, NOT PKCS#8)
- **Auth**: Register at https://quant.itigerup.com/#developer → generate RSA keypair → submit public key → receive `tiger_id`
- **Paper trading**: Supports US, HK, China A-shares. **SGX paper trading NOT confirmed** — may require funded live account.
- **Stop orders supported**: Unlike Alpaca crypto, Tiger supports native stop/stop-limit/trailing stop orders.
- **Order types**: market, limit, stop, stop-limit, trailing stop, bracket
- **Markets**: US (NASDAQ, NYSE), HK (HKEX), Singapore (SGX), China A-shares
- **Executor**: `core/tiger_executor.py` — activated via `EXECUTOR_MODE=tiger`

### Broker Reconciliation
- **BrokerReconciler** (`core/broker_sync.py`): Compares internal portfolio against broker positions every heartbeat cycle (5 min)
- **Ghost detection**: Broker has position, internal doesn't → auto-creates internal entry with default stops
- **Orphan detection**: Internal has position, broker doesn't → removes stale internal entry
- **Qty mismatch**: Adjusts internal quantities to match broker (FIFO for over-tracked, newest-adjust for under-tracked)
- **Auto-fix**: Controlled by `BROKER_SYNC_AUTO_FIX=true` in `.env` — off by default, enable for paper trading
- **Pending order handling**: `close_position()` cancels pending orders for symbol before DELETE; skips close if orders stuck in `pending_cancel`

### Multi-Tenant Deployment
- Each user gets their own VPS with their own `.env` file (same git repo)
- `EXECUTOR_MODE` controls which broker per tenant:
  - `paper` — simulated fills (no broker needed)
  - `alpaca` — Alpaca paper/live (US stocks, ETFs, crypto)
  - `ibkr` — Interactive Brokers (US, SGX, HK, global)
  - `tiger` — Tiger Brokers (US, SGX, HK, China A-shares)
  - `live` — RoutingExecutor (crypto→Coinbase, stocks→Alpaca, SGX→IBKR)
- API keys (Alpaca, IBKR, Tiger, Coinbase, LLM, Telegram) are all per-user via `.env`
- GA tracking: use a separate Google Analytics property per user
- Data files (`data/`) are per-VPS, no namespacing needed

### Scheduler Resilience
- **Scheduler is single-threaded**: The `schedule` library runs all tasks on the main thread. If any task (e.g. LLM call) hangs, ALL subsequent tasks are blocked. Fixed with task timeout wrapper — each task runs in a daemon thread with `thread.join(timeout)`. Timeouts: heartbeat 2min, news/chart 5min, proactive/session 10min, weekly 15min.
- **Scheduler watchdog**: Heartbeat checks `_last_scheduler_tick`. If main loop idle >20min → `os._exit(1)` for systemd restart. Sends Telegram alert before exit.
- **Graceful shutdown**: SIGTERM starts 10s force-exit timer (`threading.Timer` + `os._exit(0)`) to prevent zombie processes from non-cooperative threads.
- **WebSocket keepalive**: Server sends `{"category":"system","event_type":"ping"}` every 30s if no events, preventing idle disconnects.

### Dashboard Scheduler
- **Countdown showed 0m 0s**: `schedule` library's `job.next_run` is a naive datetime (UTC on VPS). `isoformat()` without `Z` suffix causes JavaScript to parse as browser local time, creating timezone offset. Fixed by appending `Z` in `/api/scheduler` endpoint.

### Google Analytics
- **Dynamic injection**: `GA4_MEASUREMENT_ID` and `GA4_API_SECRET` are read from `.env`. HTML templates use `{{GA4_MEASUREMENT_ID}}` placeholder, replaced by `server.py` at serve time.
- **Client-side (gtag.js)**: Installed in `index.html` and `agent-floor.html` `<head>` — page views, user sessions
- **Server-side (Measurement Protocol)**: `core/ga4_tracker.py` subscribes to `event_bus` and forwards trade events to `POST https://www.google-analytics.com/mp/collect`. Events: `trade_signal`, `trade_executed`, `stop_loss_triggered`, `circuit_breaker_triggered`, `system_startup`, `agent_escalation`
- **Multi-tenant**: Each tenant must have their own GA property. Set `GA4_MEASUREMENT_ID` and `GA4_API_SECRET` in each tenant's `.env`. Do NOT reuse across deployments — traffic will be mixed.
- **Sukaimi's property**: `G-3DWEFFH8S4` / `1n7J8BhAQiC_R_rXhkvWrA` (set in VPS `.env`, not hardcoded)

## Product Family
- **Tradebot** = product name (the trading system)
- **Tradebot Quant** = Sukaimi's instance, always the most advanced. The showcase. (Formerly 'Queen' — renamed to 'Quant' for quantitative trading connotation)
- **Tradebot Full** = premium customer package (near-identical to Quant)
- **Tradebot Lite** = budget customer package (fewer features)
- **TradeHive** = internal-only platform, collects learnings from ALL Tradebot instances (Sukaimi access only)

## Cost Budget
- Paper trading: ~$1/month (DeepSeek + Sonnet, Kimi disabled)
- VPS: ~$5/month
- Live trading: ~$6-8/month (LLM + VPS)
