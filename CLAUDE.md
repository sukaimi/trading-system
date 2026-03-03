# CLAUDE.md — Trading System Project Guide

## Project Overview

Autonomous multi-agent AI trading system for BTC, ETH, GLDM (gold), SLV (silver) using a 4-tier intelligence stack.

**Owner**: Sukaimi (Code&Canvas)
**Stack**: Python 3.12 / Direct LLM API calls / Ubuntu 24.04 VPS
**Executor**: Alpaca (paper trading) — not IBKR
**PRD**: `docs/TRADING_AGENT_PRD.md`
**Dashboard**: `http://VPS_IP:8080` — Lotus creature + trading dashboard
**Repo**: `https://github.com/sukaimi/trading-system`

## Architecture

### 4-Tier Intelligence Stack
| Tier | Model | Cost | Modules |
|------|-------|------|---------|
| 0 | Pure Python | $0 | RiskManager, Executor, Heartbeat, Portfolio, StopLossMonitor |
| 1 | DeepSeek V3.2 | ~$0.08/mo | NewsScout, TradeJournal |
| 2 | Kimi K2.5 | ~$0.50/mo | MarketAnalyst (escalated), DevilsAdvocate |
| 3 | Claude Opus 4.6 | ~$2.25/mo | WeeklyStrategist, CircuitBreaker, SelfOptimizer |

### Decision Flow
```
NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal
Heartbeat (5 min): health check → stop-loss monitor → circuit breaker
News scan (30 min): full pipeline
3 daily sessions: Asian Open (00:00 UTC), European Overlap (08:00), US Close (14:00)
Weekly: Strategist + SelfOptimizer → rewrite agent params
Emergency: CircuitBreaker → halt trading
```

## Project Structure
```
trading-system/
├── agents/          # 6 agent modules
├── core/            # 12 core modules (schemas, llm_client, pipeline, portfolio, risk_manager,
│                    #   executor, heartbeat, logger, self_optimizer, alpaca_executor,
│                    #   cost_tracker, event_bus, paper_executor)
├── tools/           # 5 tools (news_fetcher, market_data, technical_indicators, correlation, telegram_bot)
├── dashboard/       # FastAPI dashboard server + static files (index.html, lotus-creature.js)
├── config/          # Dynamic params JSON (updated weekly by SelfOptimizer)
├── data/            # Persisted state, trade journal, logs, weekly reviews (gitignored)
├── tests/           # 20 test files, 265 tests (pytest)
├── docs/            # PRD, Lotus spec
├── main.py          # Entry point — 8-task scheduler + dashboard
└── requirements.txt
```

## Key Files
- `main.py` — Entry point, scheduler, executor selection (paper/alpaca/ibkr)
- `core/pipeline.py` — Signal-to-execution orchestration + `check_stop_losses()`
- `core/alpaca_executor.py` — Alpaca paper/live trading executor
- `core/cost_tracker.py` — LLM cost tracking with JSON persistence + daily budget limits (`check_budget()`)
- `core/event_bus.py` — Real-time pub/sub for dashboard WebSocket
- `core/portfolio.py` — Thread-safe portfolio state with JSON persistence
- `dashboard/server.py` — FastAPI server (REST + WebSocket)
- `dashboard/static/index.html` — Dashboard UI + Lotus creature (two-view toggle)
- `dashboard/static/lotus-creature.js` — Animated canvas creature engine (6 stages)
- `config/risk_params.json` — Risk limits (max position 7%, daily loss 5%, drawdown 15%)

## Development Commands
```bash
source venv/bin/activate
pytest tests/ -v                        # Run all 265 tests
pytest tests/test_stop_loss_monitor.py -v  # Run specific test file
python main.py                          # Start full system (scheduler + dashboard on :8080)
```

## VPS Deployment
```
VPS: Ubuntu 24.04 LTS
SSH: ssh trader@VPS_IP  (root login disabled, key-only auth)
Services:
  - trading-system.service — main trading system (runs as trader)
  - webhook.service — GitHub auto-deploy listener on port 9000
Auto-deploy: git push → GitHub webhook → VPS pulls + restarts
Deploy script: /opt/deploy.sh
Firewall (UFW): ports 22 (SSH), 8080 (dashboard), 9000 (webhook)
Fail2ban: active on SSH
```

### Manual VPS commands
```bash
ssh trader@VPS_IP
sudo systemctl status trading-system    # Check status
sudo systemctl restart trading-system   # Restart
journalctl -u trading-system -f         # Live logs
```

## Environment Variables
All API keys in `.env` (never committed). See `.env.example` for template.
Required: DEEPSEEK_API_KEY, KIMI_API_KEY, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Required: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, EXECUTOR_MODE
Optional: GOOGLE_API_KEY (fallback), ALPHA_VANTAGE_API_KEY, CRYPTOCOMPARE_API_KEY

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

### Current Phase: Paper Trading
- Running on VPS with Alpaca paper account
- Monitoring trades, agent decisions, costs

### Pending
- [ ] **Phase 5: Optimization** — Analyze paper trading data, tune parameters (needs 10-20+ trades)
- [ ] **Phase 6: Micro Live** — $100 SGD on Alpaca live
- [ ] **Phase 7: Scale** — Increase capital if profitable
- [ ] Set up GitHub webhook for auto-deploy (add in repo settings → payload URL: http://VPS_IP:9000, secret: see webhook.service)

## Known Issues & Lessons Learned

### Alpaca Limitations
- **Crypto stop orders not supported**: Alpaca returns `{"code":40010001,"message":"invalid order type for crypto order"}` for stop/stop_limit orders on crypto. This is why we built the software stop-loss monitor.
- **Minimum $10 for crypto**: Crypto orders must use notional ordering with minimum $10. Tiny qty orders fail.
- **Price API returns $0 for crypto**: Alpaca's data API (`/v2/assets/{symbol}/bars`) often returns 0 for crypto prices. Use CoinGecko fallback via `MarketDataFetcher`.

### Kimi API
- **International vs Chinese endpoint**: Keys from `platform.moonshot.ai` (international) use `api.moonshot.ai/v1`. Keys from Chinese platform use `api.moonshot.cn/v1`. They are NOT interchangeable — wrong endpoint gives 401.

### LLM Response Parsing
- **TimeHorizon 'N/A'**: DeepSeek sometimes returns "N/A" for `time_horizon` field. Fixed with `_parse_time_horizon()` fallback in `agents/market_analyst.py`.
- Always validate/fallback LLM enum responses — models don't always respect the schema.

### Portfolio State
- **Phantom PnL from test orders**: Early Alpaca test orders created fake -40% daily PnL entries that persisted in `portfolio_state.json` and triggered the circuit breaker repeatedly. When resetting, clear BOTH `portfolio_state.json` AND `circuit_breaker_log.json`.
- **Stop-loss prices must be persisted**: Positions need `quantity` and `stop_loss_price` fields saved to survive restarts. Added in Phase 4.

### Dashboard
- **Script paths must be absolute**: `index.html` is served from `/` route (not `/static/`), so `<script src="lotus-creature.js">` resolves to `/lotus-creature.js` (404). Must use `<script src="/static/lotus-creature.js">`.
- **WebSocket null guard**: When creating `LotusCreature` with `wsUrl: null`, the `_connectWS()` method must guard against `new WebSocket(null)` which crashes. Fixed with `if (!this.opts.wsUrl) return;`.
- **Palette glow is an array**: `getState().palette.glow` returns `[r,g,b]` array, not a CSS color string. Must convert to `rgb(r,g,b)` before using in `style.color`.
- **Canvas background mismatch**: When canvas is smaller than viewport, the page background color doesn't match the canvas radial gradient. Fix: stretch canvas to fill viewport with `position: absolute; width: 100%; height: 100%`.

### Cost Tracking
- **CostTracker was in-memory only**: Costs reset on every restart. Fixed by adding JSON persistence to `data/cost_state.json`. Now survives restarts.
- **Standalone dashboard loses costs**: Starting dashboard independently (not through `main.py`) means `_cost_tracker` is None. Always run via `main.py`.
- **Circuit breaker burned $0.44 in 2 days**: No cooldown/dedup on `escalate_to_opus()` — every 5-min heartbeat re-fired identical Opus calls at $0.04 each when stale portfolio data persisted. Fixed with 4 guards:
  - Pipeline: `run_circuit_breaker_check()` skips entirely when `portfolio.halted` is True
  - Circuit breaker: 1-hour cooldown + trigger dedup on Opus calls (cached decision reuse)
  - Pre-flight: minimal `max_tokens=5` ping instead of full JSON prompt
  - CostTracker: `check_budget()` with daily per-provider limits ($0.15/day Anthropic hard cap), enforced in `LLMClient.call_anthropic()` before every API call

### VPS Security
- **Never run as root**: Trading system and webhook should run as a dedicated `trader` user.
- **Disable SSH root + password**: Use key-only auth for `trader` user. Harden in `/etc/ssh/sshd_config.d/01-hardening.conf`.
- **Always enable UFW**: Default deny incoming, allow only needed ports (22, 8080, 9000).
- **Install fail2ban**: Protects SSH from brute-force attacks.

## Cost Budget
- Paper trading: ~$3/month (all LLM APIs)
- VPS: ~$5/month
- Live trading: ~$8-10/month (LLM + VPS)
