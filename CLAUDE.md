# CLAUDE.md — Trading System Project Guide

## Project Overview

Autonomous multi-agent AI trading system for BTC, ETH, GLDM (gold), SLV (silver) on Interactive Brokers (IBKR) using a 4-tier intelligence stack.

**Owner**: Sukaimi (Code&Canvas)
**Stack**: Python 3.10+ / Direct LLM API calls / Hostinger VPS
**PRD**: `docs/TRADING_AGENT_PRD.md`

## Architecture

### 4-Tier Intelligence Stack
| Tier | Model | Cost | Modules |
|------|-------|------|---------|
| 0 | Pure Python | $0 | RiskManager, Executor, Heartbeat, Portfolio |
| 1 | DeepSeek V3.2 | ~$0.08/mo | NewsScout, TradeJournal |
| 2 | Kimi K2.5 | ~$0.50/mo | MarketAnalyst (escalated), DevilsAdvocate |
| 3 | Claude Opus 4.6 | ~$2.25/mo | WeeklyStrategist, CircuitBreaker, SelfOptimizer |

### Decision Flow
```
NewsScout → MarketAnalyst → DevilsAdvocate → RiskManager → Executor → TradeJournal
Weekly: Strategist + SelfOptimizer → rewrite agent params
Emergency: CircuitBreaker → halt trading
Always: Heartbeat → health checks every 5 min
```

## Project Structure
```
trading-system/
├── agents/          # 6 agent modules (NewsScout, MarketAnalyst, DevilsAdvocate, TradeJournal, WeeklyStrategist, CircuitBreakerAgent)
├── core/            # 9 core modules (schemas, llm_client, pipeline, portfolio, risk_manager, executor, heartbeat, logger, self_optimizer)
├── tools/           # 5 tools (news_fetcher, market_data, technical_indicators, correlation, telegram_bot)
├── config/          # Dynamic params JSON (updated weekly by SelfOptimizer)
├── data/            # Persisted state, trade journal, logs, weekly reviews
├── tests/           # 18 test files (pytest)
├── systemd/         # Service files for VPS deployment
├── docs/            # PRD
├── main.py          # Entry point — 8-task scheduler
└── requirements.txt # 14 dependencies
```

## Key Files
- `main.py` — Entry point, scheduler (heartbeat/5min, news/30min, 3 daily sessions, weekly review)
- `core/schemas.py` — All Pydantic v2 models for inter-agent communication
- `core/llm_client.py` — Unified LLM client (DeepSeek, Kimi, Anthropic, Gemini fallback)
- `core/pipeline.py` — Full signal-to-execution orchestration
- `core/portfolio.py` — Thread-safe portfolio state with JSON persistence
- `config/risk_params.json` — Risk limits (max position 7%, daily loss 5%, drawdown 15%)

## Development Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_risk_manager.py -v

# Run the trading system
python main.py
```

## Environment Variables
All API keys in `.env` (never committed). See `.env.example` for template.
Required: DEEPSEEK_API_KEY, KIMI_API_KEY, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional: GOOGLE_API_KEY (fallback), ALPHA_VANTAGE_API_KEY, CRYPTOCOMPARE_API_KEY

## Conventions
- All inter-agent communication uses strict JSON (Pydantic v2 validated)
- Tier 0 modules are pure Python — no LLM calls
- LLM client supports mock mode for testing (`MOCK_LLM=true`)
- Config files in `config/` are dynamic — updated weekly by SelfOptimizer
- Data files in `data/` are gitignored (private trade data)
- All state persisted to JSON for crash recovery
- Tests use pytest with mocked LLM responses

## Implementation Status

### Completed
- [x] **Phase 1: Foundation** — Project scaffolding, Tier 0 modules, schemas, tests
- [x] **Phase 2: Intelligence Layer** — LLM client, all agents, pipeline, tools
- [x] **Phase 3: Tier 3 Intelligence** — Weekly Strategist, Circuit Breaker, Self-Optimizer

### In Progress
- [ ] **Phase 3 (PRD): Paper Trading** — Deploy on VPS, run against IBKR paper account

### Pending
- [ ] Deploy full system on Hostinger KVM1 (4GB RAM)
- [ ] Install & configure IB Gateway (paper mode, port 7497)
- [ ] Set up Telegram bot for alerts
- [ ] Run against IBKR paper account for 3+ weeks
- [ ] Monitor daily: agent decision quality, false positive rate, kill rate
- [ ] Track all metrics (win rate, latency, cost per agent)
- [ ] **Phase 4: Optimization** — Analyze paper trading data, tune parameters
- [ ] **Phase 5: Micro Live** — $100 SGD on IBKR live (port 7496)
- [ ] **Phase 6: Scale** — Increase capital if profitable

## Cost Budget
- Paper trading: ~$3/month (all LLM APIs)
- Live trading: ~$6-8/month (LLM + VPS upgrade)
- Break-even at $100 SGD capital: needs ~5% monthly return
