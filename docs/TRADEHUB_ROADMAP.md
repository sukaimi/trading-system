# TradeHub Roadmap — Investor Board Recommendations v2

*Generated: 2026-03-08 | Board: Buffett, Lynch, Soros, Burry, Dalio*
*Unified Board Rating: 6.5/10 — CONDITIONAL APPROVAL*
*Timeline: 4 weeks to v3 (live in Week 1)*

---

## Core Problem

TradeHub is UNDERTRADING. 1 completed trade in 7+ weeks. The pipeline's compounding gates
create a structural bias toward inaction. The learning systems are starving for data.

> "The real risk is not going live. The real risk is spending another seven weeks
> with one paper trade." — Soros

> "Show us the trades." — Burry

---

## NEXT-LEVEL AI-NATIVE EDGES

### Edge 1: Reflexivity Detection Through Narrative Analysis (Soros)
Enhance NewsScout to detect not just sentiment but the *structure* of market narratives —
when commentary begins citing price action as evidence for a thesis (self-reinforcing loops).
Flag the stage of the reflexive cycle (formation, peak conviction, reversal). Impossible for
keyword-based sentiment tools — requires understanding the *logic* of narrative construction.

### Edge 2: Regulatory Filing Delta Analysis (Burry)
For AAPL/NVDA/TSLA/AMZN/META: ingest quarterly SEC filings and LLM-diff material changes
between consecutive filings — new risk factors, accounting shifts, revenue recognition changes.
For crypto: on-chain metrics (exchange inflows, whale movements). Surface micro-changes as
signals before any analyst note is published.

### Edge 3: Earnings Call Cross-Pollination (Buffett + Burry)
Ingest earnings call transcripts within minutes. LLM interprets CEO/CFO tone, hedging language,
evasive answers. Cross-reference each company's commentary against theses for OTHER portfolio
holdings (NVDA data center demand → AMZN/META cloud capex thesis). Cross-pollination across
simultaneous earnings calls is physically impossible for a human trader.

### Edge 4: Continuous Principle Extraction and Injection (Dalio)
After every closed trade, mine the full record to extract generalizable principles in natural
language. Store in growing library. Before each new trade, inject relevant principles into
analyst/devil prompts. Compounds non-linearly: marginal at 50 trades, proprietary moat at 500.
Cannot be replicated without making those same 500 trades.

### Edge 5 (Phase 2): Multi-Agent Adversarial Consensus (Lynch)
Replace single MarketAnalyst with four perspective agents — Value, Momentum, Macro, Contrarian.
Track which *patterns of agreement/disagreement* produce best outcomes. Meta-signal from agent
disagreement patterns is unique to multi-agent architectures.

---

## 4-WEEK COMPRESSED ROADMAP

### Week 1: GO LIVE ($200, v1)

#### Day 1 (Monday) — Execution Verification
- [ ] Fund accounts: $100 Alpaca live + $100 Coinbase
- [ ] Set Coinbase API keys in VPS `.env` (`COINBASE_API_KEY`, `COINBASE_API_SECRET`)
- [ ] Test Coinbase round-trip: buy $10 BTC, sell immediately, verify both fill
- [ ] Test Alpaca live: buy $10 SPY fractional, sell immediately
- [ ] Test RoutingExecutor: crypto→Coinbase, stocks→Alpaca
- [ ] Adjust risk params: `max_position_pct`: 5%, `max_open_positions`: 5
- [ ] Set `EXECUTOR_MODE=live` on VPS, deploy
- [ ] Monitor one full heartbeat + news scan cycle
- [ ] Verify stop-loss monitor picks up live positions
- [ ] **GATE: Full cycle clean before proceeding**

#### Day 2 (Tuesday) — Proactive Scan (CRITICAL)
- [ ] Build `run_proactive_scan()` in `pipeline.py`
  - Iterates all 14 assets, fetches regime + technicals
  - Asks MarketAnalyst for setup evaluation (regardless of news)
  - Wire to run once per session (3x daily)
- [ ] Test in dry-run mode (log theses, don't execute)
- [ ] Deploy live, monitor first proactive scan cycle
- [ ] Review generated theses — reasonable? DA killing too many?

#### Day 3 (Wednesday) — Monitor and Tune
- [ ] Review all signals, theses, verdicts, risk decisions from Days 1-2
- [ ] If zero trades: lower `min_confidence_for_trade` to 0.15
- [ ] If DA killing >80%: raise `min_challenges_for_kill` from 3 to 4
- [ ] Verify Telegram alerts arriving
- [ ] Check dashboard shows live positions, PnL, equity curve

#### Day 4 (Thursday) — Validate Live Trades
- [ ] Target: 2-5 live trades should exist by now
- [ ] Verify stop-losses ratcheting correctly (trailing stops)
- [ ] Verify position sizing above broker minimums
- [ ] Fix any bugs from live execution
- [ ] Check portfolio state sync (internal vs broker)

#### Day 5 (Friday) — Week 1 Review
- [ ] Count trades, PnL, win/loss
- [ ] Run SelfOptimizer manually with real data
- [ ] Document Coinbase/Alpaca quirks discovered
- [ ] Plan Week 2 priorities based on what broke

#### Days 6-7 (Weekend)
- [ ] Let system run (crypto trades 24/7)
- [ ] Monitor via Telegram only
- [ ] Sunday review: weekend trades? Errors?

### Week 2: Data-Driven Tuning (v1.5)
- [ ] Tune parameters using Week 1 trade data
- [x] Add reflexivity detection to NewsScout prompt (Edge 1) — DONE
- [x] Begin principle extraction in SelfOptimizer (Edge 4) — DONE
- [ ] Raise `max_position_pct` to 7%, `max_open_positions` to 8 (if stable)
- [ ] Target: 10-15 cumulative live trades

### Week 3: Earnings Intelligence (v2)
- [ ] Implement earnings call transcript ingestion
- [ ] Add cross-pollination logic (Edge 3)
- [ ] Add proactive scan regime templates ("TRENDING_UP: look for pullback to 20-SMA")
- [ ] Run SelfOptimizer with 2+ weeks of data
- [ ] Target: 20-30 cumulative live trades

### Week 4: Knowledge Compounding (v3)
- [ ] Deploy principle library — extracted principles injected into analyst/devil prompts
- [ ] Add source attribution: which agent sources best trades
- [ ] Full portfolio review: correlation, sector, regime distribution
- [ ] Set `max_open_positions` to 12-15 if stable
- [ ] Target: 35-50 cumulative live trades
- [ ] Decide on capital increase based on results

### Deferred (Not in v1-v3)
- Multi-agent adversarial consensus (Edge 5) — needs more data
- Filing delta analysis (Edge 2) — next earnings season
- TradeHive multi-tenant backend — separate project
- Dashboard visual improvements — cosmetic
- Additional test coverage — 607 tests sufficient

---

## PRIORITISED SUGGESTIONS (Reorganised)

### Tier 1: CRITICAL (Week 1 — Before/During Go-Live)

| # | Suggestion | Complexity | Impact |
|---|-----------|-----------|--------|
| 1 | [x] **Proactive scan method** — system evaluates all 14 assets 3x/day regardless of news (DONE - v52) | Medium | HIGHEST — solves undertaking |
| 2 | **Test live execution paths** — Coinbase + Alpaca round-trip verification | Simple | Blocking — must work |
| 3 | **Adjust risk params for $200** — max_position 5%, max_open 5 for Week 1 | Simple | Safety net |
| 4 | [x] **Lower confidence thresholds** — bias toward action during learning phase (DONE - v52) | Simple | HIGH — generates data |
| 5 | [x] **Permanent kill switch** — 25% drawdown = permanent halt in live mode (DONE - v52) | Simple | Safety net |

### Tier 2: HIGH (Weeks 2-3 — Early Live)

| # | Suggestion | Complexity | Impact |
|---|-----------|-----------|--------|
| 6 | [x] **Reflexivity detection in NewsScout** (Edge 1) — forming/peak/reversal detection (DONE - v52) | Medium | Alpha source |
| 7 | [x] **Principle extraction in SelfOptimizer** (Edge 4) — extract_principles() after each trade exit (DONE - v52) | Medium | Compounding edge |
| 8 | [x] **Regime-dependent holding periods** — TRENDING=168h, RANGING=48h, HIGH_VOL=24h, LOW_VOL=96h (DONE - v52) | Medium | Lets winners run |
| 9 | [x] **Signal funnel dashboard** — /api/signal-funnel endpoint + dashboard panel (DONE - v52) | Simple | Diagnostic |
| 10 | [x] **Asymmetric R:R enforcement** — min 2:1 R:R gate + TP floor (DONE - v52) | Medium | Quality filter |

### Tier 2.5: IN PROGRESS (Smart Signal Boosters)

| # | Suggestion | Complexity | Impact |
|---|-----------|-----------|--------|
| 10a | **Liquidity sweep detection** — detect stop hunts + reclaims on OHLCV candles, boost proactive scan signals | Medium | Alpha source — institutional footprint |
| 10b | **Volume anomaly scoring** — flag 2x+ avg volume as institutional interest, boost signal strength | Simple | Signal quality filter |

### Tier 3: MEDIUM (Week 4+ — After 30 Trades)

| # | Suggestion | Complexity | Impact |
|---|-----------|-----------|--------|
| 11 | **Earnings call cross-pollination** (Edge 3) | Complex | Seasonal alpha |
| 12 | **Parameter replay validation** | Medium | Prevents blind optimization |
| 13 | **Chart-signal LLM bypass** | Medium | Cost reduction |
| 14 | **Reduce universe to 5-6 for micro live** | Simple | Focus capital |
| 15 | **Regime transition detection** | Complex | Alpha source |

### Tier 3.5: DEFERRED (After 30+ Trades — Needs Data)

| # | Suggestion | Complexity | Impact |
|---|-----------|-----------|--------|
| 15a | **Put/call ratio** — options flow as macro sentiment proxy | Medium | Institutional signal |
| 15b | **Unusual options activity** — free API integration | Medium | Smart money tracking |

### Tier 4: LOW (When Profitable)

| # | Suggestion | Complexity |
|---|-----------|-----------|
| 16 | Filing delta analysis (Edge 2) | Complex |
| 17 | Multi-agent adversarial consensus (Edge 5) | Complex |
| 18 | Local ML pre-filter | Complex |
| 19 | Multi-timeframe confirmation | Complex |
| 20 | Drawdown-adjusted sizing (Kelly) | Medium |
| 21 | Correlation regime shift detection | Complex |
| 22 | Full event-driven backtester | Complex |
| 23 | Cross-instance learning (TradeHive) | Complex |
| 24 | Google Calendar integration — earnings events, circuit breaker blocks, market hours | Simple |
| 25 | Gmail integration — weekly reports, trade confirmations, alert escalation fallback | Medium |

---

## RISK ASSESSMENT (1-Week Go-Live)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Coinbase executor untested with real money | Medium | Low-Med | Day 1 round-trip test with $10 |
| Position sizing below broker minimums | Low | Low | 5% of $200=$10, above both mins |
| System still doesn't trade | Med-High w/o proactive scan; Low with | HIGH | Day 2 proactive scan is #1 priority |
| Stop-loss fails on live positions | Low | Medium | 72h forced exit + circuit breaker backstop |
| LLM cost spike from proactive scans | Medium | Low | 42 calls/day = $0.04/day = $1.20/month |
| Founder overreacts to small losses | Medium | HIGH | Pre-commit: no frequency-reducing changes for 7 days |

### Board Consensus on Timeline
> "One week to live is not just realistic — it is necessary. The system has 607 tests,
> a working VPS with auto-deploy, and a founder who built the entire thing. The engineering
> risk is near zero. The financial risk is bounded at $200. The greater risk is continuing
> to paper trade." — Unanimous

---

## GO-LIVE READINESS (Revised Milestones)

### Before Going Live (Day 1)
- [ ] Coinbase round-trip verified with real $10
- [ ] Alpaca live round-trip verified with real $10
- [ ] RoutingExecutor routes correctly (crypto→Coinbase, stocks→Alpaca)
- [ ] Risk params adjusted for $200 scale
- [ ] Kill switch configured (permanent halt at $150 equity)
- [ ] One full pipeline cycle completes without errors

### Week 1 Targets
- Trades executed: 5-10
- Max single loss: <$5
- System uptime: >99%
- All stop-losses trigger correctly

### Week 4 Targets (Go/No-Go for Capital Increase)
- Cumulative trades: 35-50
- Win rate: >40%
- Profit factor: >1.2
- Max drawdown: <15%
- Principle library: 10+ extracted rules

---

## Board Ratings

| Investor | Original | Revised | Key Quote |
|----------|----------|---------|-----------|
| Ray Dalio | 7.5 | 7.5 | "The machine is well-designed. Feed it real trades." |
| George Soros | 7.0 | 7.0 | "The edge is in the transition, not the classification." |
| Peter Lynch | 6.5 | 7.0 | "Know what you own, and know why you own it." |
| Warren Buffett | 5.0 | 6.0 | "A magnificent piece of engineering. Now let it trade." |
| Michael Burry | 4.0 | 5.5 | "Show us the trades." |
| **BOARD** | — | **6.5** | **Conditional approval. Go live Week 1.** |

---

---

## PRODUCT LINE & PLATFORM (Phase 7+)

### Tradebot Product Tiers
| Tier | Description | Status |
|------|------------|--------|
| **Tradebot Quant** | Sukaimi's instance — always the most advanced, showcase of all features | Active (paper trading) |
| **Tradebot Full** | Premium customer package — near-identical to Quant | Spec needed |
| **Tradebot Lite** | Budget customer package — fewer features, lower LLM tier | Spec needed |

**TODO:**
- [ ] Spec Tradebot Quant vs Full vs Lite feature matrix (which modules, which LLM tiers, which assets)
- [ ] Define pricing model for Full and Lite
- [ ] Build tenant provisioning (VPS setup script, .env template, onboarding flow)
- [ ] Document upgrade path: Lite → Full → Quant

### TradeHive (Internal-Only Platform)
Central platform that collects learnings from ALL Tradebot instances. Sukaimi access only.

- Hosting: Separate VPS (~$5/mo)
- Storage: SQLite → PostgreSQL at 10+ tenants

**TODO:**
- [ ] Build TradeHive API (ingest trade outcomes, principles, signal accuracy from all instances)
- [ ] Build cross-instance learning aggregator (Edge 23 in roadmap)
- [ ] Add TradeHive client to each Tradebot instance (phone-home on trade close)
- [ ] Dashboard for Sukaimi: view all instances, aggregate stats, principle library

**Status: PLANNING ONLY — do NOT implement until explicitly instructed.**

---

### Recent Fixes (2026-03-08)
- [x] Fix gridlock: duplicate asset changed from fatal flaw to soft flag in DA
- [x] Pre-LLM filter no longer blocks held assets
- [x] Risk manager allows duplicate assets through (DA handles sizing)
- [x] Correlation limit relaxed from 0.50 → 0.65
- [x] TP floor ensures take-profit respects min R:R after regime adjustment
- [x] Signal funnel + R:R enforcement deployed
- [x] Strategy version: v52
- [x] Test count: 627
