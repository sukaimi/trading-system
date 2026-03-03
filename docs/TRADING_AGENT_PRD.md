# AUTONOMOUS AI TRADING AGENT — PRODUCT REQUIREMENTS DOCUMENT
## Drop this into Claude Code. It knows what to do.

**Version**: 1.0  
**Date**: March 1, 2026  
**Owner**: Sukaimi (Code&Canvas)  
**Stack**: Python 3.10+ / CrewAI / Hostinger VPS (KVM1 → KVM2)

---

# TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Architecture & Intelligence Stack](#2-architecture--intelligence-stack)
3. [Agent Definitions (All 10)](#3-agent-definitions)
4. [Communication Schemas](#4-communication-schemas)
5. [Tools & Data Sources](#5-tools--data-sources)
6. [Security & Monitoring](#6-security--monitoring)
7. [Deployment & Infrastructure](#7-deployment--infrastructure)
8. [Implementation Phases](#8-implementation-phases)
9. [Cost Model](#9-cost-model)
10. [Success Criteria](#10-success-criteria)

---

# 1. SYSTEM OVERVIEW

## What This Is

An autonomous multi-agent AI trading system that trades crypto (BTC, ETH), gold (GLDM), and silver (SLV) on Interactive Brokers (IBKR) using a 4-tier intelligence stack. The system scans news, generates trade theses, stress-tests them via a contrarian agent, executes trades, journals everything, and self-optimizes weekly.

## Core Philosophy

**"Right model for the right job. No compromises on critical decisions. Zero waste on tasks that don't need intelligence."**

- 95% of operations are Python (free) or DeepSeek ($0.28/1M tokens)
- Only strategic decisions use expensive models
- The system rewrites its own rules every week via Opus 4.6 self-optimization
- Paper trade first, prove the system works, then go live with $100 SGD

## Key Constraints

- **Capital**: $100 SGD initial live (paper trading first)
- **VPS**: Hostinger KVM1 (4GB RAM) for paper → KVM2 (8GB) for live
- **No GPU**: All AI via API, no local models
- **Monthly budget**: ~$3/month paper, ~$6-8/month live (all-in)
- **Timezone**: Singapore (SGT, UTC+8)
- **Broker**: Interactive Brokers (IBKR) Lite — $0 commissions
- **Assets**: BTC, ETH (crypto), GLDM (gold ETF), SLV (silver ETF)

---

# 2. ARCHITECTURE & INTELLIGENCE STACK

## 4-Tier Intelligence Stack

| Tier | Model | Cost (per 1M tokens) | Use Cases |
|------|-------|---------------------|-----------|
| **Tier 0: Free** | Pure Python | $0 | Heartbeat, health checks, position math, stop-loss triggers, IBKR API calls, cron scheduling, security monitoring |
| **Tier 1: Workhorse** | DeepSeek V3.2 | $0.28 in / $0.42 out | News scanning, sentiment tagging, market data parsing, trade journaling, routine analysis |
| **Tier 2: Analyst** | Kimi K2.5 | $0.60 in / $3.00 out | Trade thesis generation (escalated), Devil's Advocate challenges, daily analysis |
| **Tier 3: Commander** | Claude Opus 4.6 | $5.00 in / $25.00 out | Weekly strategy review, circuit breaker, self-optimization directives |

**Fallback chain**: DeepSeek → Gemini 2.0 Flash (free via Google AI Studio) → retry queue

## Agent Roster

| # | Agent | Tier | Model | Calls/Day | Monthly Cost |
|---|-------|------|-------|-----------|-------------|
| 1 | News Scout | 1 | DeepSeek V3.2 | 48 | ~$0.05 |
| 2 | Market Analyst | 1→2 | DeepSeek → Kimi (escalation) | 10-15 | ~$0.20 |
| 3 | Devil's Advocate | 2 | Kimi K2.5 | 3-5 | ~$0.30 |
| 4 | Risk Manager | 0 | Python | Always on | $0 |
| 5 | Executor | 0 | Python (IBKR API) | As needed | $0 |
| 6 | Trade Journal | 1 | DeepSeek V3.2 | 5 | ~$0.03 |
| 7 | Weekly Strategist | 3 | Opus 4.6 | 4/month | ~$1.00 |
| 8 | Circuit Breaker | 3 | Opus 4.6 | 1-2/month | ~$0.50 |
| 9 | Self-Optimizer | 3 | Opus 4.6 | 4/month | ~$0.75 |
| 10 | Heartbeat | 0 | Python | Every 5 min | $0 |

## Decision Flow

```
News Scout (DeepSeek, every 30 min)
  ↓ signal_alert JSON if signal detected
Market Analyst (DeepSeek → escalates to Kimi if confidence ≥ 0.6)
  ↓ trade_thesis JSON or no_trade
Devil's Advocate (Kimi — only triggered on trade proposals)
  ↓ APPROVED / MODIFIED / KILLED
Risk Manager (Python — position sizing, limit checks)
  ↓ execution_order JSON
Executor (Python — IBKR API)
  ↓ order_confirmation
Trade Journal (DeepSeek — logs everything)
  ↓ written to trade_journal.json

Weekly: Strategist + Self-Optimizer (Opus) → rewrites agent prompts
Emergency: Circuit Breaker (Opus) → can halt all trading
Always: Heartbeat (Python) → health checks every 5 min
```

---

# 3. AGENT DEFINITIONS

## 3.1 — NEWS SCOUT
**Tier 1 | DeepSeek V3.2 | ~48 calls/day | ~$0.05/month**

### CrewAI Config
```python
news_scout = Agent(
    role="Financial News Intelligence Analyst",
    goal="Scan, filter, and classify financial news that could impact portfolio assets (crypto, gold, silver). Extract actionable trading signals from noise. Never miss a market-moving event. Never cry wolf on irrelevant news.",
    backstory="""You are a former Bloomberg Terminal news analyst who spent 12 years 
    at a macro hedge fund filtering 3,000+ daily headlines into the 5-10 that actually 
    move markets. You have an obsessive eye for separating signal from noise. You know 
    that 95% of financial news is recycled commentary and only 5% contains genuinely 
    new information. You specialize in:
    
    - Central bank policy signals (Fed, ECB, BOJ rate decisions, minutes, speeches)
    - Geopolitical events with market impact (sanctions, trade wars, conflicts)
    - Crypto-specific catalysts (ETF approvals, exchange hacks, regulatory actions)
    - Precious metals drivers (inflation data, USD strength, safe-haven flows)
    - Black swan detection (unexpected events with outsized market impact)
    
    You NEVER forward recycled opinion pieces. You ONLY escalate when you detect 
    genuinely new information that isn't already priced in. Your false positive rate 
    is below 10% — when you raise an alert, traders pay attention.""",
    
    llm="deepseek/deepseek-chat",
    tools=["news_rss_tool", "news_api_tool", "crypto_news_tool"],
    verbose=True,
    memory=True,
    max_iter=3,
    inject_date=True,
    date_format="%B %d, %Y %H:%M UTC"
)
```

### Prompt Rules
- Signal classification: 0.8-1.0 CRITICAL | 0.5-0.7 NOTABLE | 0.3-0.4 MONITOR | 0.0-0.2 NOISE (discard)
- Anti-noise: If 3+ outlets have same story → already priced in, downgrade by 0.2
- Speculation filter: "could/might/may" without citing source → downgrade by 0.3
- Scheduled events: Only alert if actual deviates from consensus by >10%
- Weekend crypto penalty: Increase scrutiny, low volume amplifies false signals

### Schedule
- Every 30 min during market hours
- Every 60 min off-hours
- Immediate scan on CRITICAL keyword triggers (Fed, hack, crash, halt)

### Output → `signal_alert` JSON
```json
{
  "type": "signal_alert",
  "timestamp": "ISO-8601",
  "source": "news_scout",
  "asset": "BTC|ETH|GLDM|SLV|MACRO",
  "signal_strength": 0.0-1.0,
  "headline": "Max 100 chars",
  "sentiment": "bullish|bearish|neutral|uncertain",
  "category": "central_bank|geopolitical|regulatory|technical|macro|crypto_specific|precious_metals",
  "new_information": "What NEW fact isn't priced in?",
  "raw_sources": ["source1", "source2"],
  "urgency": "critical|high|medium|low",
  "already_priced_in": false,
  "confidence_in_classification": 0.0-1.0
}
```

### Dynamic Parameters (updated weekly by Self-Optimizer)
```json
{
  "signal_weights": {
    "central_bank": 0.9, "geopolitical": 0.8, "regulatory_crypto": 0.8,
    "inflation_data": 0.7, "whale_movement": 0.5, "analyst_opinion": 0.3
  },
  "scan_frequency_minutes": 30,
  "min_signal_threshold": 0.4,
  "weekend_signal_penalty": 0.15,
  "max_alerts_per_hour": 5
}
```

### Performance Metrics
- False positive rate (target: <15%)
- Miss rate (target: 0%)
- Signal-to-noise ratio (target: <5%)
- Latency from publish to alert (target: <5 min)

---

## 3.2 — MARKET ANALYST
**Tier 1→2 | DeepSeek → Kimi K2.5 (escalation) | ~10-15 calls/day | ~$0.20/month**

### CrewAI Config
```python
market_analyst = Agent(
    role="Quantitative Market Analyst & Trade Thesis Generator",
    goal="Transform raw market signals into structured trade theses with clear entry/exit criteria, confidence scores, and risk quantification. Generate trade ideas ONLY when edge exists — silence is a valid output.",
    backstory="""You are a former quantitative analyst at Bridgewater Associates who 
    spent 10 years building systematic trading models for Ray Dalio's All-Weather 
    portfolio. You specialize in:
    
    - Cross-asset correlation analysis (crypto ↔ gold ↔ USD ↔ bonds)
    - Regime detection (risk-on vs risk-off environments)
    - Multi-timeframe technical analysis (4H, daily, weekly)
    - Macro overlay (how central bank policy affects each asset class)
    
    Your edge: crypto, gold, and silver respond to the SAME macro forces (real rates, 
    USD strength, risk appetite) but with different lag times and amplitudes. You 
    exploit these timing differences.
    
    You have a strict "no thesis, no trade" policy. You NEVER chase momentum without 
    a structural reason. You require minimum 2 independent confirming signals before 
    proposing any trade. You explicitly quantify what could go wrong.""",
    
    llm="deepseek/deepseek-chat",
    tools=["market_data_tool", "technical_indicator_tool", "correlation_tool"],
    verbose=True,
    memory=True,
    reasoning=True,
    max_reasoning_attempts=2,
    max_iter=5,
    inject_date=True
)
```

### Thesis Requirements (min 2 of 3 must confirm)
- □ Fundamental catalyst (macro event, policy change, supply shock)
- □ Technical setup (key level break, RSI divergence, volume confirmation)
- □ Cross-asset confirmation (e.g., USD weakening + gold strengthening)

### Confidence Scoring
- 0.8-1.0: All 3 signals. Strong edge. Full position.
- 0.6-0.79: 2 of 3. Moderate edge. Reduced position.
- 0.4-0.59: 1 signal. Marginal — pass.
- Below 0.4: No trade.

### Escalation Logic (DeepSeek → Kimi)
```python
def should_escalate(thesis):
    if thesis["confidence"] >= 0.6 and thesis["suggested_position_pct"] > 3.0:
        return True
    if thesis["asset"] in ["BTC", "ETH"] and thesis["time_horizon"] == "swing":
        return True
    if len(thesis["what_could_go_wrong"]) >= 3:
        return True
    return False
```

### Technical Indicators
RSI(14), MACD(12/26/9), Bollinger Bands(20,2σ), 50/200 SMA, Volume vs 20d avg, ATR(14), DXY, BTC-Gold 30d correlation

### Scheduled Analysis (non-triggered)
- 08:00 SGT: Asian session opening — overnight recap
- 16:00 SGT: European overlap — cross-session check
- 22:00 SGT: US session — daily close analysis

### Output → `trade_thesis` JSON
```json
{
  "type": "trade_thesis",
  "timestamp": "ISO-8601",
  "source": "market_analyst",
  "model_used": "deepseek|kimi",
  "escalated": false,
  "asset": "BTC|ETH|GLDM|SLV",
  "direction": "long|short|neutral",
  "confidence": 0.0-1.0,
  "thesis": "2-3 sentence rationale",
  "confirming_signals": {
    "fundamental": {"present": true, "description": "..."},
    "technical": {"present": true, "description": "..."},
    "cross_asset": {"present": false, "description": "..."}
  },
  "entry_trigger": "Specific condition",
  "invalidation_level": "Price where thesis is wrong",
  "time_horizon": "1-3 days|1 week|swing",
  "suggested_position_pct": 3.0,
  "risk_reward_ratio": "1:2.5",
  "supporting_data": {
    "rsi_14": 35.2, "macd_signal": "bullish_cross", "volume_vs_avg": 1.4,
    "usd_dxy": 103.5, "btc_correlation_gold_30d": 0.42
  },
  "what_could_go_wrong": ["reason1", "reason2"],
  "triggering_alert_id": "signal_alert_xxx"
}
```

### Dynamic Parameters (updated weekly)
```json
{
  "min_confidence_for_trade": 0.6,
  "max_open_positions": 3,
  "signal_weights": {"fundamental": 0.35, "technical": 0.40, "cross_asset": 0.25},
  "asset_allocation_limits": {"BTC": 0.30, "ETH": 0.20, "GLDM": 0.30, "SLV": 0.20},
  "min_risk_reward_ratio": 1.5
}
```

---

## 3.3 — DEVIL'S ADVOCATE
**Tier 2 | Kimi K2.5 | ~3-5 calls/day | ~$0.30/month**

### CrewAI Config
```python
devils_advocate = Agent(
    role="Contrarian Risk Challenger & Pre-Trade Stress Tester",
    goal="Find every reason the proposed trade will FAIL. Kill bad trades before they lose money. You are the last line of intellectual defense before real capital is deployed.",
    backstory="""You are a former risk manager at Citadel known as "The Executioner" 
    because you killed 60% of proposed trades before execution. Every killed trade 
    that would have lost money earned you a point. Every killed trade that would have 
    been profitable cost you two points. Over 8 years, you maintained a net positive 
    score.
    
    Your methodology:
    1. ASSUME THE THESIS IS WRONG. Start from the position it will lose money.
    2. IDENTIFY THE CROWDED TRADE. If everyone sees the setup, the edge is gone.
    3. STRESS TEST. What happens if the key assumption DOES fail? Calculate loss.
    4. CHECK THE BASE RATE. What % of similar setups historically profited?
    5. FIND SECOND-ORDER EFFECTS. The obvious catalyst is priced in. What reverses it?
    
    Three outputs: APPROVED | APPROVED_WITH_MODIFICATION | KILLED
    You approve ~40% of trades. The 60% you kill save more than the 40% earn.""",
    
    llm="moonshot/kimi-k2.5",
    tools=["market_data_tool", "correlation_tool", "volatility_tool"],
    verbose=True,
    memory=True,
    reasoning=True,
    max_reasoning_attempts=3,
    max_iter=5,
    inject_date=True
)
```

### 7 Challenge Framework
Every thesis is challenged on ALL 7 dimensions:

1. **CROWDED TRADE CHECK** — Score 0-1. Is everyone already positioned?
2. **ASSUMPTION STRESS TEST** — What's the single critical assumption? What if it's wrong?
3. **TIMING CHALLENGE** — FOMO reaction or genuine entry? Has the move already happened?
4. **CORRELATION TRAP** — Adding risk or diversifying? Portfolio concentration after trade?
5. **HISTORICAL BASE RATE** — Win rate of similar setups? Sample size?
6. **SECOND-ORDER EFFECTS** — Non-obvious consequence that could reverse the trade?
7. **SIZE APPROPRIATENESS** — Confidence→size map: 0.6-0.7→3% | 0.7-0.8→5% | 0.8+→7%

### Decision Framework
- 0-1 challenges flagged → **APPROVED**
- 2-3 challenges flagged → **APPROVED_WITH_MODIFICATION**
- 4+ challenges OR any fatal flaw → **KILLED**

### Fatal Flaws (auto-kill)
- Position would breach daily loss limit
- Trade adds >50% correlation to existing positions
- No invalidation level defined
- Thesis relies on single unconfirmed news source
- Crowded trade score ≥ 0.8

### Output → `devils_verdict` JSON
```json
{
  "type": "devils_verdict",
  "timestamp": "ISO-8601",
  "source": "devils_advocate",
  "original_thesis_id": "thesis_xxx",
  "verdict": "APPROVED|APPROVED_WITH_MODIFICATION|KILLED",
  "challenges": {
    "crowded_trade": {"score": 0.3, "reasoning": "..."},
    "assumption_stress_test": {"critical_assumption": "...", "if_wrong": "...", "worst_case_loss_usd": 2.50},
    "timing": {"flag": false, "reasoning": "..."},
    "correlation_trap": {"flag": false, "post_trade_concentration": 0.30},
    "base_rate": {"available": true, "historical_win_rate": 0.62, "sample_size": 24},
    "second_order_effects": {"identified": ["..."], "severity": "medium"},
    "size_check": {"proposed_pct": 5.0, "recommended_pct": 3.5}
  },
  "flags_raised": 2,
  "fatal_flaws": [],
  "modifications": ["Reduce size to 3.5%", "Add VIX exit trigger"],
  "confidence_adjusted": 0.65,
  "final_reasoning": "Thesis sound but needs smaller position."
}
```

### Kill Rate Targets
- Kill rate: 40-60%
- False kill rate: <20%
- Modification rate: 20-30%
- Clean approval rate: 15-25%

---

## 3.4 — RISK MANAGER
**Tier 0 | Pure Python | Always running | $0/month**

### Purpose
Position sizing, limit enforcement, security monitoring. NO AI — pure code is faster and more reliable for math.

### Implementation (Python, not CrewAI Agent)
```python
class RiskManager:
    """Pure Python risk management — no LLM calls"""
    
    def __init__(self, config):
        self.max_position_pct = config["max_position_pct"]          # 7%
        self.max_daily_loss_pct = config["max_daily_loss_pct"]       # 5%
        self.max_total_drawdown_pct = config["max_total_drawdown_pct"] # 15%
        self.max_open_positions = config["max_open_positions"]        # 3
        self.max_correlation_exposure = config["max_correlation"]     # 0.50
        self.stop_loss_atr_multiplier = config["stop_loss_atr_mult"] # 2.0
    
    def validate_order(self, execution_order, portfolio_state):
        """Returns (approved: bool, reason: str, adjusted_order: dict)"""
        checks = []
        
        # Check 1: Position size within limits
        if execution_order["position_size_pct"] > self.max_position_pct:
            checks.append(f"Position {execution_order['position_size_pct']}% exceeds max {self.max_position_pct}%")
        
        # Check 2: Daily loss limit
        daily_pnl = portfolio_state["daily_pnl_pct"]
        if daily_pnl <= -self.max_daily_loss_pct:
            return False, "DAILY LOSS LIMIT REACHED — NO NEW TRADES", None
        
        # Check 3: Total drawdown
        if portfolio_state["drawdown_from_peak_pct"] >= self.max_total_drawdown_pct:
            return False, "MAX DRAWDOWN REACHED — CIRCUIT BREAKER", None
        
        # Check 4: Open position count
        if len(portfolio_state["open_positions"]) >= self.max_open_positions:
            return False, f"Max {self.max_open_positions} positions reached", None
        
        # Check 5: Correlation check
        # ... (calculate correlation with existing positions)
        
        # Check 6: Stop-loss must be defined
        if not execution_order.get("stop_loss"):
            return False, "No stop-loss defined — rejected", None
        
        # All passed
        return True, "APPROVED", execution_order
    
    def calculate_position_size(self, confidence, atr, portfolio_value):
        """ATR-based position sizing adjusted by confidence"""
        base_risk_per_trade = 0.02  # Risk 2% per trade
        stop_distance = atr * self.stop_loss_atr_multiplier
        position_value = (portfolio_value * base_risk_per_trade) / stop_distance
        
        # Scale by confidence
        confidence_scalar = min(confidence, 1.0)
        position_value *= confidence_scalar
        
        # Cap at max
        max_value = portfolio_value * (self.max_position_pct / 100)
        return min(position_value, max_value)
```

### Risk Parameters (updated weekly by Self-Optimizer)
```json
{
  "max_position_pct": 7.0,
  "max_daily_loss_pct": 5.0,
  "max_total_drawdown_pct": 15.0,
  "max_open_positions": 3,
  "max_correlation": 0.50,
  "stop_loss_atr_mult": 2.0,
  "base_risk_per_trade_pct": 2.0,
  "circuit_breaker_drawdown_pct": 25.0
}
```

### Security Monitoring (also in Risk Manager)
Python cron jobs, no AI needed:
- API key age tracking → Telegram alert at 30 days
- Failed auth monitoring → alert after 3 consecutive failures
- Unexpected trade detection → trade outside agent workflow = immediate halt + alert
- VPS resource anomaly → CPU > 80% sustained 5 min = alert
- IBKR API restricted to localhost only
- All keys in `.env`, never hardcoded
- UFW firewall + fail2ban for SSH

---

## 3.5 — EXECUTOR
**Tier 0 | Pure Python (IBKR API) | As needed | $0/month**

### Purpose
Execute validated orders on IBKR. No AI — just API calls with retry logic and confirmation.

### Implementation
```python
class Executor:
    """IBKR trade execution — pure Python"""
    
    def __init__(self, ibkr_config):
        self.ib = IB()  # ib_insync library
        self.paper_mode = ibkr_config["paper_mode"]
        self.host = "127.0.0.1"
        self.port = 7497 if self.paper_mode else 7496
    
    def execute(self, execution_order):
        """Execute a validated order. Returns confirmation JSON."""
        try:
            self.ib.connect(self.host, self.port, clientId=1)
            
            contract = self._build_contract(execution_order["asset"])
            order = self._build_order(execution_order)
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(2)
            
            # Set stop-loss as bracket order
            if execution_order.get("stop_loss"):
                self._place_stop_loss(contract, execution_order)
            
            # Set take-profit
            if execution_order.get("take_profit"):
                self._place_take_profit(contract, execution_order)
            
            return {
                "type": "order_confirmation",
                "timestamp": datetime.utcnow().isoformat(),
                "order_id": trade.order.orderId,
                "asset": execution_order["asset"],
                "direction": execution_order["direction"],
                "quantity": execution_order["quantity"],
                "fill_price": trade.orderStatus.avgFillPrice,
                "status": trade.orderStatus.status,
                "thesis_id": execution_order["thesis_id"]
            }
        except Exception as e:
            return {
                "type": "order_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "thesis_id": execution_order["thesis_id"]
            }
        finally:
            self.ib.disconnect()
    
    def _build_contract(self, asset):
        if asset in ["BTC", "ETH"]:
            return Crypto(asset, "PAXOS", "USD")
        elif asset == "GLDM":
            return Stock("GLDM", "ARCA", "USD")
        elif asset == "SLV":
            return Stock("SLV", "ARCA", "USD")
```

### IBKR Configuration
- Library: `ib_insync` (Python wrapper for IBKR TWS API)
- Paper trading port: 7497
- Live trading port: 7496
- TWS or IB Gateway must be running on VPS
- Client ID: 1 (single bot)
- Auto-restart: systemd service for IB Gateway

### Supported Order Types
- Market orders (default for small positions)
- Limit orders (for entries near key levels)
- Stop-loss orders (always set — bracket style)
- Take-profit orders (optional)

---

## 3.6 — TRADE JOURNAL
**Tier 1 | DeepSeek V3.2 | ~5 calls/day | ~$0.03/month**

### CrewAI Config
```python
trade_journal = Agent(
    role="Quantitative Trade Documentarian",
    goal="Create meticulous, structured records of every trade decision — entries, exits, theses, outcomes, and lessons. Your journal is the raw data that feeds the weekly self-optimization loop. Incomplete records = blind optimization.",
    backstory="""You are an obsessive trade documentarian who worked at DE Shaw's 
    operations desk. You believe that every trade outcome — win or loss — contains 
    information that future decisions should learn from. You record not just WHAT 
    happened, but WHY the decision was made, what the market context was, and 
    whether the original thesis was correct.
    
    You never editorialise. You record facts. You structure data for machine 
    consumption, not human reading. Your journal entries are designed to be 
    consumed by the Self-Optimizer agent for pattern detection.""",
    
    llm="deepseek/deepseek-chat",
    verbose=True,
    memory=True,
    max_iter=2
)
```

### Journal Entry Schema (appended to `trade_journal.json`)
```json
{
  "trade_id": "trade_20260301_143500",
  "timestamp_open": "ISO-8601",
  "timestamp_close": "ISO-8601 | null if open",
  "asset": "BTC",
  "direction": "long",
  "entry_price": 62500.00,
  "exit_price": null,
  "position_size_usd": 3.50,
  "position_size_pct": 3.5,
  "stop_loss_price": 61250.00,
  "take_profit_price": 65625.00,
  "thesis_summary": "Fed pause + RSI oversold + gold confirming",
  "thesis_confidence_original": 0.72,
  "thesis_confidence_after_devil": 0.65,
  "devil_modifications": ["Reduced size from 5% to 3.5%"],
  "news_trigger": "Fed signals rate pause — Reuters",
  "confirming_signals": ["fundamental", "technical"],
  "market_context": {
    "dxy": 103.5,
    "vix": 18.2,
    "btc_rsi_14": 35.2,
    "regime": "risk_on"
  },
  "outcome": {
    "pnl_usd": null,
    "pnl_pct": null,
    "thesis_correct": null,
    "exit_reason": null,
    "hold_duration_hours": null
  },
  "lessons": null
}
```

### What Gets Journaled
- Every trade entry + exit
- Every NO_TRADE decision (with reasoning)
- Every KILLED trade from Devil's Advocate
- Daily market state summaries (from Market Analyst scheduled runs)
- All signal alerts from News Scout (even discarded ones, summarized)

### Close/Exit Entry (appended when trade closes)
```json
{
  "trade_id": "trade_20260301_143500",
  "timestamp_close": "2026-03-03T09:15:00Z",
  "exit_price": 64800.00,
  "outcome": {
    "pnl_usd": 0.26,
    "pnl_pct": 3.68,
    "thesis_correct": true,
    "exit_reason": "take_profit_hit",
    "hold_duration_hours": 42.5
  },
  "lessons": "Fed pause thesis played out. RSI bounce from 35 was reliable signal. Cross-asset gold confirmation added conviction. Devil's size reduction was correct — weekend volume was thin."
}
```

---

## 3.7 — WEEKLY STRATEGIST
**Tier 3 | Claude Opus 4.6 | 4 calls/month (every Sunday) | ~$1.00/month**

### CrewAI Config
```python
weekly_strategist = Agent(
    role="Chief Investment Strategist",
    goal="Conduct a comprehensive weekly review of all trading activity, identify patterns in wins and losses, assess whether the current strategy is working, and produce specific, actionable directives for the coming week.",
    backstory="""You are a former CIO of a $500M systematic fund who now consults 
    for algorithmic trading startups. You've seen hundreds of trading systems fail 
    and the pattern is always the same: they don't learn from their mistakes fast 
    enough. Your weekly review is the difference between a system that compounds 
    knowledge and one that repeats errors.
    
    You think in terms of:
    - Win rate vs expectancy (a 40% win rate is fine if winners are 3x losers)
    - Regime awareness (is the strategy aligned with current market conditions?)
    - Agent performance (which agent is adding value, which is noise?)
    - Risk calibration (are we taking appropriate risk for the opportunity set?)
    
    You produce SPECIFIC directives, not vague suggestions. "Increase Fed news 
    weight from 0.6 to 0.8" not "pay more attention to central banks."
    
    You are the only agent authorized to change strategy parameters.""",
    
    llm="anthropic/claude-opus-4-6",
    verbose=True,
    memory=True,
    reasoning=True,
    max_reasoning_attempts=3,
    max_iter=5
)
```

### Weekly Input Package (assembled by Trade Journal)
```json
{
  "week_ending": "2026-03-07",
  "portfolio_summary": {
    "starting_value": 100.00,
    "ending_value": 103.50,
    "weekly_return_pct": 3.5,
    "max_drawdown_pct": 1.2,
    "sharpe_estimate": 2.1
  },
  "trade_summary": {
    "total_trades": 5,
    "wins": 3,
    "losses": 1,
    "no_trades": 1,
    "win_rate": 0.75,
    "avg_win_pct": 3.2,
    "avg_loss_pct": -1.5,
    "expectancy_per_trade_pct": 1.88
  },
  "agent_performance": {
    "news_scout": {"alerts_sent": 24, "false_positives": 3, "missed_events": 0},
    "market_analyst": {"theses_proposed": 7, "no_trades": 4, "escalated_to_kimi": 2},
    "devils_advocate": {"received": 3, "approved": 1, "modified": 1, "killed": 1},
    "risk_manager": {"orders_validated": 2, "orders_rejected": 0}
  },
  "full_trade_journal": [...],
  "current_strategy_params": {...},
  "current_risk_params": {...},
  "market_regime_this_week": "risk_on_moderate"
}
```

### Weekly Output → Strategy Directives
```json
{
  "type": "weekly_strategy_directive",
  "timestamp": "ISO-8601",
  "source": "weekly_strategist",
  "model": "opus-4.6",
  "week_reviewed": "2026-03-01 to 2026-03-07",
  
  "assessment": {
    "overall": "Strategy performing well. 3.5% weekly return with low drawdown.",
    "what_worked": [
      "Fed news signals → crypto trades had 100% hit rate this week",
      "Devil's Advocate size reduction prevented overexposure on thin weekend"
    ],
    "what_failed": [
      "SLV trade thesis was technically sound but ignored USD strength — loss",
      "News Scout generated 3 false positives on crypto regulatory rumors"
    ],
    "regime_assessment": "Risk-on environment likely to continue next week. Fed pause narrative supporting crypto and gold."
  },
  
  "parameter_changes": [
    {
      "target": "news_scout",
      "parameter": "signal_weights.regulatory_crypto",
      "old_value": 0.8,
      "new_value": 0.6,
      "reason": "3 false positives on regulatory rumors this week. Reduce weight until confirmed by multiple sources."
    },
    {
      "target": "market_analyst",
      "parameter": "signal_weights.cross_asset",
      "old_value": 0.25,
      "new_value": 0.35,
      "reason": "SLV loss was caused by ignoring USD strength. Cross-asset confirmation needs more weight."
    },
    {
      "target": "risk_manager",
      "parameter": "stop_loss_atr_mult",
      "old_value": 2.0,
      "new_value": 1.8,
      "reason": "Tighter stops would have reduced SLV loss by 30%."
    }
  ],
  
  "next_week_focus": [
    "Watch for FOMC minutes Wednesday — could shift regime",
    "Crypto weekend volume has been declining — reduce weekend position sizes",
    "Gold-BTC correlation increasing — monitor for divergence trades"
  ],
  
  "risk_adjustments": {
    "max_position_pct": 7.0,
    "max_daily_loss_pct": 5.0,
    "note": "No changes to risk parameters — current levels appropriate"
  }
}
```

---

## 3.8 — CIRCUIT BREAKER
**Tier 3 | Claude Opus 4.6 | 1-2 calls/month (ideally 0) | ~$0.50/month**

### Purpose
Emergency decision-making when the system hits critical thresholds. This is the "big red button" that can halt all trading.

### Implementation (Hybrid: Python trigger → Opus decision)
```python
class CircuitBreaker:
    """Python monitors thresholds. Opus makes crisis decisions."""
    
    TRIGGERS = {
        "daily_loss_pct": -5.0,         # Day loss exceeds 5%
        "total_drawdown_pct": -15.0,    # Total drawdown from peak
        "consecutive_losses": 5,         # 5 losses in a row
        "vix_spike": 35.0,              # VIX above 35 (extreme fear)
        "flash_crash_pct": -10.0,       # Any asset drops 10% in 1 hour
        "api_failure_count": 3,          # 3 consecutive API failures
    }
    
    def check(self, portfolio_state, market_data):
        triggered = []
        
        if portfolio_state["daily_pnl_pct"] <= self.TRIGGERS["daily_loss_pct"]:
            triggered.append("daily_loss_limit")
        if portfolio_state["drawdown_from_peak_pct"] <= self.TRIGGERS["total_drawdown_pct"]:
            triggered.append("max_drawdown")
        if portfolio_state["consecutive_losses"] >= self.TRIGGERS["consecutive_losses"]:
            triggered.append("losing_streak")
        if market_data.get("vix", 0) >= self.TRIGGERS["vix_spike"]:
            triggered.append("vix_extreme")
        
        if triggered:
            # IMMEDIATELY halt new trades
            self.halt_new_trades()
            # Call Opus for crisis decision
            return self.escalate_to_opus(triggered, portfolio_state, market_data)
        
        return None
```

### Opus Crisis Prompt
```
CIRCUIT BREAKER ACTIVATED.

Triggers fired: {triggered_list}
Current portfolio state: {portfolio_state}
Current market data: {market_data}
Open positions: {open_positions}
Recent trade history: {last_10_trades}

YOU MUST DECIDE:
1. CLOSE ALL POSITIONS — nuclear option, go to 100% cash
2. CLOSE LOSING POSITIONS ONLY — keep winners, cut losers
3. HOLD — triggers are noise, positions are fundamentally sound
4. REDUCE ALL — cut all position sizes by 50%
5. HEDGE — open offsetting positions

For each decision, explain:
- Why this is the right call given the specific trigger
- What conditions would change your decision
- When to resume normal trading

Output strict JSON with decision, reasoning, and resume conditions.
```

### Circuit Breaker Output
```json
{
  "type": "circuit_breaker_decision",
  "timestamp": "ISO-8601",
  "source": "circuit_breaker",
  "model": "opus-4.6",
  "triggers_fired": ["daily_loss_limit", "consecutive_losses"],
  "decision": "CLOSE_LOSING_POSITIONS",
  "positions_to_close": ["trade_20260305_091500"],
  "positions_to_keep": ["trade_20260304_160000"],
  "reasoning": "Daily loss driven by single bad SLV trade. BTC position is profitable and thesis intact. Close SLV, keep BTC.",
  "resume_conditions": "Resume normal trading tomorrow if VIX < 25 and no new triggers.",
  "notify_owner": true,
  "telegram_message": "⚠️ CIRCUIT BREAKER: Closed SLV position (-2.3%). Keeping BTC (+1.5%). Resuming tomorrow."
}
```

---

## 3.9 — SELF-OPTIMIZER
**Tier 3 | Claude Opus 4.6 | 4 calls/month (post-weekly review) | ~$0.75/month**

### Purpose
Takes the Weekly Strategist's directives and AUTOMATICALLY implements them by rewriting agent system prompts and strategy parameters. This is what makes the system learn.

### Implementation
```python
class SelfOptimizer:
    """Translates Opus strategy directives into actual config changes"""
    
    def apply_directives(self, weekly_directive):
        changes_applied = []
        
        for change in weekly_directive["parameter_changes"]:
            target_agent = change["target"]
            param_path = change["parameter"]
            new_value = change["new_value"]
            old_value = change["old_value"]
            
            # Update the config file
            self._update_config(target_agent, param_path, new_value)
            
            # Log the change with version
            changes_applied.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": target_agent,
                "parameter": param_path,
                "old_value": old_value,
                "new_value": new_value,
                "reason": change["reason"],
                "directive_source": weekly_directive["week_reviewed"],
                "version": self._increment_version()
            })
        
        # Write optimization log
        self._append_to_log(changes_applied)
        
        # Notify via Telegram
        self._send_telegram_summary(changes_applied)
        
        return changes_applied
    
    def _update_config(self, agent, param_path, value):
        """Update agent's dynamic parameters JSON file"""
        config_file = f"config/{agent}_params.json"
        config = json.load(open(config_file))
        
        # Navigate nested path (e.g., "signal_weights.regulatory_crypto")
        keys = param_path.split(".")
        obj = config
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
        
        json.dump(config, open(config_file, "w"), indent=2)
    
    def _increment_version(self):
        """Track strategy version for A/B comparison"""
        version_file = "config/strategy_version.json"
        v = json.load(open(version_file))
        v["version"] += 1
        v["last_updated"] = datetime.utcnow().isoformat()
        json.dump(v, open(version_file, "w"))
        return v["version"]
```

### Optimization Log Schema (appended to `optimization_log.json`)
```json
{
  "version": 4,
  "timestamp": "2026-03-07T23:30:00Z",
  "directive_week": "2026-03-01 to 2026-03-07",
  "changes": [
    {
      "agent": "news_scout",
      "parameter": "signal_weights.regulatory_crypto",
      "old_value": 0.8,
      "new_value": 0.6,
      "reason": "3 false positives on regulatory rumors"
    }
  ],
  "portfolio_value_at_change": 103.50,
  "cumulative_return_pct": 3.5
}
```

### Rollback Capability
If next week's performance degrades after a parameter change, the Self-Optimizer can rollback:
```python
def rollback(self, version_to_restore):
    """Restore parameters from a previous version"""
    log = json.load(open("data/optimization_log.json"))
    target = [entry for entry in log if entry["version"] == version_to_restore]
    # Reverse all changes from versions after target
```

---

## 3.10 — HEARTBEAT
**Tier 0 | Pure Python | Every 5 minutes | $0/month**

### Implementation
```python
class Heartbeat:
    """System health monitor — cron every 5 minutes"""
    
    def check(self):
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # 1. VPS health
        status["checks"]["cpu"] = psutil.cpu_percent() < 80
        status["checks"]["ram"] = psutil.virtual_memory().percent < 85
        status["checks"]["disk"] = psutil.disk_usage("/").percent < 90
        
        # 2. API connectivity
        status["checks"]["ibkr"] = self._ping_ibkr()
        status["checks"]["deepseek"] = self._ping_api("https://api.deepseek.com/v1/models")
        status["checks"]["kimi"] = self._ping_api("https://api.moonshot.cn/v1/models")
        
        # 3. Process health
        status["checks"]["crewai_running"] = self._check_process("crewai")
        status["checks"]["ib_gateway"] = self._check_process("ibgateway")
        
        # 4. Security
        status["checks"]["api_keys_fresh"] = self._check_key_age() < 30  # days
        status["checks"]["failed_logins"] = self._check_fail2ban() < 3
        
        # 5. Trading state
        status["checks"]["circuit_breaker_active"] = not self._is_halted()
        
        # Alert if any check fails
        failures = [k for k, v in status["checks"].items() if not v]
        if failures:
            self._send_telegram_alert(f"⚠️ HEARTBEAT FAILURES: {', '.join(failures)}")
        
        return status
```

---

# 4. COMMUNICATION SCHEMAS

All inter-agent communication uses strict JSON. See schemas defined in each agent section above. Summary of message types:

| Message Type | From | To | Purpose |
|-------------|------|-----|---------|
| `signal_alert` | News Scout | Market Analyst | New market signal detected |
| `no_signal` | News Scout | Log | Nothing actionable found |
| `trade_thesis` | Market Analyst | Devil's Advocate | Proposed trade |
| `no_trade` | Market Analyst | Trade Journal | Signal evaluated, no edge |
| `devils_verdict` | Devil's Advocate | Risk Manager | Approved/Modified/Killed |
| `execution_order` | Risk Manager | Executor | Validated order |
| `order_confirmation` | Executor | Trade Journal | Fill details |
| `order_error` | Executor | Risk Manager + Telegram | Execution failed |
| `journal_entry` | Trade Journal | Storage | Trade record |
| `weekly_strategy_directive` | Weekly Strategist | Self-Optimizer | Strategy changes |
| `circuit_breaker_decision` | Circuit Breaker | Executor + Telegram | Emergency action |
| `heartbeat_status` | Heartbeat | Log + Telegram (on failure) | System health |
| `optimization_applied` | Self-Optimizer | Log + Telegram | Config changes made |

---

# 5. TOOLS & DATA SOURCES

## News Sources (Free Tier)
| Source | Type | Rate Limit | Assets |
|--------|------|-----------|--------|
| Reuters RSS | Macro/FX | Unlimited | All |
| CoinDesk RSS | Crypto | Unlimited | BTC, ETH |
| Kitco RSS | Precious Metals | Unlimited | GLDM, SLV |
| Alpha Vantage News API | Multi-asset | 5/min, 500/day | All |
| CryptoCompare News API | Crypto | 100K/month | BTC, ETH |
| Fed Calendar (scrape) | Central Bank | Daily | Macro |

## Market Data
| Source | Data | Cost |
|--------|------|------|
| yfinance | Price, volume, technicals | Free |
| Alpha Vantage | Technical indicators | Free (500 calls/day) |
| IBKR TWS API | Real-time quotes, positions | Free with account |
| CoinGecko | Crypto prices | Free (30 calls/min) |

## LLM APIs
| Provider | Model | API Endpoint | Key Location |
|----------|-------|-------------|--------------|
| DeepSeek | V3.2 | api.deepseek.com | .env: DEEPSEEK_API_KEY |
| Moonshot | Kimi K2.5 | api.moonshot.cn | .env: KIMI_API_KEY |
| Anthropic | Opus 4.6 | api.anthropic.com | .env: ANTHROPIC_API_KEY |
| Google | Gemini Flash (fallback) | generativelanguage.googleapis.com | .env: GOOGLE_API_KEY |

## Python Libraries
```
crewai[tools]>=0.152.0
ib_insync>=0.9.86
yfinance>=0.2.31
pandas>=2.0
numpy>=1.24
feedparser>=6.0       # RSS parsing
requests>=2.31
python-telegram-bot>=20.0
psutil>=5.9           # System monitoring
schedule>=1.2         # Cron-like scheduling
python-dotenv>=1.0
```

---

# 6. SECURITY & MONITORING

## Security (built into Risk Manager + Heartbeat)
- All API keys in `.env` file, loaded via `python-dotenv`
- `.env` in `.gitignore` — never committed
- IBKR API bound to localhost (127.0.0.1) only
- SSH key-only auth, password disabled
- UFW firewall: only 22 (SSH) + Telegram webhook port open
- fail2ban: ban after 3 failed SSH attempts
- API key rotation reminders every 30 days via Telegram

## Monitoring
- Heartbeat every 5 minutes → failures alert via Telegram
- All agent outputs logged to JSON files
- Strategy version tracking for A/B analysis
- Telegram bot for: alerts, daily summaries, weekly reports, circuit breaker notifications

## Telegram Bot Alerts
| Priority | Alert Type | Example |
|----------|-----------|---------|
| 🔴 CRITICAL | Circuit breaker fired | "CIRCUIT BREAKER: Daily loss limit -5%. All trades halted." |
| 🟡 WARNING | Heartbeat failure | "IBKR API connection lost. Retrying..." |
| 🟢 INFO | Trade executed | "LONG BTC @ $62,500. Size: $3.50 (3.5%). SL: $61,250" |
| 📊 DAILY | Daily summary | "Day P&L: +$0.85 (+0.85%). Open: 2 positions" |
| 📋 WEEKLY | Strategy review | Full Opus weekly directive summary |

---

# 7. DEPLOYMENT & INFRASTRUCTURE

## VPS Setup (Hostinger KVM1 → KVM2)

### Paper Trading Phase (KVM1 — 4GB RAM)
```bash
# System
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3-pip python3-venv git ufw fail2ban -y

# Firewall
sudo ufw allow 22
sudo ufw enable

# Project
cd /home
git clone [repo] trading-system
cd trading-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with API keys

# IB Gateway (headless)
# Download IB Gateway from IBKR site
# Configure for paper trading (port 7497)

# Systemd services
sudo cp systemd/trading-agent.service /etc/systemd/system/
sudo cp systemd/ib-gateway.service /etc/systemd/system/
sudo systemctl enable trading-agent ib-gateway
sudo systemctl start ib-gateway
sudo systemctl start trading-agent
```

### Live Trading Phase (KVM2 — 8GB RAM)
Same setup, but:
- IB Gateway configured for live trading (port 7496)
- Paper mode flag set to `false` in config
- Tighter security monitoring
- Daily backup of trade journal + config

## File Structure
```
trading-system/
├── agents/
│   ├── __init__.py
│   ├── news_scout.py
│   ├── market_analyst.py
│   ├── devils_advocate.py
│   ├── trade_journal.py
│   ├── weekly_strategist.py
│   └── circuit_breaker_agent.py
├── core/
│   ├── __init__.py
│   ├── risk_manager.py
│   ├── executor.py
│   ├── heartbeat.py
│   └── self_optimizer.py
├── tools/
│   ├── __init__.py
│   ├── news_fetcher.py
│   ├── market_data.py
│   ├── technical_indicators.py
│   ├── correlation.py
│   └── telegram_bot.py
├── config/
│   ├── news_scout_params.json
│   ├── market_analyst_params.json
│   ├── devils_advocate_params.json
│   ├── risk_params.json
│   ├── strategy_version.json
│   └── assets.json
├── data/
│   ├── trade_journal.json
│   ├── optimization_log.json
│   └── weekly_reviews/
├── systemd/
│   ├── trading-agent.service
│   └── ib-gateway.service
├── tests/
│   ├── test_risk_manager.py
│   ├── test_executor.py
│   ├── test_news_scout.py
│   └── test_circuit_breaker.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── TRADING_AGENT_PRD.md          # This file
```

---

# 8. IMPLEMENTATION PHASES

## Phase 1: Foundation (Week 1)
- [ ] Set up VPS (KVM1) with Python, UFW, fail2ban
- [ ] Open IBKR paper trading account (SingPass verification)
- [ ] Set up all API accounts: DeepSeek, Kimi, Anthropic, Google (fallback)
- [ ] Install IB Gateway on VPS (paper mode)
- [ ] Create project structure, requirements.txt, .env
- [ ] Build Tier 0 agents: Heartbeat, Risk Manager, Executor (Python only)
- [ ] Test IBKR paper connectivity — place/cancel test orders
- [ ] Set up Telegram bot for alerts

## Phase 2: Intelligence Layer (Week 2)
- [ ] Build News Scout agent (DeepSeek) with RSS + API tools
- [ ] Build Market Analyst agent (DeepSeek + Kimi escalation)
- [ ] Build Devil's Advocate agent (Kimi)
- [ ] Build Trade Journal agent (DeepSeek)
- [ ] Create all JSON schemas for inter-agent communication
- [ ] Wire agents together in CrewAI — test full flow with mock data

## Phase 3: Paper Trading (Weeks 3-5)
- [ ] Deploy full system on KVM1
- [ ] Run against IBKR paper account
- [ ] Monitor daily: Are agents making sensible decisions?
- [ ] Track all metrics: win rate, false positive rate, kill rate, latency
- [ ] Build Weekly Strategist agent (Opus) — run first weekly review
- [ ] Build Self-Optimizer — test automatic parameter updates
- [ ] Build Circuit Breaker — test with simulated drawdown

## Phase 4: Optimization (Week 6)
- [ ] Analyze 3 weeks of paper trading data
- [ ] Review every Opus weekly directive — were adjustments correct?
- [ ] Identify and fix: false positives, missed signals, bad position sizing
- [ ] Run Self-Optimizer for 2+ cycles — verify parameter improvements
- [ ] Stress test Circuit Breaker with edge cases

## Phase 5: Micro Live (Weeks 7-10)
- [ ] Upgrade to KVM2 (8GB RAM)
- [ ] Fund IBKR with $100 SGD
- [ ] Switch IB Gateway to live (port 7496)
- [ ] Deploy with conservative parameters (tighter stops, smaller positions)
- [ ] Monitor daily, intervene only on circuit breaker
- [ ] Goal: Don't lose >15% in first month
- [ ] Track: actual P&L vs paper trading performance

## Phase 6: Scale (Weeks 11+)
- [ ] If profitable: increase to $200 SGD
- [ ] If break-even: continue learning, adjust strategy
- [ ] If losing: analyze why, fix, or pause
- [ ] Target: $500 SGD by month 3 if system proves profitable

---

# 9. COST MODEL

## Paper Trading Phase (~$3.05/month)
| Item | Cost |
|------|------|
| DeepSeek V3.2 (Tier 1) | ~$0.30 |
| Kimi K2.5 (Tier 2) | ~$0.50 |
| Claude Opus 4.6 (Tier 3) | ~$2.25 |
| VPS KVM1 (already paying) | $0 |
| IBKR paper account | $0 |
| News APIs (free tiers) | $0 |
| **TOTAL** | **~$3.05** |

## Live Trading Phase (~$6-8/month)
| Item | Cost |
|------|------|
| DeepSeek V3.2 | ~$0.30 |
| Kimi K2.5 | ~$0.50 |
| Claude Opus 4.6 | ~$2.25 |
| VPS KVM2 upgrade | +$3-5 |
| IBKR Lite | $0 |
| **TOTAL** | **~$6-8** |

## Break-Even Analysis
| Capital | 5% Monthly Return | Monthly Cost | Net |
|---------|-------------------|-------------|-----|
| $100 SGD | +$5.00 | -$3.05 | **+$1.95** ✅ |
| $200 SGD | +$10.00 | -$6-8 | **+$2-4** ✅ |
| $500 SGD | +$25.00 | -$6-8 | **+$17-19** ✅ |

---

# 10. SUCCESS CRITERIA

## Paper Trading Phase (must achieve before going live)
- [ ] System runs 3+ weeks without manual intervention
- [ ] Win rate > 50% (by trade count)
- [ ] Expectancy positive (avg win × win rate - avg loss × loss rate > 0)
- [ ] No missed critical news events
- [ ] Devil's Advocate kill rate 40-60%
- [ ] Circuit breaker fires correctly on simulated drawdown
- [ ] Self-Optimizer successfully applies 2+ weekly directive cycles
- [ ] All agents within cost budget

## Live Trading Phase (monthly targets)
- [ ] Portfolio return > monthly LLM costs (break-even minimum)
- [ ] Max drawdown < 15%
- [ ] No daily loss > 5%
- [ ] System uptime > 99% (heartbeat monitoring)
- [ ] Strategy version improves week-over-week (measured by Sharpe estimate)

---

# APPENDIX: API SETUP QUICK REFERENCE

### DeepSeek
```
URL: https://platform.deepseek.com
Free: 5M tokens for first 30 days
Model: deepseek-chat (V3.2)
Pricing after trial: $0.28/$0.42 per 1M tokens
```

### Kimi K2.5 (Moonshot)
```
URL: https://platform.moonshot.cn
Bonus: $5 recharge gets $5 bonus ($10 total credit)
Model: moonshot-v1-auto (Kimi K2.5)
Pricing: $0.60/$3.00 per 1M tokens
```

### Claude Opus 4.6 (Anthropic)
```
URL: https://console.anthropic.com
Free: $5 credit on signup
Model: claude-opus-4-6
Pricing: $5.00/$25.00 per 1M tokens
```

### Gemini 2.0 Flash (Google — fallback only)
```
URL: https://aistudio.google.com
Free: 15 RPM / 1M TPM / 1,500 RPD
Model: gemini-2.0-flash
Pricing: Free tier sufficient for fallback use
```

### Interactive Brokers
```
URL: https://www.interactivebrokers.com.sg
Account: Individual, Lite plan ($0 commissions)
Verification: SingPass (Singapore residents)
API: TWS API via IB Gateway (headless)
Paper: Port 7497 | Live: Port 7496
Library: ib_insync (Python)
Assets: Crypto (PAXOS), US ETFs (GLDM, SLV)
```

---

**END OF PRD. This document contains everything needed to build the complete system. Drop into Claude Code and execute phase by phase.**
