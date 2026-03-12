# Alpha Vantage Integration Design

**Status**: Plan
**Date**: 2026-03-12
**API tier**: Free (25 calls/min, 500 calls/day)
**Key**: Already in `.env` as `ALPHA_VANTAGE_API_KEY`

---

## Current State

Alpha Vantage is already partially integrated:
- `tools/news_fetcher.py` calls `NEWS_SENTIMENT` (function=NEWS_SENTIMENT) every 15-min news scan
- Returns `overall_sentiment_score` and per-ticker `relevance_score`, attached to articles
- Rate-limited to 5 calls/min internally (`_check_av_rate_limit`)
- Ticker mapping in `AV_TICKERS` dict (e.g., BTC -> CRYPTO:BTC)

What is **not** used: earnings data, sector performance, additional technical indicators.

---

## 1. What to Integrate (Priority Order)

### P0: Ticker-Level Sentiment Aggregation (Low effort, high value)

**Problem**: The existing integration fetches sentiment per-article but never aggregates it into a single ticker-level sentiment score. The MarketAnalyst LLM prompt receives raw articles, not a pre-computed sentiment signal.

**What to do**: Extract and aggregate the per-ticker `ticker_sentiment` array from `NEWS_SENTIMENT` response. Each item has `ticker_sentiment_score` (-1 to 1) and `ticker_sentiment_label` (Bearish/Neutral/Bullish). Compute a weighted average (weighted by `relevance_score`) per ticker and pass it into `_gather_technical_data()` as `av_sentiment`.

**API cost**: 0 extra calls. The data is already in the `NEWS_SENTIMENT` response; we just discard most of it.

### P1: Earnings Data (Medium effort, high value)

**Problem**: `core/earnings_calendar.py` uses hardcoded dates for 5 stocks. Open universe stocks have no earnings data at all. This means we can enter positions days before a surprise earnings report.

**What to do**: Use `EARNINGS` function (`function=EARNINGS&symbol=AAPL`) to fetch upcoming and recent earnings (actual vs estimated EPS, surprise %). Cache aggressively -- earnings dates change rarely.

**Enriches**:
- `EarningsCalendar.days_until_earnings()` -- replace hardcoded dict with API-backed data
- `MarketAnalyst` -- add EPS surprise history to thesis prompt ("AAPL has beaten estimates 8 of last 8 quarters")
- `RiskManager` -- existing `has_earnings_soon()` gate works automatically once calendar is populated

**API cost**: 1 call per ticker, cacheable for 7 days. ~14 calls for core assets per week, plus 1 call per new open-universe ticker.

### P2: Sector Performance (Medium effort, medium value)

**Problem**: `RiskManager._get_sector()` knows which sector an asset is in, but has no data on relative sector strength. We can't tilt toward strong sectors or avoid weak ones.

**What to do**: Use `SECTOR` function (`function=SECTOR`) -- returns real-time sector performance rankings (1d, 5d, 1mo, 3mo, YTD, 1yr) for all 11 GICS sectors. Single call, no ticker needed.

**Enriches**:
- `MarketAnalyst` -- add sector momentum to market context ("Technology +2.3% this week, rank #2/11")
- `DevilsAdvocate` -- flag trades in weakest sectors
- `RiskManager` -- future enhancement: sector momentum filter

**API cost**: 1 call. Cacheable for 4 hours (sector rankings are slow-moving intraday).

### P3: Additional Indicators -- VWAP, Stochastic, OBV (Low priority)

**Assessment**: Not worth API calls. All three are trivially computable from OHLCV data we already fetch via yfinance.

- **VWAP**: `cumsum(close * volume) / cumsum(volume)` -- 5 lines of numpy in `TechnicalIndicators`
- **Stochastic**: `(close - low_N) / (high_N - low_N) * 100` -- 10 lines of numpy
- **OBV**: `cumsum(sign(delta_close) * volume)` -- 3 lines of numpy

**Recommendation**: Add these to `tools/technical_indicators.py` as pure Python. Zero API cost.

---

## 2. Architecture

### Option A (Recommended): Extend `MarketDataFetcher` with an `AlphaVantageClient` helper

```
tools/
  market_data.py          # MarketDataFetcher (unchanged interface)
  alpha_vantage.py        # NEW: AlphaVantageClient (all AV API calls + caching)
  news_fetcher.py         # Refactor: delegate AV news calls to AlphaVantageClient
  technical_indicators.py # Add vwap(), stochastic(), obv()
```

**Why a separate file**: The AV client needs its own rate limiter, cache layer, and error handling. Putting it in `market_data.py` would bloat that file. The news fetcher would import and delegate its existing AV calls to the shared client (single rate limiter).

### AlphaVantageClient responsibilities

```python
class AlphaVantageClient:
    def __init__(self):
        self._key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._cache: dict[str, tuple[float, Any]] = {}  # key -> (expiry_ts, data)
        self._daily_calls = 0
        self._daily_reset = date.today()
        self._call_timestamps: list[float] = []  # for per-minute rate limiting

    def news_sentiment(self, tickers: list[str]) -> dict  # existing, moved from news_fetcher
    def ticker_sentiment(self, ticker: str) -> dict       # aggregated score from news_sentiment
    def earnings(self, ticker: str) -> dict                # EARNINGS function
    def sector_performance(self) -> dict                   # SECTOR function

    def _call(self, params: dict) -> dict | None           # rate-limited, cached, counted
```

### Caching Strategy

| Endpoint | Cache TTL | Key | Rationale |
|----------|-----------|-----|-----------|
| `NEWS_SENTIMENT` | 15 min | `news:{tickers_hash}` | Matches news scan interval |
| `EARNINGS` | 7 days | `earnings:{ticker}` | Dates don't change often |
| `SECTOR` | 4 hours | `sector` | Sector rankings are slow-moving |

Cache is in-memory dict with expiry timestamps. On restart, cold cache is fine -- data repopulates within one cycle. No disk persistence needed (unlike portfolio state).

### Fallback Behavior

All AV calls return `None` or empty dict on failure. No agent should crash or change behavior if AV is unavailable:
- `ticker_sentiment()` -> `None` means LLM prompt omits the sentiment line (no signal, not a negative signal)
- `earnings()` -> falls back to hardcoded `EARNINGS_DATES` dict (current behavior)
- `sector_performance()` -> omitted from market context

### Rate Limiter

Single shared rate limiter in `AlphaVantageClient._call()`:
- Track timestamps of last 25 calls; block if 25th-most-recent is < 60s ago
- Track `_daily_calls` counter; hard-stop at 450/day (leave 50-call buffer)
- Reset counter when `date.today()` changes

---

## 3. Rate Budget

### Current daily call usage

| Trigger | Frequency | AV calls/trigger | Daily calls |
|---------|-----------|-------------------|-------------|
| News scan (NEWS_SENTIMENT) | Every 15 min = 96/day | 1 | 96 |
| **Total current** | | | **96** |

### Proposed additions

| Trigger | Frequency | AV calls/trigger | Daily calls |
|---------|-----------|-------------------|-------------|
| News scan (NEWS_SENTIMENT) | 96/day | 1 (already counted) | 96 |
| Sector performance (SECTOR) | Every 4h = 6/day | 1 | 6 |
| Earnings refresh (EARNINGS) | On new open-universe ticker | 1 per ticker | ~5-10 |
| Earnings refresh (core 14) | Weekly | 5 (stocks only) | ~1 |
| **Total proposed** | | | **~108** |

**Margin**: 500 - 108 = 392 calls/day headroom. Well within budget even with bursty open-universe discovery days.

### If budget becomes tight

1. Reduce `NEWS_SENTIMENT` to every 30 min (48 calls/day, save 48)
2. Cache `EARNINGS` for 14 days instead of 7 (negligible savings, but reduces burst)
3. Call `SECTOR` only during 3 trading sessions (3 calls/day, save 3)

---

## 4. Integration Points

### NewsScout (`agents/news_scout.py`)
- **Already integrated**: receives articles with `sentiment_score` from AV
- **Enhancement**: Pass aggregated ticker sentiment to the LLM prompt so it sees "AV sentiment for AAPL: 0.34 (Bullish)" instead of raw per-article scores

### MarketAnalyst (`agents/market_analyst.py`)
- **In `_gather_technical_data()`**: Add `av_sentiment` field (float, -1 to 1) from `ticker_sentiment()`
- **In `ANALYSIS_PROMPT`**: Add "Alpha Vantage Sentiment: {av_sentiment}" to the market context block
- **In market context**: Add sector performance ranking when available
- **Earnings context**: Add "days until earnings: N, last 4 EPS surprises: [+5%, -2%, ...]"

### ChartAnalyst (`agents/chart_analyst.py`)
- **No change needed**. Chart analyst is pure price-action; adding sentiment would dilute its specialization.

### DevilsAdvocate
- **Enhancement**: Include sector rank in counter-argument prompt ("Trade is in Energy sector, currently ranked #9/11 over 1 month")

### EarningsCalendar (`core/earnings_calendar.py`)
- **Replace hardcoded dict**: `EARNINGS_DATES` becomes a cache populated from `AlphaVantageClient.earnings()`. Hardcoded dict stays as ultimate fallback.
- **Add EPS surprise data**: New method `get_eps_history(ticker) -> list[dict]` with actual/estimated/surprise_pct

### RiskManager (`core/risk_manager.py`)
- **No code change needed initially**. It already calls `EarningsCalendar.has_earnings_soon()`. Once the calendar is API-backed, it automatically covers open-universe stocks.
- **Future**: Sector momentum filter (reject trades in bottom-3 sectors unless confidence > 0.7)

---

## 5. Implementation Order

1. **Create `tools/alpha_vantage.py`** -- client with rate limiter, cache, `_call()` method
2. **Add `ticker_sentiment()`** -- aggregate existing NEWS_SENTIMENT data per ticker
3. **Refactor `news_fetcher.py`** -- delegate AV calls to the shared client
4. **Wire into MarketAnalyst** -- add `av_sentiment` to tech data dict and prompt
5. **Add `earnings()`** -- fetch and cache EARNINGS data
6. **Upgrade EarningsCalendar** -- API-backed with hardcoded fallback
7. **Add `sector_performance()`** -- fetch and cache SECTOR data
8. **Wire sector data into MarketAnalyst market context**
9. **Add `vwap()`, `stochastic()`, `obv()` to TechnicalIndicators** (pure Python, no AV)
10. **Tests**: Mock AV responses, test caching/rate-limiting/fallback

Steps 1-4 are the minimum viable integration. Steps 5-8 can follow in a second pass. Step 9 is independent.
