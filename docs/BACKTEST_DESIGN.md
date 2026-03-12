# Backtest Module — Design Document

**Purpose**: Validate and optimize risk parameters against historical data. The system currently shows ~90% win rate but loses money — tiny wins get eaten by larger losses. This module identifies the parameter combinations that maximize risk-adjusted returns, not just win rate.

**Non-goals**: This is NOT a full strategy backtester. It does not simulate LLM agent decisions, news signals, or the full pipeline. It tests *what happens after a trade is entered* — specifically whether stop-loss, take-profit, trailing stop, and position sizing parameters produce profitable outcomes.

---

## 1. Scope — What Gets Tested

| Parameter | Current Value | Sweep Range | Step |
|-----------|--------------|-------------|------|
| `default_stop_loss_pct` | 3.0% | 1.0% - 8.0% | 0.5% |
| `default_take_profit_pct` | 6.0% | 3.0% - 15.0% | 1.0% |
| `stop_loss_atr_mult` | 2.0 | 1.0 - 4.0 | 0.5 |
| `take_profit_atr_mult` | 3.0 | 2.0 - 6.0 | 0.5 |
| `trailing_stop_activation_pct` | 2.0% | 1.0% - 5.0% | 0.5% |
| `trailing_stop_distance_pct` | 1.5% | 0.5% - 3.0% | 0.5% |
| `max_position_pct` | 5.0% | 3.0% - 10.0% | 1.0% |
| `base_risk_per_trade_pct` | 1.5% | 0.5% - 3.0% | 0.5% |
| `max_correlation` | 0.65 | 0.40 - 0.80 | 0.05 |
| `min_reward_risk_ratio` | 2.0 | 1.5 - 4.0 | 0.5 |
| Holding period (forced exit) | 72h | 24h - 168h | 24h |

Sweep modes (run one dimension at a time to keep runtime manageable):
- **stops**: SL% x TP% grid (15 x 13 = 195 combos)
- **atr-stops**: SL ATR mult x TP ATR mult grid (7 x 9 = 63 combos)
- **trailing**: activation% x distance% grid (9 x 6 = 54 combos)
- **sizing**: max_position% x base_risk% grid (8 x 6 = 48 combos)
- **holding**: holding period x SL% (7 x 15 = 105 combos)
- **full**: Top-N from each sub-sweep combined (capped at ~500 combos)

---

## 2. Architecture

### New files

```
tools/
  backtest/
    __init__.py          # Package init, exports run_backtest()
    __main__.py          # CLI entry point (python -m tools.backtest)
    engine.py            # BacktestEngine — core simulation loop (~300 lines)
    data_loader.py       # HistoricalDataLoader — yfinance + disk cache (~120 lines)
    portfolio_sim.py     # SimulatedPortfolio — tracks equity, positions, drawdown (~200 lines)
    param_sweep.py       # ParameterSweeper — generates param combos, runs engine in parallel (~150 lines)
    report.py            # Reporter — metrics calculation + output formatting (~150 lines)
    signals.py           # SignalGenerator — synthetic entry signals from indicators (~100 lines)
```

**Total estimated**: 7 files, ~1,020 lines of new code.

### Integration with existing codebase

- **Reuses directly**: `TechnicalIndicators` (ATR, RSI, MACD, Bollinger, ADX, SMA, liquidity sweep, volume anomaly), `TradingFriction` (spread, commission, borrow)
- **Reads**: `config/risk_params.json` as baseline parameters
- **Does NOT touch**: agents, pipeline, LLM client, portfolio state, executor, dashboard — completely isolated

### Dependency diagram

```
CLI (__main__.py)
  → ParameterSweeper (param_sweep.py)
    → BacktestEngine (engine.py)
      → HistoricalDataLoader (data_loader.py)   [yfinance + cache]
      → SimulatedPortfolio (portfolio_sim.py)    [equity tracking]
      → SignalGenerator (signals.py)             [entry logic]
      → TechnicalIndicators                      [existing]
      → TradingFriction                          [existing]
    → Reporter (report.py)                       [metrics + output]
```

---

## 3. Data Loading

### `HistoricalDataLoader`

```
data_loader.get_historical(symbol, period="2y", interval="1d") -> pd.DataFrame
```

**Source**: yfinance via `yf.Ticker(symbol).history(period, interval)`

**Disk cache**: `data/backtest_cache/{symbol}_{period}_{interval}.parquet`
- Cache key: `{symbol}_{period}_{interval}`
- TTL: 24 hours (re-download if file mtime > 24h old)
- Format: Parquet (fast read/write, columnar, small on disk)
- On cache miss: fetch from yfinance, save to parquet, return DataFrame

**DataFrame columns**: `date, open, high, low, close, volume` (lowercase, date as index)

**Symbol mapping**: Reuse `YFINANCE_SYMBOLS` dict from `tools/market_data.py` for BTC-USD, DX-Y.NYB, etc.

**Multi-asset loading**: `get_universe(symbols, period, interval) -> dict[str, pd.DataFrame]` — loads all symbols in parallel using `concurrent.futures.ThreadPoolExecutor(max_workers=4)` to respect yfinance rate limits.

---

## 4. Backtest Engine

### Core loop

The engine processes one bar at a time, chronologically. On each bar:

```
for each bar in historical_data[warmup_period:]:
    1. Update prices for all held positions
    2. Check stop-losses (fixed % and ATR-based)
    3. Check take-profits (fixed % and ATR-based)
    4. Check trailing stops (activate, update, trigger)
    5. Check holding period forced exits
    6. Generate new entry signals (if any)
    7. For each signal: size position via RiskManager.calculate_position_size()
    8. Apply entry friction (TradingFriction.total_entry_cost)
    9. Record trade in portfolio
    10. Apply daily borrow cost for shorts
    11. Record equity snapshot
```

### Signal generation (synthetic entries)

Since we are NOT simulating LLM decisions, we need a deterministic signal generator to create realistic entry points. This is the key design decision — the signals must be **good enough** to test exit parameters, but not so fancy that we are also optimizing entry strategy.

**Approach**: Use a simple mean-reversion + trend-following hybrid based on existing indicators:

- **Long signal**: RSI < 35 AND price > SMA(50) AND MACD histogram > 0
- **Short signal**: RSI > 65 AND price < SMA(50) AND MACD histogram < 0
- **Confidence**: Mapped from RSI distance (RSI=20 -> conf=0.9, RSI=35 -> conf=0.6)
- **Rate limiter**: Max 1 signal per asset per 5 bars (avoid over-trading)

This is intentionally simple. The point is to generate a stream of plausible entries so the exit parameter optimization has enough trades to be statistically meaningful. The signal generator is a pluggable class — can be swapped later for replay of actual historical signals from `data/trade_journal.json`.

### `SimulatedPortfolio`

Tracks:
- `equity`: float (starts at initial capital, default $10,000)
- `peak_equity`: float (for drawdown calc)
- `cash`: float
- `positions`: list of `{asset, direction, entry_price, quantity, stop_loss, take_profit, trailing_stop, entry_bar, entry_date}`
- `closed_trades`: list of `{asset, direction, entry_price, exit_price, pnl, pnl_pct, bars_held, exit_reason}`
- `equity_curve`: list of `{date, equity}` (one per bar)
- `daily_pnl`: list of `{date, pnl_pct}` (for Sharpe calculation)

Position exits record an `exit_reason` enum: `stop_loss | take_profit | trailing_stop | holding_period | end_of_data`

Friction is applied on both entry and exit. For shorts held multiple bars, daily borrow cost is deducted from equity.

---

## 5. Parameter Sweeps

### `ParameterSweeper`

1. Takes a sweep mode string (`stops`, `atr-stops`, `trailing`, `sizing`, `holding`, `full`)
2. Generates list of parameter dicts (each is a full `risk_params` override)
3. Runs `BacktestEngine` for each param set
4. Collects results into a DataFrame
5. Sorts by primary metric (default: Sharpe ratio)

**Parallelism**: `concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count-1)` — each backtest is CPU-bound, not IO-bound. Each worker gets its own engine instance (no shared state).

**Progress**: `tqdm` progress bar (already likely installed; if not, simple counter print).

### `full` mode

1. Run `stops` sweep, take top-5 SL/TP combos
2. Run `trailing` sweep, take top-5 trailing combos
3. Run `sizing` sweep, take top-3 sizing combos
4. Cartesian product of top combos: 5 x 5 x 3 = 75 full backtests
5. Report overall best

---

## 6. Output / Reporting

### Metrics calculated per backtest run

| Metric | Formula |
|--------|---------|
| **Sharpe ratio** | mean(daily_returns) / std(daily_returns) * sqrt(252) |
| **Max drawdown** | max peak-to-trough decline in equity curve |
| **Win rate** | winning_trades / total_trades |
| **Profit factor** | gross_profit / gross_loss |
| **Avg win / avg loss** | mean(winning_pnl) / mean(losing_pnl) — the key ratio to fix |
| **Expectancy** | (win_rate * avg_win) - (loss_rate * avg_loss) |
| **Total return** | (final_equity - initial_equity) / initial_equity |
| **Trade count** | Total closed trades |
| **Avg bars held** | Average holding period in bars |
| **Exit reason breakdown** | Count by stop_loss, take_profit, trailing_stop, holding_period |

### Output formats

1. **Console summary**: Top-10 parameter combos ranked by Sharpe, printed as formatted table
2. **CSV export**: `data/backtest_results/{sweep_mode}_{timestamp}.csv` — all combos with all metrics
3. **Best params JSON**: `data/backtest_results/best_params.json` — ready to diff against `config/risk_params.json`
4. **Equity curve plot** (optional, if matplotlib available): `data/backtest_results/equity_{timestamp}.png` for the best combo

### Console output example

```
=== BACKTEST RESULTS: stops sweep (195 combos, 14 assets, 2y) ===

Rank  SL%   TP%   Sharpe  MaxDD   WinRate  PF     AvgW/AvgL  Trades  Return
  1   2.0%  8.0%   1.42   -12.3%   62%    2.14    3.21       847     +34.2%
  2   2.5%  10.0%  1.38   -11.8%   58%    2.31    3.87       723     +31.5%
  3   1.5%  6.0%   1.29   -14.1%   71%    1.82    2.54       1043    +28.7%
 ...
 CURRENT  3.0%  6.0%   0.87   -18.2%   89%    0.92    0.71       912     -4.3%  ← YOU ARE HERE

Best params saved to: data/backtest_results/best_params.json
Full results saved to: data/backtest_results/stops_20260312_143022.csv
```

The "YOU ARE HERE" marker shows current production params for comparison.

---

## 7. CLI Interface

Entry point: `python -m tools.backtest`

```
Usage:
  python -m tools.backtest --sweep stops      # SL% x TP% grid
  python -m tools.backtest --sweep atr-stops  # ATR multiplier grid
  python -m tools.backtest --sweep trailing   # Trailing stop grid
  python -m tools.backtest --sweep sizing     # Position sizing grid
  python -m tools.backtest --sweep holding    # Holding period grid
  python -m tools.backtest --sweep full       # Combined top-N from all sweeps

Options:
  --sweep MODE        Sweep mode (required)
  --period PERIOD     yfinance period string (default: 2y)
  --interval INTERVAL yfinance interval string (default: 1d)
  --capital AMOUNT    Initial portfolio equity in USD (default: 10000)
  --assets SYMBOLS    Comma-separated assets (default: core 14)
  --workers N         Parallel workers (default: cpu_count - 1)
  --output DIR        Output directory (default: data/backtest_results/)
  --no-friction       Disable trading friction simulation
  --plot              Generate equity curve plot for best combo
  --verbose           Print per-trade details for best combo
```

### Examples

```bash
# Quick test: just stops on 3 assets, 6 months
python -m tools.backtest --sweep stops --assets AAPL,NVDA,SPY --period 6mo

# Full 2-year backtest on all core assets
python -m tools.backtest --sweep stops --period 2y

# ATR-based stops with equity curve
python -m tools.backtest --sweep atr-stops --period 1y --plot

# Combined optimization
python -m tools.backtest --sweep full --period 2y --capital 200

# No friction (see raw parameter impact)
python -m tools.backtest --sweep stops --no-friction
```

---

## 8. Estimated Effort

| File | Lines | Complexity | Notes |
|------|-------|-----------|-------|
| `tools/backtest/__init__.py` | 10 | trivial | Exports |
| `tools/backtest/__main__.py` | 80 | low | argparse + orchestration |
| `tools/backtest/data_loader.py` | 120 | low | yfinance + parquet cache |
| `tools/backtest/engine.py` | 300 | **medium** | Core simulation loop, the heart |
| `tools/backtest/portfolio_sim.py` | 200 | medium | Position tracking, equity calc |
| `tools/backtest/param_sweep.py` | 150 | medium | Grid generation, parallel dispatch |
| `tools/backtest/report.py` | 150 | low | Metrics math, formatting |
| `tools/backtest/signals.py` | 100 | low | RSI/MACD/SMA signal rules |
| **Total** | **~1,110** | | |

### Test files

| File | Lines | Coverage |
|------|-------|----------|
| `tests/test_backtest_engine.py` | 200 | Engine loop, stop/TP triggers, trailing stops |
| `tests/test_backtest_portfolio.py` | 100 | SimulatedPortfolio state management |
| `tests/test_backtest_loader.py` | 60 | Cache hit/miss, symbol mapping |
| `tests/test_backtest_report.py` | 80 | Sharpe, drawdown, profit factor math |
| **Total** | **~440** | |

**Grand total**: ~1,550 lines across 12 files.

### Dependencies

- **Already installed**: yfinance, pandas, numpy
- **Needed**: `pyarrow` (for parquet read/write) — `pip install pyarrow`
- **Optional**: `matplotlib` (for equity curve plots), `tqdm` (for progress bars)

---

## 9. Key Design Decisions

**Why synthetic signals instead of replaying real trades?**
We only have ~20 real trades so far — not enough for statistical significance. Synthetic signals generate hundreds of entries per asset over 2 years, giving us the sample size needed to validate exit parameters. Once we have 100+ real trades in `data/trade_journal.json`, we can add a `--signals replay` mode.

**Why sweep one dimension at a time?**
Full grid across all 11 parameters = billions of combinations. Single-dimension sweeps with top-N combination in `full` mode keeps runtime under 10 minutes while still finding the global neighborhood of optimal params.

**Why Parquet for cache?**
OHLCV data for 14 assets x 2 years = ~7,000 rows per asset. Parquet reads in <10ms vs CSV at ~50ms. Disk usage is ~50KB per asset. Total cache: <1MB.

**Why not use the existing `MarketDataFetcher.get_ohlcv()`?**
It returns `list[dict]`, which we would immediately convert to a DataFrame anyway. The data loader fetches raw yfinance DataFrames directly, avoiding the intermediate dict conversion and enabling Parquet caching.

**Why ProcessPoolExecutor instead of ThreadPoolExecutor?**
Each backtest is pure CPU computation (no IO after data is loaded). Python's GIL makes threads useless for CPU-bound work. Processes give true parallelism.
