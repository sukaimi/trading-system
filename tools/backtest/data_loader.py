"""Historical data loader with yfinance + Parquet disk cache.

Fetches OHLCV data from yfinance and caches to Parquet files for fast
subsequent reads. Cache TTL is 24 hours.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# Reuse yfinance symbol mapping from existing tools
YFINANCE_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

# Core 14 assets
CORE_ASSETS = [
    "BTC", "ETH", "GLDM", "SLV", "AAPL", "NVDA", "TSLA", "AMZN",
    "SPY", "META", "TLT", "XLE", "EWS", "FXI",
]

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


class HistoricalDataLoader:
    """Load OHLCV data from yfinance with Parquet disk cache."""

    def __init__(self, cache_dir: Path | str | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, period: str, interval: str) -> Path:
        safe_symbol = symbol.replace("-", "_").replace(".", "_").replace("^", "_")
        return self._cache_dir / f"{safe_symbol}_{period}_{interval}.parquet"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < CACHE_TTL_SECONDS

    def get_historical(
        self, symbol: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Returns DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex named 'date'.
        """
        cache_path = self._cache_path(symbol, period, interval)

        if self._is_cache_valid(cache_path):
            df = pd.read_parquet(cache_path)
            if not df.empty:
                return df

        # Fetch from yfinance
        import yfinance as yf

        yf_sym = YFINANCE_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned from yfinance for {symbol} ({yf_sym})")

        # Normalize column names to lowercase
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df.index.name = "date"

        # Keep only OHLCV columns
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols]

        # Save to cache
        try:
            df.to_parquet(cache_path)
        except Exception:
            pass  # Non-fatal — cache is optional

        return df

    def get_universe(
        self,
        symbols: list[str] | None = None,
        period: str = "2y",
        interval: str = "1d",
        max_workers: int = 4,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple symbols in parallel.

        Returns dict mapping symbol -> DataFrame.
        Symbols that fail to load are silently skipped.
        """
        if symbols is None:
            symbols = CORE_ASSETS

        results: dict[str, pd.DataFrame] = {}

        def _load(sym: str) -> tuple[str, pd.DataFrame | None]:
            try:
                return sym, self.get_historical(sym, period, interval)
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sym, df in pool.map(lambda s: _load(s), symbols):
                if df is not None and not df.empty:
                    results[sym] = df

        return results
