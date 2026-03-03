"""Market data retrieval from yfinance and CoinGecko.

yfinance: ETFs (GLDM, SLV) and indices (DXY, VIX)
CoinGecko: Crypto prices (BTC, ETH) — free, no API key required
All methods return dicts, never raise on failure.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import requests
import yfinance as yf

from core.logger import setup_logger

log = setup_logger("trading.market_data")

YFINANCE_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GLDM": "GLDM",
    "SLV": "SLV",
    "AAPL": "AAPL",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "AMZN": "AMZN",
    "SPY": "SPY",
    "META": "META",
    "EWS": "EWS",
    "FXI": "FXI",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum"}
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class MarketDataFetcher:
    """Fetch market data from yfinance and CoinGecko."""

    def __init__(self):
        self._last_cg_call: float = 0.0
        self._cg_interval: float = 2.0  # seconds between CoinGecko calls

    def get_price(self, symbol: str) -> dict[str, Any]:
        """Get current price for a symbol.

        Returns:
            {"symbol": str, "price": float, "currency": "USD",
             "source": str, "timestamp": str}
        """
        # Try CoinGecko for crypto
        if symbol in COINGECKO_IDS:
            price = self._fetch_coingecko_price(COINGECKO_IDS[symbol])
            if price is not None:
                return {
                    "symbol": symbol,
                    "price": price,
                    "currency": "USD",
                    "source": "coingecko",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        # Fallback to yfinance for all symbols
        yf_sym = YFINANCE_SYMBOLS.get(symbol, symbol)
        price = self._fetch_yfinance_price(yf_sym)
        if price is not None:
            return {
                "symbol": symbol,
                "price": price,
                "currency": "USD",
                "source": "yfinance",
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "symbol": symbol,
            "price": 0.0,
            "error": f"Failed to fetch price for {symbol}",
        }

    def get_ohlcv(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get OHLCV data via yfinance.

        Returns list of {"date", "open", "high", "low", "close", "volume"} dicts.
        Returns empty list on failure.
        """
        yf_sym = YFINANCE_SYMBOLS.get(symbol, symbol)
        try:
            ticker = yf.Ticker(yf_sym)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                log.warning("No OHLCV data for %s", symbol)
                return []

            result = []
            for date, row in df.iterrows():
                result.append({
                    "date": str(date),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0)),
                })
            return result
        except Exception as e:
            log.warning("OHLCV fetch failed for %s: %s", symbol, e)
            return []

    def get_market_context(self) -> dict[str, Any]:
        """Fetch current market context for journaling."""
        context: dict[str, Any] = {}
        for symbol in ["DXY", "VIX", "BTC", "ETH", "GLDM", "SLV"]:
            data = self.get_price(symbol)
            key = symbol.lower() if symbol in ("DXY", "VIX") else f"{symbol.lower()}_price"
            context[key] = data.get("price", 0.0)
        return context

    def _fetch_coingecko_price(self, coin_id: str) -> float | None:
        """Fetch price from CoinGecko with rate limiting."""
        now = time.time()
        wait = self._cg_interval - (now - self._last_cg_call)
        if wait > 0:
            time.sleep(wait)

        try:
            resp = requests.get(
                f"{COINGECKO_BASE}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10,
            )
            self._last_cg_call = time.time()
            if resp.status_code == 200:
                data = resp.json()
                return data.get(coin_id, {}).get("usd")
        except Exception as e:
            log.warning("CoinGecko fetch failed for %s: %s", coin_id, e)
        return None

    def _fetch_yfinance_price(self, yf_symbol: str) -> float | None:
        """Fetch latest price from yfinance."""
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            log.warning("yfinance fetch failed for %s: %s", yf_symbol, e)
        return None
