"""Market data retrieval tools.

Phase 2 implementation. Placeholder stubs for:
- yfinance price/volume data
- Alpha Vantage technical indicators
- IBKR real-time quotes
- CoinGecko crypto prices
"""


class MarketDataFetcher:
    """Fetch market data from multiple sources."""

    def get_price(self, symbol: str) -> dict:
        """Get current price for a symbol."""
        raise NotImplementedError("Phase 2 — yfinance/CoinGecko price fetch")

    def get_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "1d") -> list[dict]:
        """Get OHLCV data for a symbol."""
        raise NotImplementedError("Phase 2 — yfinance OHLCV")

    def get_realtime_quote(self, symbol: str) -> dict:
        """Get real-time quote from IBKR."""
        raise NotImplementedError("Phase 2 — IBKR real-time quotes")
