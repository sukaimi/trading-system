"""Tests for tools/market_data.py — all external calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from tools.market_data import MarketDataFetcher


@pytest.fixture
def mdf():
    return MarketDataFetcher()


class TestGetPrice:
    @patch("tools.market_data.requests.get")
    def test_coingecko_success(self, mock_get, mdf):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"bitcoin": {"usd": 67000.0}}
        mock_get.return_value = mock_resp

        result = mdf.get_price("BTC")
        assert result["price"] == 67000.0
        assert result["source"] == "coingecko"

    @patch("tools.market_data.requests.get")
    @patch("tools.market_data.yf.Ticker")
    def test_coingecko_failure_yfinance_fallback(self, mock_yf, mock_get, mdf):
        # CoinGecko fails
        mock_get.side_effect = Exception("timeout")

        # yfinance works
        mock_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.__getitem__ = MagicMock(return_value=MagicMock(iloc=MagicMock(__getitem__=MagicMock(return_value=66000.0))))
        mock_ticker.history.return_value = mock_hist
        mock_yf.return_value = mock_ticker

        result = mdf.get_price("BTC")
        assert result["price"] == 66000.0
        assert result["source"] == "yfinance"

    @patch("tools.market_data.requests.get")
    @patch("tools.market_data.yf.Ticker")
    def test_all_sources_fail(self, mock_yf, mock_get, mdf):
        mock_get.side_effect = Exception("fail")
        mock_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = True
        mock_ticker.history.return_value = mock_hist
        mock_yf.return_value = mock_ticker

        result = mdf.get_price("BTC")
        assert result["price"] == 0.0
        assert "error" in result

    @patch("tools.market_data.yf.Ticker")
    def test_etf_uses_yfinance(self, mock_yf, mdf):
        mock_ticker = MagicMock()
        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.__getitem__ = MagicMock(return_value=MagicMock(iloc=MagicMock(__getitem__=MagicMock(return_value=45.0))))
        mock_ticker.history.return_value = mock_hist
        mock_yf.return_value = mock_ticker

        result = mdf.get_price("GLDM")
        assert result["price"] == 45.0


class TestGetOHLCV:
    @patch("tools.market_data.yf.Ticker")
    def test_empty_dataframe(self, mock_yf, mdf):
        mock_ticker = MagicMock()
        mock_df = MagicMock()
        mock_df.empty = True
        mock_ticker.history.return_value = mock_df
        mock_yf.return_value = mock_ticker

        result = mdf.get_ohlcv("BTC")
        assert result == []

    @patch("tools.market_data.yf.Ticker")
    def test_exception_returns_empty(self, mock_yf, mdf):
        mock_yf.side_effect = Exception("network error")
        result = mdf.get_ohlcv("BTC")
        assert result == []


class TestGetMarketContext:
    @patch.object(MarketDataFetcher, "get_price")
    def test_returns_all_keys(self, mock_price, mdf):
        mock_price.return_value = {"price": 100.0}
        ctx = mdf.get_market_context()
        assert "dxy" in ctx
        assert "vix" in ctx
        assert "btc_price" in ctx
