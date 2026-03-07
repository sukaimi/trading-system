"""Technical indicator calculations using numpy.

All indicators are pure functions operating on price arrays.
No external API calls. No ta-lib dependency.
"""

from __future__ import annotations

import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for trading analysis."""

    def rsi(self, prices: list[float], period: int = 14) -> float:
        """Calculate Relative Strength Index using Wilder's smoothing.

        Args:
            prices: Closing prices, oldest first. Needs >= period + 1 values.
            period: RSI lookback period (default 14).

        Returns:
            RSI value 0-100. Returns 50.0 if insufficient data.
        """
        if len(prices) < period + 1:
            return 50.0

        arr = np.array(prices, dtype=float)
        deltas = np.diff(arr)

        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder's smoothing: first average is SMA, then EMA with alpha=1/period
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def macd(self, prices: list[float]) -> dict:
        """Calculate MACD (12/26/9).

        Returns:
            {"macd_line": float, "signal_line": float, "histogram": float}
            Returns zeroes if insufficient data (need >= 35 prices).
        """
        if len(prices) < 35:
            return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)

        # MACD line = EMA(12) - EMA(26), aligned to shorter array
        min_len = min(len(ema12), len(ema26))
        macd_line = [
            ema12[len(ema12) - min_len + i] - ema26[len(ema26) - min_len + i]
            for i in range(min_len)
        ]

        if len(macd_line) < 9:
            return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

        signal_line = self._ema(macd_line, 9)

        ml = macd_line[-1]
        sl = signal_line[-1]
        return {
            "macd_line": float(ml),
            "signal_line": float(sl),
            "histogram": float(ml - sl),
        }

    def bollinger_bands(
        self, prices: list[float], period: int = 20, std: float = 2.0
    ) -> dict:
        """Calculate Bollinger Bands.

        Returns:
            {"upper": float, "middle": float, "lower": float, "bandwidth": float}
            Returns zeroes if insufficient data.
        """
        if len(prices) < period:
            return {"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0}

        window = np.array(prices[-period:], dtype=float)
        middle = float(np.mean(window))
        stdev = float(np.std(window, ddof=1)) if period > 1 else 0.0

        upper = middle + std * stdev
        lower = middle - std * stdev
        bandwidth = (upper - lower) / middle if middle != 0 else 0.0

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
        }

    def atr(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float:
        """Calculate Average True Range using Wilder's smoothing.

        Args:
            highs/lows/closes: OHLC data, same length, oldest first.
            period: ATR lookback (default 14).

        Returns:
            ATR value. Returns 0.0 if insufficient data.
        """
        n = min(len(highs), len(lows), len(closes))
        if n < period + 1:
            return 0.0

        h = np.array(highs[:n], dtype=float)
        l = np.array(lows[:n], dtype=float)
        c = np.array(closes[:n], dtype=float)

        # True Range for each bar (starting from index 1)
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )

        if len(tr) < period:
            return 0.0

        # Wilder's smoothing
        atr_val = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr_val = (atr_val * (period - 1) + tr[i]) / period

        return float(atr_val)

    def sma(self, prices: list[float], period: int) -> float:
        """Calculate Simple Moving Average of the last `period` prices.

        Returns 0.0 if insufficient data.
        """
        if len(prices) < period or period <= 0:
            return 0.0
        return float(np.mean(prices[-period:]))

    def adx(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float:
        """Calculate Average Directional Index using Wilder's smoothing.

        ADX measures trend strength regardless of direction.
        > 25 = trending, < 20 = ranging/weak trend.

        Args:
            highs/lows/closes: OHLC data, same length, oldest first.
            period: ADX lookback (default 14).

        Returns:
            ADX value 0-100. Returns 0.0 if insufficient data.
        """
        n = min(len(highs), len(lows), len(closes))
        if n < period * 2 + 1:
            return 0.0

        h = np.array(highs[:n], dtype=float)
        l = np.array(lows[:n], dtype=float)
        c = np.array(closes[:n], dtype=float)

        # Directional movement
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # True Range
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l[1:] - c[:-1]),
            ),
        )

        if len(tr) < period:
            return 0.0

        # Wilder's smoothing for TR, +DM, -DM
        atr_val = float(np.sum(tr[:period]))
        plus_dm_smooth = float(np.sum(plus_dm[:period]))
        minus_dm_smooth = float(np.sum(minus_dm[:period]))

        # First DI values
        plus_di_series = []
        minus_di_series = []

        if atr_val > 0:
            plus_di_series.append(100.0 * plus_dm_smooth / atr_val)
            minus_di_series.append(100.0 * minus_dm_smooth / atr_val)
        else:
            plus_di_series.append(0.0)
            minus_di_series.append(0.0)

        for i in range(period, len(tr)):
            atr_val = atr_val - (atr_val / period) + tr[i]
            plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm[i]
            minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm[i]

            if atr_val > 0:
                plus_di_series.append(100.0 * plus_dm_smooth / atr_val)
                minus_di_series.append(100.0 * minus_dm_smooth / atr_val)
            else:
                plus_di_series.append(0.0)
                minus_di_series.append(0.0)

        # Calculate DX from DI values
        dx_series = []
        for pdi, mdi in zip(plus_di_series, minus_di_series):
            di_sum = pdi + mdi
            if di_sum > 0:
                dx_series.append(100.0 * abs(pdi - mdi) / di_sum)
            else:
                dx_series.append(0.0)

        if len(dx_series) < period:
            return 0.0

        # ADX = Wilder's smoothed DX
        adx_val = float(np.mean(dx_series[:period]))
        for i in range(period, len(dx_series)):
            adx_val = (adx_val * (period - 1) + dx_series[i]) / period

        return float(adx_val)

    def sma_series(self, prices: list[float], period: int) -> list[float]:
        """Calculate SMA series for the given period.

        Returns list of SMA values, one per bar starting from index period-1.
        Returns empty list if insufficient data.
        """
        if len(prices) < period or period <= 0:
            return []
        arr = np.array(prices, dtype=float)
        # Cumulative sum trick for efficient rolling mean
        cumsum = np.cumsum(arr)
        cumsum = np.insert(cumsum, 0, 0)
        sma_vals = (cumsum[period:] - cumsum[:-period]) / period
        return [float(v) for v in sma_vals]

    def _ema(self, prices: list[float], period: int) -> list[float]:
        """Calculate Exponential Moving Average series.

        Uses standard EMA with multiplier = 2 / (period + 1).
        Returns the full EMA series (same length as input minus period + 1).
        """
        if len(prices) < period:
            return []

        arr = np.array(prices, dtype=float)
        multiplier = 2.0 / (period + 1)

        # Seed with SMA of first `period` values
        ema = [float(np.mean(arr[:period]))]
        for price in arr[period:]:
            ema.append(float(price * multiplier + ema[-1] * (1 - multiplier)))

        return ema
