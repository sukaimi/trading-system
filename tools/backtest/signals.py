"""Synthetic signal generator for backtesting.

Generates deterministic entry signals based on RSI + SMA + MACD indicators.
This is intentionally simple — the goal is to produce enough plausible entries
so exit parameter optimization has statistical significance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tools.technical_indicators import TechnicalIndicators


@dataclass
class BacktestSignal:
    """A synthetic entry signal."""

    asset: str
    bar_index: int
    direction: str  # "long" or "short"
    confidence: float  # 0.0 - 1.0
    price: float
    date: str


class SignalGenerator:
    """Generate synthetic entry signals from RSI + SMA + MACD.

    Long signal: RSI < 35 AND price > SMA(50) AND MACD histogram > 0
    Short signal: RSI > 65 AND price < SMA(50) AND MACD histogram < 0
    Rate limiter: Max 1 signal per asset per 5 bars.
    """

    def __init__(self, rsi_long_threshold: float = 35.0, rsi_short_threshold: float = 65.0,
                 sma_period: int = 50, cooldown_bars: int = 5):
        self._ti = TechnicalIndicators()
        self._rsi_long = rsi_long_threshold
        self._rsi_short = rsi_short_threshold
        self._sma_period = sma_period
        self._cooldown = cooldown_bars

    def generate(self, asset: str, df: pd.DataFrame, warmup: int = 50) -> list[BacktestSignal]:
        """Generate entry signals for a single asset over its full history.

        Args:
            asset: Symbol name.
            df: OHLCV DataFrame with columns: open, high, low, close, volume.
            warmup: Minimum bars before first signal (for indicator warmup).

        Returns:
            List of BacktestSignal objects.
        """
        signals: list[BacktestSignal] = []
        closes = df["close"].tolist()
        dates = [str(d) for d in df.index]

        if len(closes) < warmup:
            return signals

        last_signal_bar = -self._cooldown  # Allow signal on first eligible bar

        for i in range(warmup, len(closes)):
            # Rate limiter
            if i - last_signal_bar < self._cooldown:
                continue

            # Calculate indicators using data up to current bar
            window = closes[:i + 1]

            rsi = self._ti.rsi(window, period=14)
            sma = self._ti.sma(window, self._sma_period)
            macd = self._ti.macd(window)

            if sma == 0.0:
                continue

            current_price = closes[i]
            histogram = macd["histogram"]

            # Long signal
            if rsi < self._rsi_long and current_price > sma and histogram > 0:
                confidence = self._rsi_to_confidence(rsi, direction="long")
                signals.append(BacktestSignal(
                    asset=asset,
                    bar_index=i,
                    direction="long",
                    confidence=confidence,
                    price=current_price,
                    date=dates[i],
                ))
                last_signal_bar = i

            # Short signal
            elif rsi > self._rsi_short and current_price < sma and histogram < 0:
                confidence = self._rsi_to_confidence(rsi, direction="short")
                signals.append(BacktestSignal(
                    asset=asset,
                    bar_index=i,
                    direction="short",
                    confidence=confidence,
                    price=current_price,
                    date=dates[i],
                ))
                last_signal_bar = i

        return signals

    @staticmethod
    def _rsi_to_confidence(rsi: float, direction: str) -> float:
        """Map RSI distance from threshold to confidence score.

        Long: RSI=20 -> 0.9, RSI=35 -> 0.6
        Short: RSI=80 -> 0.9, RSI=65 -> 0.6
        """
        if direction == "long":
            # Linear interpolation: RSI 20->0.9, RSI 35->0.6
            conf = 0.6 + (35.0 - rsi) / 15.0 * 0.3
        else:
            # Linear interpolation: RSI 65->0.6, RSI 80->0.9
            conf = 0.6 + (rsi - 65.0) / 15.0 * 0.3

        return min(max(conf, 0.5), 0.95)
