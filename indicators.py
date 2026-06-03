"""indicators.py – Pure-Python technical indicator library for FAST.

This module exposes standalone, stateless indicator functions that work on
pandas Series / DataFrames and have no Streamlit dependency.  Each function
mirrors the logic used inline in ``app.py`` and is covered by the unit-test
suite in ``tests/test_helpers.py``.

Indicator functions
-------------------
compute_rsi(prices, window)
    Relative Strength Index using Wilder EWM smoothing.

compute_atr(high, low, close, window)
    Average True Range measuring rolling volatility.

compute_macd(prices, fast, slow, signal)
    MACD line, signal line, and histogram.

compute_stochastic(high, low, close, k_window, d_window)
    Stochastic oscillator %K and %D lines.

compute_bollinger(prices, window, n_std)
    Bollinger Bands (upper, middle/SMA, lower, %B, band-width).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Uses exponential weighted moving averages (EWM) with ``com=window-1``
    to match the industry-standard Wilder smoothing method.

    Args:
        prices: A pandas Series of closing prices.
        window: Look-back period in days (default 14).

    Returns:
        pandas Series of RSI values in the range [0, 100].
        The first ``window - 1`` values will be ``NaN``.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute the Average True Range (ATR).

    True Range is the maximum of:
    - ``High - Low``
    - ``|High - Previous Close|``
    - ``|Low - Previous Close|``

    ATR is the EWM of True Range with ``span=window``.

    Args:
        high:   Series of daily high prices.
        low:    Series of daily low prices.
        close:  Series of daily closing prices.
        window: Smoothing window in days (default 14).

    Returns:
        pandas Series of ATR values. First ``window - 1`` values are ``NaN``.
    """
    prev_close = close.shift(1)
    df = pd.DataFrame({"High": high, "Low": low, "prev_close": prev_close})
    true_range = df.apply(
        lambda r: max(
            r["High"] - r["Low"],
            abs(r["High"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0.0,
            abs(r["Low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0.0,
        ),
        axis=1,
    )
    return true_range.ewm(span=window, min_periods=window).mean()


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Args:
        prices: Series of closing prices.
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        (macd_line, signal_line, histogram) – three pandas Series.
        Histogram = MACD line − Signal line.
    """
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Compute the Stochastic Oscillator %K and %D.

    %K = 100 × (Close − Lowest Low) / (Highest High − Lowest Low)
    %D = Simple moving average of %K over ``d_window`` periods.

    Args:
        high:     Series of daily high prices.
        low:      Series of daily low prices.
        close:    Series of daily closing prices.
        k_window: Look-back period for %K (default 14).
        d_window: Smoothing period for %D (default 3).

    Returns:
        (pct_k, pct_d) – two pandas Series in the range [0, 100].
    """
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    pct_k = (
        100
        * (close - lowest_low)
        / (highest_high - lowest_low).replace(0, float("nan"))
    )
    pct_d = pct_k.rolling(d_window).mean()
    return pct_k, pct_d


def compute_bollinger(
    prices: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands, %B, and Band Width.

    Args:
        prices: Series of closing prices.
        window: Rolling window period (default 20).
        n_std:  Number of standard deviations for band width (default 2).

    Returns:
        DataFrame with columns:
            - ``SMA``       – Middle band (simple moving average)
            - ``Upper``     – Upper band (SMA + n_std × σ)
            - ``Lower``     – Lower band (SMA − n_std × σ)
            - ``%B``        – Percent B: position of price within the bands [0, 1]
            - ``BandWidth`` – (Upper − Lower) / SMA × 100 (%)
    """
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_range = (upper - lower).replace(0, float("nan"))
    pct_b = (prices - lower) / band_range
    band_width = band_range / sma * 100
    return pd.DataFrame(
        {
            "SMA": sma,
            "Upper": upper,
            "Lower": lower,
            "%B": pct_b,
            "BandWidth": band_width,
        }
    )
