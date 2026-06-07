"""indicators.py – Pure-Python technical indicator library for FAST.

This module exposes standalone, stateless indicator functions that work on
pandas Series / DataFrames and have no Streamlit dependency.  Each function
mirrors the logic used inline in ``app.py`` and is covered by the unit-test
suite in ``tests/test_indicators.py``.

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

compute_sma(prices, window)
    Simple Moving Average.

compute_ema(prices, window)
    Exponential Moving Average using EWM with span.

compute_vwap(high, low, close, volume)
    Volume Weighted Average Price (cumulative).

compute_obv(close, volume)
    On-Balance Volume (cumulative directional volume).

compute_cci(high, low, close, window, constant)
    Commodity Channel Index measuring deviation from rolling mean.
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


def compute_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Compute the Simple Moving Average (SMA).

    Args:
        prices: Series of prices.
        window: Rolling window period in days (default 20).

    Returns:
        pandas Series of SMA values. First ``window - 1`` values are ``NaN``.
    """
    return prices.rolling(window=window).mean()


def compute_ema(prices: pd.Series, window: int = 20) -> pd.Series:
    """Compute the Exponential Moving Average (EMA).

    Args:
        prices: Series of prices.
        window: Rolling window period in days (default 20).

    Returns:
        pandas Series of EMA values. First ``window - 1`` values are ``NaN``.
    """
    return prices.ewm(span=window, min_periods=window).mean()


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute the Volume Weighted Average Price (VWAP).

    VWAP is the cumulative sum of (Typical Price x Volume) divided by the
    cumulative sum of Volume.  It represents the average price weighted by
    trading activity and is widely used as a fair-value benchmark.

    Typical Price = (High + Low + Close) / 3

    Args:
        high:   Series of daily high prices.
        low:    Series of daily low prices.
        close:  Series of daily closing prices.
        volume: Series of daily trading volumes.

    Returns:
        pandas Series of VWAP values (same index as inputs).
        Returns NaN for any row where cumulative volume is zero.
    """
    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical_price * volume).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, float("nan"))
    return vwap


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    OBV is a cumulative volume momentum indicator.  On a day when the closing
    price rises above the previous close, volume is added; on a down day,
    volume is subtracted; on a flat day, OBV is unchanged.

    Args:
        close:  Series of daily closing prices.
        volume: Series of daily trading volumes.

    Returns:
        pandas Series of OBV values starting at 0 (same index as inputs).
    """
    direction = close.diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    direction.iloc[0] = 0  # First day has no prior close
    obv = (direction * volume).cumsum()
    return obv


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """Compute the Commodity Channel Index (CCI).

    CCI measures how far the Typical Price deviates from its rolling mean,
    normalised by the mean absolute deviation (MAD) times a scaling constant.

    CCI = (Typical Price - SMA(TP)) / (constant * MAD)

    Values above +100 are traditionally considered overbought; below -100,
    oversold.

    Args:
        high:     Series of daily high prices.
        low:      Series of daily low prices.
        close:    Series of daily closing prices.
        window:   Rolling look-back period (default 20).
        constant: Lambert constant for normalisation (default 0.015).

    Returns:
        pandas Series of CCI values.  First ``window - 1`` values are ``NaN``.
    """
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(window).mean()
    mad = typical_price.rolling(window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma_tp) / (constant * mad.replace(0, float("nan")))
    return cci
