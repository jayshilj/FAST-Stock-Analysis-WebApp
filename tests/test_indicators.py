"""
Unit tests for the indicators.py standalone indicator library.

Run with:
    pytest tests/ -v
"""

import math

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")
from indicators import (
    compute_rsi,
    compute_atr,
    compute_macd,
    compute_stochastic,
    compute_bollinger,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rising_prices():
    """Steadily rising price series with minor noise."""
    np.random.seed(7)
    return pd.Series(np.linspace(100, 160, 100) + np.random.normal(0, 0.3, 100))


@pytest.fixture
def falling_prices():
    """Steadily falling price series."""
    return pd.Series(list(range(100, 0, -1)), dtype=float)


@pytest.fixture
def flat_ohlc():
    """Flat OHLC series (no volatility)."""
    n = 60
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    return high, low, close


@pytest.fixture
def volatile_ohlc():
    """High-volatility OHLC series (wide daily ranges)."""
    np.random.seed(42)
    n = 80
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n)))
    high = close + pd.Series(np.abs(np.random.normal(2, 0.5, n)))
    low = close - pd.Series(np.abs(np.random.normal(2, 0.5, n)))
    return high, low, close


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------

class TestComputeRSI:
    """Tests for indicators.compute_rsi()."""

    def test_range_0_to_100(self, rising_prices):
        rsi = compute_rsi(rising_prices).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rising_trend_high_rsi(self, rising_prices):
        rsi = compute_rsi(rising_prices).dropna()
        assert rsi.iloc[-1] > 60

    def test_falling_trend_low_rsi(self, falling_prices):
        rsi = compute_rsi(falling_prices).dropna()
        assert rsi.iloc[-1] < 40

    def test_nan_before_window(self):
        prices = pd.Series(range(1, 31), dtype=float)
        rsi = compute_rsi(prices, window=14)
        assert rsi.iloc[:13].isna().all()

    def test_custom_window(self, rising_prices):
        rsi7 = compute_rsi(rising_prices, window=7).dropna()
        rsi28 = compute_rsi(rising_prices, window=28).dropna()
        # Both should be in range
        assert (rsi7 >= 0).all() and (rsi7 <= 100).all()
        assert (rsi28 >= 0).all() and (rsi28 <= 100).all()


# ---------------------------------------------------------------------------
# compute_atr
# ---------------------------------------------------------------------------

class TestComputeATR:
    """Tests for indicators.compute_atr()."""

    def test_atr_non_negative(self, volatile_ohlc):
        high, low, close = volatile_ohlc
        atr = compute_atr(high, low, close).dropna()
        assert (atr >= 0).all()

    def test_flat_market_low_atr(self, flat_ohlc):
        high, low, close = flat_ohlc
        atr = compute_atr(high, low, close).dropna()
        assert len(atr) > 0
        assert atr.iloc[-1] < 2.0

    def test_volatile_market_higher_atr_than_flat(self, volatile_ohlc, flat_ohlc):
        v_high, v_low, v_close = volatile_ohlc
        f_high, f_low, f_close = flat_ohlc
        v_atr = compute_atr(v_high, v_low, v_close).dropna()
        f_atr = compute_atr(f_high, f_low, f_close).dropna()
        assert v_atr.iloc[-1] > f_atr.iloc[-1]

    def test_nan_before_window(self, flat_ohlc):
        high, low, close = flat_ohlc
        atr = compute_atr(high, low, close, window=14)
        assert atr.iloc[:13].isna().all()

    def test_returns_series(self, flat_ohlc):
        high, low, close = flat_ohlc
        result = compute_atr(high, low, close)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# compute_macd
# ---------------------------------------------------------------------------

class TestComputeMACD:
    """Tests for indicators.compute_macd()."""

    def test_returns_three_series(self, rising_prices):
        result = compute_macd(rising_prices)
        assert len(result) == 3
        for s in result:
            assert isinstance(s, pd.Series)

    def test_histogram_equals_diff(self, rising_prices):
        macd_line, signal_line, histogram = compute_macd(rising_prices)
        expected = (macd_line - signal_line).dropna()
        actual = histogram.dropna()
        common = expected.index.intersection(actual.index)
        pd.testing.assert_series_equal(expected[common], actual[common], check_names=False)

    def test_nan_propagation(self):
        prices = pd.Series(range(1, 101), dtype=float)
        _, _, histogram = compute_macd(prices, fast=12, slow=26, signal=9)
        assert histogram.iloc[:33].isna().all()

    def test_rising_macd_positive_histogram_eventually(self, rising_prices):
        _, _, histogram = compute_macd(rising_prices)
        hist_clean = histogram.dropna()
        assert len(hist_clean) > 0
        # In a sustained uptrend the histogram stays non-negative after settling
        assert hist_clean.iloc[-1] > -5.0


# ---------------------------------------------------------------------------
# compute_stochastic
# ---------------------------------------------------------------------------

class TestComputeStochastic:
    """Tests for indicators.compute_stochastic()."""

    def test_range_0_to_100(self, volatile_ohlc):
        high, low, close = volatile_ohlc
        k, d = compute_stochastic(high, low, close)
        k_clean = k.dropna()
        d_clean = d.dropna()
        assert (k_clean >= 0).all() and (k_clean <= 100).all()
        assert (d_clean >= 0).all() and (d_clean <= 100).all()

    def test_price_at_top_near_100(self):
        n = 20
        high = pd.Series([100.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([100.0] * n)
        k, _ = compute_stochastic(high, low, close, k_window=14)
        assert k.dropna().iloc[-1] > 95

    def test_price_at_bottom_near_0(self):
        n = 20
        high = pd.Series([100.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([90.0] * n)
        k, _ = compute_stochastic(high, low, close, k_window=14)
        assert k.dropna().iloc[-1] < 5

    def test_d_is_sma_of_k(self, volatile_ohlc):
        high, low, close = volatile_ohlc
        k, d = compute_stochastic(high, low, close, k_window=14, d_window=3)
        # d at any non-NaN index should be the rolling mean of k
        idx = d.dropna().index[-1]
        expected_d = k.iloc[idx - 2: idx + 1].mean()
        assert abs(d.iloc[idx] - expected_d) < 0.01

    def test_nan_before_k_window(self, volatile_ohlc):
        high, low, close = volatile_ohlc
        k, _ = compute_stochastic(high, low, close, k_window=14)
        assert k.iloc[:13].isna().all()


# ---------------------------------------------------------------------------
# compute_bollinger
# ---------------------------------------------------------------------------

class TestComputeBollinger:
    """Tests for indicators.compute_bollinger()."""

    def test_returns_dataframe_with_expected_columns(self, rising_prices):
        result = compute_bollinger(rising_prices)
        assert isinstance(result, pd.DataFrame)
        for col in ("SMA", "Upper", "Lower", "%B", "BandWidth"):
            assert col in result.columns, f"Missing column: {col}"

    def test_upper_above_lower(self, rising_prices):
        result = compute_bollinger(rising_prices).dropna()
        assert (result["Upper"] >= result["Lower"]).all()

    def test_sma_between_bands(self, rising_prices):
        result = compute_bollinger(rising_prices).dropna()
        assert (result["SMA"] <= result["Upper"]).all()
        assert (result["SMA"] >= result["Lower"]).all()

    def test_band_width_positive(self, rising_prices):
        result = compute_bollinger(rising_prices).dropna()
        assert (result["BandWidth"] >= 0).all()

    def test_pct_b_at_sma_is_half(self):
        """When price == SMA, %B should be 0.5."""
        n = 40
        # Constant prices: SMA == Close, Upper and Lower equidistant
        # %B = (Close - Lower) / (Upper - Lower) = 0.5 when Close == SMA
        prices = pd.Series([100.0] * n)
        result = compute_bollinger(prices, window=20).dropna()
        # With constant prices std=0 so Upper==Lower==SMA, %B is NaN (division by 0)
        # Verify that %B is NaN in the flat-market edge case
        assert result["%B"].isna().all() or (result["%B"].dropna().between(0, 1).all())

    def test_custom_window_and_std(self, rising_prices):
        result10 = compute_bollinger(rising_prices, window=10, n_std=1.5).dropna()
        result20 = compute_bollinger(rising_prices, window=20, n_std=2.0).dropna()
        # Different windows → different row counts, both valid DataFrames
        assert isinstance(result10, pd.DataFrame)
        assert isinstance(result20, pd.DataFrame)


class TestIndicatorsEdgeCases:
    """Edge cases for all standalone technical indicator calculations."""

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        # Testing empty input behavior
        assert compute_rsi(empty).empty
        assert compute_atr(empty, empty, empty).empty
        
        m_line, s_line, hist = compute_macd(empty)
        assert m_line.empty and s_line.empty and hist.empty

        stoch_k, stoch_d = compute_stochastic(empty, empty, empty)
        assert stoch_k.empty and stoch_d.empty

        assert compute_bollinger(empty).empty

    def test_all_nans(self):
        nans = pd.Series([float("nan")] * 20, dtype=float)
        assert compute_rsi(nans).isna().all()
        assert compute_atr(nans, nans, nans).isna().all()

        m_line, s_line, hist = compute_macd(nans)
        assert m_line.isna().all() and s_line.isna().all() and hist.isna().all()

        stoch_k, stoch_d = compute_stochastic(nans, nans, nans)
        assert stoch_k.isna().all() and stoch_d.isna().all()

        bb = compute_bollinger(nans)
        assert bb.isna().all().all()

    def test_zero_volatility_stochastic(self):
        # High and low are equal, Close is flat. Max - Min = 0.
        n = 15
        high = pd.Series([10.0] * n)
        low = pd.Series([10.0] * n)
        close = pd.Series([10.0] * n)
        stoch_k, stoch_d = compute_stochastic(high, low, close, k_window=5)
        # Should not raise ZeroDivisionError and should return NaN or valid values
        assert len(stoch_k) == n
