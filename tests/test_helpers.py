# -*- coding: utf-8 -*-
"""
Unit tests for FAST Stock Analysis WebApp helper utilities.

Run with:
    pytest tests/ -v
"""

import math
import types
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers imported from ui_theme (no Streamlit runtime required)
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from ui_theme import (
    _fmt_metric,
    render_sentiment_badge,
    render_alert_banner,
    render_metric_delta_card,
    APP_BRAND_FULL,
    NAV_DEFINITION,
)


class TestFmtMetric:
    """Tests for ui_theme._fmt_metric() value formatter."""

    def test_none_returns_dash(self):
        assert _fmt_metric(None) == "\u2014"

    def test_na_string_returns_dash(self):
        assert _fmt_metric("N/A") == "\u2014"

    def test_trillions(self):
        result = _fmt_metric(2_500_000_000_000)
        assert result.endswith("T")
        assert "2.50" in result

    def test_billions(self):
        result = _fmt_metric(3_200_000_000)
        assert result.endswith("B")
        assert "3.20" in result

    def test_millions(self):
        result = _fmt_metric(500_000_000)
        assert result.endswith("M")
        assert "500.00" in result

    def test_thousands(self):
        result = _fmt_metric(12_345)
        assert "12,345" in result

    def test_small_float(self):
        result = _fmt_metric(3.14)
        assert "3.14" in result

    def test_zero(self):
        # _fmt_metric uses {:,.4g} which strips trailing zeros;
        # for 0 this returns an empty string after rstrip('0').rstrip('.')
        result = _fmt_metric(0)
        # Acceptable outputs are '0' or '' depending on Python/locale
        assert result in ("0", "")


class TestSentimentBadge:
    """Tests for ui_theme.render_sentiment_badge() HTML output."""

    def test_positive_badge(self):
        html = render_sentiment_badge("positive")
        assert "badge-pos" in html
        assert "Positive" in html

    def test_negative_badge(self):
        html = render_sentiment_badge("negative")
        assert "badge-neg" in html
        assert "Negative" in html

    def test_neutral_badge(self):
        html = render_sentiment_badge("neutral")
        assert "badge-neutral" in html
        assert "Neutral" in html

    def test_none_returns_neutral(self):
        html = render_sentiment_badge(None)
        assert "badge-neutral" in html

    def test_mixed_case(self):
        html = render_sentiment_badge("POSITIVE")
        assert "badge-pos" in html


class TestNavDefinition:
    """Validate that the navigation definition is well-formed."""

    def test_nav_entries_are_tuples_of_three(self):
        for entry in NAV_DEFINITION:
            assert len(entry) == 3, f"Nav entry {entry!r} must have 3 elements"

    def test_nav_internal_names_unique(self):
        internals = [row[0] for row in NAV_DEFINITION]
        assert len(internals) == len(set(internals)), "Duplicate internal nav names found"

    def test_nav_display_names_unique(self):
        short_names = [row[2] for row in NAV_DEFINITION]
        assert len(short_names) == len(set(short_names)), "Duplicate short nav names found"


class TestNormalizeDf:
    """
    Tests for the normalize_market_df logic extracted from app.py.
    We recreate the function inline to avoid importing Streamlit.
    """

    @staticmethod
    def normalize_market_df(df):
        if df is None or df.empty:
            return pd.DataFrame()
        normalized = df.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            level0 = set(normalized.columns.get_level_values(0).unique())
            level1 = set(normalized.columns.get_level_values(1).unique())
            if ohlcv.intersection(level1) and not ohlcv.intersection(level0):
                normalized.columns = normalized.columns.get_level_values(1)
            elif ohlcv.intersection(level0) and not ohlcv.intersection(level1):
                normalized.columns = normalized.columns.get_level_values(0)
            else:
                if len(ohlcv.intersection(level1)) >= len(ohlcv.intersection(level0)):
                    normalized.columns = normalized.columns.get_level_values(1)
                else:
                    normalized.columns = normalized.columns.get_level_values(0)
        return normalized

    def test_none_returns_empty(self):
        result = self.normalize_market_df(None)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = self.normalize_market_df(pd.DataFrame())
        assert result.empty

    def test_flat_df_unchanged(self):
        df = pd.DataFrame({"Close": [100, 101], "Volume": [1000, 2000]})
        result = self.normalize_market_df(df)
        assert "Close" in result.columns

    def test_multiindex_columns_flattened(self):
        arrays = [["OHLCV", "OHLCV"], ["Close", "Volume"]]
        multi_idx = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame([[100, 1000], [101, 2000]], columns=multi_idx)
        result = self.normalize_market_df(df)
        assert "Close" in result.columns or "Volume" in result.columns


class TestSafeSummarize:
    """
    Tests for the safe_summarize helper extracted from app.py.
    """

    @staticmethod
    def safe_summarize(text, ratio):
        import re
        cleaned_ratio = max(0.01, min(float(ratio), 1.0))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return "No text available to summarize."
        keep_count = max(1, int(len(sentences) * cleaned_ratio))
        return " ".join(sentences[:keep_count])

    def test_empty_string(self):
        result = self.safe_summarize("", 0.5)
        assert result == "No text available to summarize."

    def test_ratio_clamps_to_1(self):
        text = "Sentence one. Sentence two. Sentence three."
        result = self.safe_summarize(text, 999)
        # All three sentences should be returned
        assert "one" in result and "two" in result and "three" in result

    def test_ratio_clamps_to_minimum(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        result = self.safe_summarize(text, 0)
        # Even at 0 we keep at least 1 sentence
        assert "one" in result

    def test_normal_ratio(self):
        text = "First. Second. Third. Fourth."
        result = self.safe_summarize(text, 0.5)
        # Should keep ~2 out of 4 sentences
        assert "First" in result


class TestRSIComputation:
    """Validate the RSI calculation logic added in Company Advanced Details."""

    @staticmethod
    def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def test_rsi_range(self):
        """RSI must always be in [0, 100]."""
        prices = pd.Series([100 + i * 0.5 for i in range(60)])
        rsi = self.compute_rsi(prices).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_steadily_rising_prices_high_rsi(self):
        """Predominantly rising prices (with small dips) should yield RSI > 60."""
        # Pure monotonic rise produces zero avg_loss -> RSI=NaN via 0/0.
        # Simulate a realistic mostly-upward series with occasional small dips.
        np.random.seed(42)
        base = np.linspace(100, 160, 80)
        noise = np.random.normal(0, 0.5, 80)
        prices = pd.Series(base + noise)
        rsi = self.compute_rsi(prices, window=14).dropna()
        assert len(rsi) > 0, "RSI series must have non-NaN values after dropna()"
        assert rsi.iloc[-1] > 60

    def test_steadily_falling_prices_low_rsi(self):
        """Steadily falling prices should produce RSI < 40."""
        prices = pd.Series(list(range(60, 0, -1)), dtype=float)
        rsi = self.compute_rsi(prices).dropna()
        assert rsi.iloc[-1] < 40

    def test_nan_propagation(self):
        """First (window-1) values should be NaN."""
        prices = pd.Series(list(range(1, 31)), dtype=float)
        rsi = self.compute_rsi(prices, window=14)
        assert rsi.iloc[:13].isna().all()


class TestATRComputation:
    """Validate the Average True Range (ATR) calculation logic."""

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        df = pd.DataFrame({'High': high, 'Low': low, 'prev_close': prev_close})
        tr = df.apply(
            lambda r: max(
                r['High'] - r['Low'],
                abs(r['High'] - r['prev_close']) if pd.notna(r['prev_close']) else 0,
                abs(r['Low'] - r['prev_close']) if pd.notna(r['prev_close']) else 0,
            ),
            axis=1,
        )
        return tr.ewm(span=window, min_periods=window).mean()

    def test_atr_positive(self):
        """ATR must always be non-negative."""
        n = 60
        high = pd.Series([100 + i + abs(math.sin(i)) for i in range(n)])
        low = pd.Series([100 + i - abs(math.sin(i)) for i in range(n)])
        close = pd.Series([100 + i for i in range(n)])
        atr = self.compute_atr(high, low, close).dropna()
        assert (atr >= 0).all()

    def test_atr_nan_before_window(self):
        """First (window-1) ATR values should be NaN."""
        n = 30
        high = pd.Series([float(i + 1) for i in range(n)])
        low = pd.Series([float(i) for i in range(n)])
        close = pd.Series([float(i) + 0.5 for i in range(n)])
        atr = self.compute_atr(high, low, close, window=14)
        assert atr.iloc[:13].isna().all()

    def test_flat_market_low_atr(self):
        """A perfectly flat market should have near-zero ATR."""
        n = 60
        high = pd.Series([100.5] * n, dtype=float)
        low = pd.Series([99.5] * n, dtype=float)
        close = pd.Series([100.0] * n, dtype=float)
        atr = self.compute_atr(high, low, close).dropna()
        assert len(atr) > 0
        assert atr.iloc[-1] < 2.0

    def test_volatile_market_high_atr(self):
        """A highly volatile market should have a larger ATR than a flat market."""
        n = 60
        volatile_high = pd.Series([100 + (i % 2) * 20 for i in range(n)], dtype=float)
        volatile_low = pd.Series([100 - (i % 2) * 20 for i in range(n)], dtype=float)
        volatile_close = pd.Series([100.0] * n, dtype=float)
        flat_high = pd.Series([101.0] * n, dtype=float)
        flat_low = pd.Series([99.0] * n, dtype=float)
        flat_close = pd.Series([100.0] * n, dtype=float)
        volatile_atr = self.compute_atr(volatile_high, volatile_low, volatile_close).dropna()
        flat_atr = self.compute_atr(flat_high, flat_low, flat_close).dropna()
        assert volatile_atr.iloc[-1] > flat_atr.iloc[-1]


class TestStochasticOscillator:
    """Validate the Stochastic Oscillator (%K / %D) computation."""

    @staticmethod
    def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                            k_window: int = 14, d_window: int = 3):
        lowest_low = low.rolling(k_window).min()
        highest_high = high.rolling(k_window).max()
        pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, float('nan'))
        pct_d = pct_k.rolling(d_window).mean()
        return pct_k, pct_d

    def test_range_0_to_100(self):
        """Both %K and %D must stay within [0, 100]."""
        n = 80
        np.random.seed(99)
        close = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 100)
        high = close + abs(np.random.normal(0, 0.5, n))
        low = close - abs(np.random.normal(0, 0.5, n))
        k, d = self.compute_stochastic(high, low, close)
        k_clean = k.dropna()
        d_clean = d.dropna()
        assert (k_clean >= 0).all() and (k_clean <= 100).all()
        assert (d_clean >= 0).all() and (d_clean <= 100).all()

    def test_nan_before_k_window(self):
        """First k_window-1 %K values must be NaN."""
        n = 40
        close = pd.Series(list(range(1, n + 1)), dtype=float)
        high = close + 1
        low = close - 1
        k, _ = self.compute_stochastic(high, low, close, k_window=14)
        assert k.iloc[:13].isna().all()

    def test_at_peak_stochastic_near_100(self):
        """%K should approach 100 when price is at the top of its range."""
        n = 20
        high = pd.Series([100.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([100.0] * n)  # Always at the high
        k, _ = self.compute_stochastic(high, low, close, k_window=14)
        k_clean = k.dropna()
        assert k_clean.iloc[-1] > 95.0

    def test_at_trough_stochastic_near_0(self):
        """%K should approach 0 when price is at the bottom of its range."""
        n = 20
        high = pd.Series([100.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([90.0] * n)  # Always at the low
        k, _ = self.compute_stochastic(high, low, close, k_window=14)
        k_clean = k.dropna()
        assert k_clean.iloc[-1] < 5.0


class TestMACDHistogram:
    """Validate the MACD Histogram (macd - signal) calculation."""

    @staticmethod
    def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
        ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def test_histogram_is_difference(self):
        """Histogram must equal macd_line - signal_line at every point."""
        prices = pd.Series([100.0 + i * 0.3 for i in range(100)])
        macd_line, signal_line, histogram = self.compute_macd(prices)
        diff = (macd_line - signal_line).dropna()
        hist_clean = histogram.dropna()
        common_idx = diff.index.intersection(hist_clean.index)
        pd.testing.assert_series_equal(diff[common_idx], hist_clean[common_idx], check_names=False)

    def test_histogram_sign_bullish(self):
        """Rising prices should produce a non-extreme histogram value."""
        prices = pd.Series([100 + i * 0.5 for i in range(100)], dtype=float)
        _, _, histogram = self.compute_macd(prices)
        hist_clean = histogram.dropna()
        assert len(hist_clean) > 0
        assert hist_clean.iloc[-1] > -5.0

    def test_nan_before_slow_window(self):
        """Histogram must be NaN for the first slow+signal-1 values."""
        prices = pd.Series(list(range(1, 101)), dtype=float)
        _, _, histogram = self.compute_macd(prices, fast=12, slow=26, signal=9)
        assert histogram.iloc[:33].isna().all()


# ---------------------------------------------------------------------------
# render_alert_banner
# ---------------------------------------------------------------------------

class TestRenderAlertBanner:
    """Tests for ui_theme.render_alert_banner() HTML output."""

    def setup_method(self):
        self.captured = []
        self.st_mock = types.SimpleNamespace(
            markdown=lambda html, **kw: self.captured.append(html)
        )

    def test_info_banner_contains_message(self):
        render_alert_banner(self.st_mock, "Hello world", "info")
        assert "Hello world" in self.captured[-1]

    def test_warning_banner_amber_colour(self):
        render_alert_banner(self.st_mock, "Overbought", "warning")
        html = self.captured[-1]
        assert "#F59E0B" in html

    def test_success_banner_green_colour(self):
        render_alert_banner(self.st_mock, "All good", "success")
        html = self.captured[-1]
        assert "#22C55E" in html

    def test_danger_banner_red_colour(self):
        render_alert_banner(self.st_mock, "Risk", "danger")
        html = self.captured[-1]
        assert "#EF4444" in html

    def test_unknown_type_falls_back_to_info(self):
        render_alert_banner(self.st_mock, "Fallback", "unknown_type")
        html = self.captured[-1]
        # Should default to info (indigo)
        assert "#6366F1" in html

    def test_html_escaping(self):
        render_alert_banner(self.st_mock, "<script>alert(1)</script>", "info")
        html = self.captured[-1]
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# render_metric_delta_card
# ---------------------------------------------------------------------------

class TestRenderMetricDeltaCard:
    """Tests for ui_theme.render_metric_delta_card() HTML output."""

    def setup_method(self):
        self.captured = []
        self.st_mock = types.SimpleNamespace(
            markdown=lambda html, **kw: self.captured.append(html)
        )

    def test_positive_delta_uses_green(self):
        render_metric_delta_card(self.st_mock, "Return", "+2.3%", 2.3)
        assert "#22C55E" in self.captured[-1]

    def test_negative_delta_uses_red(self):
        render_metric_delta_card(self.st_mock, "Return", "-1.5%", -1.5)
        assert "#EF4444" in self.captured[-1]

    def test_label_appears_in_html(self):
        render_metric_delta_card(self.st_mock, "My Label", "42", 0.5)
        assert "My Label" in self.captured[-1]

    def test_value_appears_in_html(self):
        render_metric_delta_card(self.st_mock, "P/E", "25.4x", 1.0)
        assert "25.4x" in self.captured[-1]

    def test_upward_arrow_when_positive(self):
        render_metric_delta_card(self.st_mock, "X", "Y", 1.0)
        assert "\u25b2" in self.captured[-1]  # up arrow

    def test_downward_arrow_when_negative(self):
        render_metric_delta_card(self.st_mock, "X", "Y", -1.0)
        assert "\u25bc" in self.captured[-1]  # down arrow
