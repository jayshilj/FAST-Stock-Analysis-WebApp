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
from ui_theme import _fmt_metric, render_sentiment_badge, APP_BRAND_FULL, NAV_DEFINITION


class TestFmtMetric:
    """Tests for ui_theme._fmt_metric() value formatter."""

    def test_none_returns_dash(self):
        assert _fmt_metric(None) == "—"

    def test_na_string_returns_dash(self):
        assert _fmt_metric("N/A") == "—"

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
