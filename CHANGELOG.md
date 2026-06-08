# Changelog

All notable changes to the **FAST â€“ Financial Analytics with Stock Prediction and Timeseries Forecasting** project are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions. Versioning aligns with [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Simple & Exponential Moving Averages (SMA / EMA)** â€“ Added new standalone `compute_sma` and `compute_ema` functions to `indicators.py`.
- **Unit test suite for SMA / EMA** â€“ Implemented test cases checking values, NaN propagation, and edge cases for the new moving averages.
- **Developer Documentation Directory** â€“ Created a comprehensive `docs/` repository including guides for `architecture.md`, `indicators_reference.md`, `api_keys_setup.md`, `deployment.md`, `data_sources.md`, and `testing.md`.
- **Community Standards** â€“ Added root-level open-source guidelines: `CONTRIBUTING.md` for environment setup, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md` for coordinated vulnerability reporting.
- **Enhanced Code Documentation** â€“ Documented app initialization (`install_requirements()`) and main layout entry points (`main()`) in `app.py` with standard Python docstrings.
- **Stochastic Oscillator (%K / %D)** â€“ Configurable fast/slow stochastic chart with overbought (80) and oversold (20) thresholds added to *Company Advanced Details*.
- **Average True Range (ATR)** â€“ Wilder EWM-smoothed volatility meter with area fill chart added to *Company Advanced Details*.
- **Bollinger Band Width** â€“ Band-squeeze subplot showing (Upper âˆ’ Lower) / SMA Ã— 100 displayed below the main Bollinger Bands chart, helping identify consolidation phases before breakouts.
- **MACD Histogram** â€“ Color-coded (green/red) bar trace layered onto the MACD panel showing the divergence between the MACD line and its signal line.
- **`indicators.py`** â€“ Standalone indicator library (`compute_rsi`, `compute_atr`, `compute_macd`, `compute_stochastic`, `compute_bollinger`) fully decoupled from Streamlit, enabling reuse in notebooks or API endpoints.
- **Unit test expansion** â€“ `TestATRComputation`, `TestStochasticOscillator`, and `TestMACDHistogram` classes added to `tests/test_helpers.py`, growing the suite to 36+ assertions.
- **Indicators Edge Cases Tests** â€“ Wrote unit tests checking empty inputs, NaN series, and zero volatility conditions for the `indicators.py` library.

### Changed
- **Standardized Moving Averages Integration** â€“ Refactored the `Dashboard` page and `Company Advanced Details` page in `app.py` to import and consume `compute_sma` and `compute_ema` from the standalone library instead of manually computing them.
- **`compute_rsi()` helper** â€“ Extracted as a named function near other shared helpers in `app.py`; the inline RSI block in *Company Advanced Details* now calls this helper, eliminating duplication.
- **`calc_macd()` function** â€“ Now computes and returns the `histogram` column (`macd âˆ’ signal`) alongside the MACD and signal lines.
- **Standardized Indicator Integration** â€“ Refactored `app.py` to import and consume mathematical indicators (`compute_rsi`, `compute_bollinger`, `compute_stochastic`, `compute_macd`, `compute_atr`) directly from `indicators.py` instead of executing duplicate inline calculations.
- **Architecture Documentation** â€“ Updated `docs/architecture.md` system blueprint to clarify the data calculation flow between `app.py` and the core `indicators.py` mathematical module.

### Fixed
- **Meeting Summarization page** â€“ Replaced two bare `except:` clauses with typed `except (ValueError, TypeError, OSError, Exception):` to prevent `KeyboardInterrupt` / `SystemExit` from being silently swallowed.

---

## [2.0.0] â€“ 2025-05 Â· Agentic Intelligence Platform

### Added
- **Agentic Research Bot** â€“ Multi-LLM conversational AI with Google Gemini (2.5 Flash) and Perplexity (Sonar) support. Contextual grounding via live Yahoo Finance and FinViz data injection.
- **Monte Carlo (Geometric Brownian Motion)** â€“ Stochastic multi-path stock price simulation engine.
- **SARIMA Forecasting** â€“ Seasonal ARIMA model for structured statistical price prediction.
- **Premium Glassmorphic UI** â€“ `ui_theme.py` design system featuring glassmorphism, Google Fonts (Inter), and custom CSS tokens.
- **Developer Info Sidebar** â€“ Links to portfolio, GitHub, and LinkedIn within the dashboard.
- **StockTwits & Reddit Live Feed** â€“ Real-time social sentiment pipeline with VADER NLP scoring and WordCloud generation.
- **Dynamic API Key Input** â€“ Secure runtime API key entry in the sidebarâ€”no hardcoded credentials.

### Changed
- Migrated from `st.pyplot` warnings to explicit `matplotlib.use('Agg')` backend.
- Replaced deprecated `yfinance.download()` single-column logic with `Ticker.history()` primary path and robust MultiIndex column flattening.

---

## [1.0.0] â€“ 2021-04 Â· Initial Release (CSYE 7245 Final Project)

### Added
- Real-time stock price dashboard with SMA 20/50 indicators.
- Google Trends integration with Holt-Winters Exponential Smoothing forecast.
- FinViz news sentiment scraping and VADER NLP scoring.
- AWS architecture blueprint: Lambda â†’ S3 â†’ Glue â†’ Redshift.
- Meeting audio playback and text summarization (Amazon Transcribe integration).
- Power BI embedded dashboard.

---

[Unreleased]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/compare/HEAD...HEAD
[2.0.0]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/releases/tag/v2.0.0
[1.0.0]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/releases/tag/v1.0.0
