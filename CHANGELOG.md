# Changelog

All notable changes to the **FAST – Financial Analytics with Stock Prediction and Timeseries Forecasting** project are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions. Versioning aligns with [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Developer Documentation Directory** – Created a comprehensive `docs/` repository including guides for `architecture.md`, `indicators_reference.md`, `api_keys_setup.md`, `deployment.md`, `data_sources.md`, and `testing.md`.
- **Community Standards** – Added root-level open-source guidelines: `CONTRIBUTING.md` for environment setup, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md` for coordinated vulnerability reporting.
- **Enhanced Code Documentation** – Documented app initialization (`install_requirements()`) and main layout entry points (`main()`) in `app.py` with standard Python docstrings.
- **Stochastic Oscillator (%K / %D)** – Configurable fast/slow stochastic chart with overbought (80) and oversold (20) thresholds added to *Company Advanced Details*.
- **Average True Range (ATR)** – Wilder EWM-smoothed volatility meter with area fill chart added to *Company Advanced Details*.
- **Bollinger Band Width** – Band-squeeze subplot showing (Upper − Lower) / SMA × 100 displayed below the main Bollinger Bands chart, helping identify consolidation phases before breakouts.
- **MACD Histogram** – Color-coded (green/red) bar trace layered onto the MACD panel showing the divergence between the MACD line and its signal line.
- **`indicators.py`** – Standalone indicator library (`compute_rsi`, `compute_atr`, `compute_macd`, `compute_stochastic`, `compute_bollinger`) fully decoupled from Streamlit, enabling reuse in notebooks or API endpoints.
- **Unit test expansion** – `TestATRComputation`, `TestStochasticOscillator`, and `TestMACDHistogram` classes added to `tests/test_helpers.py`, growing the suite to 36+ assertions.

### Changed
- **`compute_rsi()` helper** – Extracted as a named function near other shared helpers in `app.py`; the inline RSI block in *Company Advanced Details* now calls this helper, eliminating duplication.
- **`calc_macd()` function** – Now computes and returns the `histogram` column (`macd − signal`) alongside the MACD and signal lines.

### Fixed
- **Meeting Summarization page** – Replaced two bare `except:` clauses with typed `except (ValueError, TypeError, OSError, Exception):` to prevent `KeyboardInterrupt` / `SystemExit` from being silently swallowed.

---

## [2.0.0] – 2025-05 · Agentic Intelligence Platform

### Added
- **Agentic Research Bot** – Multi-LLM conversational AI with Google Gemini (2.5 Flash) and Perplexity (Sonar) support. Contextual grounding via live Yahoo Finance and FinViz data injection.
- **Monte Carlo (Geometric Brownian Motion)** – Stochastic multi-path stock price simulation engine.
- **SARIMA Forecasting** – Seasonal ARIMA model for structured statistical price prediction.
- **Premium Glassmorphic UI** – `ui_theme.py` design system featuring glassmorphism, Google Fonts (Inter), and custom CSS tokens.
- **Developer Info Sidebar** – Links to portfolio, GitHub, and LinkedIn within the dashboard.
- **StockTwits & Reddit Live Feed** – Real-time social sentiment pipeline with VADER NLP scoring and WordCloud generation.
- **Dynamic API Key Input** – Secure runtime API key entry in the sidebar—no hardcoded credentials.

### Changed
- Migrated from `st.pyplot` warnings to explicit `matplotlib.use('Agg')` backend.
- Replaced deprecated `yfinance.download()` single-column logic with `Ticker.history()` primary path and robust MultiIndex column flattening.

---

## [1.0.0] – 2021-04 · Initial Release (CSYE 7245 Final Project)

### Added
- Real-time stock price dashboard with SMA 20/50 indicators.
- Google Trends integration with Holt-Winters Exponential Smoothing forecast.
- FinViz news sentiment scraping and VADER NLP scoring.
- AWS architecture blueprint: Lambda → S3 → Glue → Redshift.
- Meeting audio playback and text summarization (Amazon Transcribe integration).
- Power BI embedded dashboard.

---

[Unreleased]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/compare/HEAD...HEAD
[2.0.0]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/releases/tag/v2.0.0
[1.0.0]: https://github.com/jayshilj/FAST-Stock-Analysis-WebApp/releases/tag/v1.0.0
