# Changelog

All notable changes to the **FAST – Financial Analytics with Stock Prediction and Timeseries Forecasting** project are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions. Versioning aligns with [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **RSI Indicator** – Relative Strength Index (RSI) chart with configurable window and overbought/oversold thresholds added to the *Company Advanced Details* page.
- **Unit Test Suite** – `tests/test_helpers.py` and `pytest.ini` providing 20+ tests covering `_fmt_metric`, `render_sentiment_badge`, NAV integrity, `normalize_market_df`, `safe_summarize`, and RSI computation logic.
- **`conftest.py`** – Shared pytest fixtures and marker registration.
- **`requests`** – Explicit dependency declaration for the Perplexity API and social-media scrapers.

### Changed
- **`requirements.txt`** – All dependencies now carry explicit compatible-release version ranges (`>=MIN,<MAJOR+1`) to prevent silent breaking changes.
- **Live News Sentiment page** – Removed 40+ lines of duplicated scraping code. Now reuses the top-level `get_news_sentiment_df()` helper with proper error handling for fetch failures.
- **Bare `except` clauses** – All `except:` blocks in `Social Media Trends` replaced with typed exceptions (`ValueError`, `KeyError`, `IndexError`, `Exception`) to prevent silently swallowing `KeyboardInterrupt` / `SystemExit`.
- **About the Project page** – Corrected data-source description: removed reference to deprecated Twitter API; now accurately lists FinViz web scraping, StockTwits, and Reddit as the active data sources.

### Fixed
- `.gitignore` – Now properly excludes `__pycache__/`, `*.pyc`, `scratch/`, `.env`, `.vscode/`, and `.idea/`.

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
