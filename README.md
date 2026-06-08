# FAST â€“ Financial Analytics with Stock Prediction and Timeseries Forecasting

<p align="center">
  <img src="https://github.com/jayshilj/Team3_CSYE7245_Spring2021/blob/main/Final%20Project/Architecture%20Final%20AWS_FAST.jpg" width="100%" style="border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Tests-36%20passed-22C55E?logo=pytest" alt="Tests 36 passed" />
  <img src="https://img.shields.io/badge/Documentation-Reference-blueviolet?logo=read-the-docs&logoColor=white" alt="Documentation" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License" />
  <img src="https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20%7C%20Perplexity-6366F1" alt="LLM" />
</p>

---

## ðŸš€ What is FAST?

**FAST** is a production-grade **Agentic Financial Intelligence Platform** built with Streamlit. It combines real-time market data, multi-model LLM research, technical analysis, and social sentiment into a single premium glassmorphic dashboard.

> Originally developed as a CSYE 7245 Final Project (Spring 2021) at Northeastern University, FAST has since evolved substantially and now features state-of-the-art AI integrations, robust data pipelines, and a professional UI/UX layer.

---

## ðŸ“š Documentation Reference

For deeper insights into the project's internal mechanics, deployment procedures, and code standards, refer to the following resources:

* **[System Architecture](docs/architecture.md)** â€“ Software design, state management, and component architecture.
* **[Technical Indicators Reference](docs/indicators_reference.md)** â€“ Mathematical formulas, parameters, and default values for tech indicators.
* **[API Keys & Secrets Setup](docs/api_keys_setup.md)** â€“ Step-by-step instructions for obtaining and configuring Reddit/PRAW credentials.
* **[Deployment & Operations Guide](docs/deployment.md)** â€“ Guides for local run, Docker containerization, and Streamlit Community Cloud hosting.
* **[Data Sources & Sentiment Engine](docs/data_sources.md)** â€“ Descriptions of data providers (yfinance, FinViz, StockTwits) and the VADER sentiment classifier.
* **[Testing Strategy](docs/testing.md)** â€“ Unit test setup, command parameters, and testing guidelines.
* **[Contributing Guidelines](CONTRIBUTING.md)** â€“ Working with Git, styling conventions, and submitting PRs.
* **[Code of Conduct](CODE_OF_CONDUCT.md)** â€“ Committing to a welcoming, diverse, and respectful workspace.
* **[Security Policy](SECURITY.md)** â€“ Guidelines for reporting security issues and vulnerabilities.

---

## âœ¨ Key Features

### ðŸ“ˆ Real-Time Market Dashboard
- Live price, daily move, volume, and trend signal (Bullish/Bearish/Neutral)
- 2-year price history with overlay volume bars
- Interactive SMA 20/50 trend indicators

### ðŸ”® Advanced Forecasting Engine (Triple Model)
| Model | Description |
|---|---|
| **Holt-Winters** | Exponential smoothing capturing level, trend, and seasonality |
| **SARIMA** | Seasonal ARIMA with statistical lag correlation |
| **Monte Carlo (GBM)** | Stochastic multi-path simulation with historical drift and volatility |

### ðŸ“ Technical Indicators (Company Advanced Details)
- **Moving Average** â€“ SMA & EMA with configurable window and period
- **MACD** â€“ 12/26/9-period MACD with signal line on dual-axis chart
- **RSI** â€“ Relative Strength Index (configurable window, 7â€“30 days) with overbought/oversold thresholds
- **Bollinger Bands** â€“ Configurable SMA centre Â± N standard deviations with shaded channel

### ðŸ¤– Agentic Research Bot
- **Multi-LLM**: Toggle between **Google Gemini (2.5 Flash)** and **Perplexity (Sonar)**
- **Contextual Grounding**: Injects live price, P/E, market cap, sector, and FinViz headlines as system prompt
- **Predefined Questions**: One-click analysis templates (bull/bear case, news summary, valuation)
- **Dynamic API Key Input**: Secure runtime entry â€” no hardcoded credentials

### ðŸ’¬ Real-Time Social Sentiment
- **StockTwits**: Live messages with native sentiment tags and VADER NLP fallback
- **Reddit** (WallStreetBets + Stocks + Investing): Authenticated PRAW or fallback unauthenticated search
- **Sentiment Distribution**: Donut chart (Bullish / Bearish / Neutral)
- **WordCloud**: Filtered for noise, stopwords, and generic financial terms

### ðŸ“° Live News Sentiment (FinViz)
- Real-time headline scraping for any S&P 500 ticker
- VADER NLP compound scoring with Positive / Neutral / Negative classification
- Pie chart distribution + sentiment-over-time bar chart

### ðŸ”Ž Google Trends Forecasting
- Keyword interest over time from Google Trends
- Seasonal decomposition (monthly averages, quarterly bar chart)
- Multi-model forecasting (Holt-Winters / SARIMA / Monte Carlo)

### ðŸŽ™ï¸ Meeting Summarization (Legacy)
- MP3 audio playback for financial meeting recordings
- Ratio-based text summarization from Amazon Transcribe outputs

### ðŸ’Ž Premium UI/UX
- **Glassmorphic design system** via `ui_theme.py` with Inter typeface
- Dark theme by default, fully compatible with Streamlit light/dark toggles
- Custom `config.toml` theme tokens matching the design palette

---

## ðŸ›  Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.10 and 3.12)
- **Git**
- **LLM API Credentials** *(optional)*: Google Gemini API Key or Perplexity Sonar API Key

### Installation & Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jayshilj/FAST-Stock-Analysis-WebApp.git
   cd FAST-Stock-Analysis-WebApp
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv

   # Windows (PowerShell)
   .\venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys** *(optional â€” can also be entered at runtime in the sidebar)*:
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your-gemini-key"
   $env:PERPLEXITY_API_KEY="your-perplexity-key"

   # Linux / macOS
   export GEMINI_API_KEY="your-gemini-key"
   export PERPLEXITY_API_KEY="your-perplexity-key"
   ```

5. **Configure Reddit API** *(optional, for authenticated Reddit scraping)*:
   Create `.streamlit/secrets.toml`:
   ```toml
   [reddit]
   client_id = "your-reddit-client-id"
   client_secret = "your-reddit-client-secret"
   user_agent = "FAST-Stock-Analysis/1.0"
   ```

6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## ðŸ§ª Running the Test Suite

```bash
pip install pytest
pytest tests/ -v
```

The suite covers 28 unit tests across:
- `_fmt_metric()` â€” value formatting (T/B/M/K/raw)
- `render_sentiment_badge()` â€” HTML badge generation
- `NAV_DEFINITION` â€” structural integrity (uniqueness, arity)
- `normalize_market_df()` â€” MultiIndex column flattening
- `safe_summarize()` â€” ratio clamping and sentence selection
- RSI computation â€” range `[0, 100]`, directional behavior, NaN propagation

---

## ðŸ— Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        FAST Dashboard                            â”‚
â”‚                     Streamlit + ui_theme.py                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
              â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚   Data Sources       â”‚    â”‚   Intelligence Layer          â”‚
   â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚    â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
   â”‚ Yahoo Finance        â”‚    â”‚ Gemini 2.5 Flash (Google)     â”‚
   â”‚ FinViz (scraping)    â”‚    â”‚ Perplexity Sonar              â”‚
   â”‚ Google Trends        â”‚    â”‚ VADER Sentiment (NLP)         â”‚
   â”‚ StockTwits API       â”‚    â”‚ TextBlob                      â”‚
   â”‚ Reddit (PRAW / JSON) â”‚    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
              â”‚
   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   â”‚  Analytics & ML      â”‚
   â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
   â”‚ Holt-Winters (HW)    â”‚
   â”‚ SARIMA               â”‚
   â”‚ Monte Carlo (GBM)    â”‚
   â”‚ RSI                  â”‚
   â”‚ Bollinger Bands      â”‚
   â”‚ SMA / EMA / MACD     â”‚
   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Cloud Infrastructure Vision (AWS)
| Service | Role |
|---|---|
| **AWS Lambda** | Serverless periodic scrapers for Reddit & StockTwits |
| **Amazon S3** | Data lake for raw JSON sentiment payloads |
| **AWS Glue** | ETL pipeline to partitioned Parquet stores |
| **Amazon Redshift** | Columnar data warehouse for long-term analytics |

---

## ðŸ“ Project Structure

```text
.
â”œâ”€â”€ .streamlit/
â”‚   â”œâ”€â”€ config.toml          # Streamlit theme and server configuration
â”‚   â””â”€â”€ secrets.toml         # (gitignored) Reddit/API credentials
â”œâ”€â”€ Datasets/
â”‚   â””â”€â”€ SP500.csv            # S&P 500 ticker universe
â”œâ”€â”€ Images/                  # Architecture diagrams
â”œâ”€â”€ Audio Files/             # Meeting audio samples (legacy)
â”œâ”€â”€ inference-data/          # Meeting transcription data (legacy)
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ conftest.py          # Pytest configuration and shared fixtures
â”‚   â””â”€â”€ test_helpers.py      # 28 unit tests for helpers and utilities
â”œâ”€â”€ app.py                   # Main Streamlit application (1700+ lines)
â”œâ”€â”€ ui_theme.py              # Design system: CSS, layout, and HTML components
â”œâ”€â”€ requirements.txt         # Versioned Python dependencies
â”œâ”€â”€ pytest.ini               # Pytest runner configuration
â”œâ”€â”€ CHANGELOG.md             # Version history (Keep a Changelog format)
â””â”€â”€ README.md                # You are here
```

---

## ðŸ“ Authors

<b>[Jayshil Jain](https://www.linkedin.com/in/jayshiljain/)</b>
<b>[Sagar Shah](https://www.linkedin.com/in/shahsagar95/)</b>
<b>[Akash M Dubey](https://www.linkedin.com/in/akashmdubey/)</b>

---

## ðŸ“„ License

This project is licensed under the MIT License â€“ see the [LICENSE](LICENSE) file for details.

---

## ðŸ“‹ Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of all notable changes.
