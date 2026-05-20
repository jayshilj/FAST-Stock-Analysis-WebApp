# FAST - Financial Analytics with Stock Prediction and Timeseries Forecasting

<p align="center">
  <img src="https://github.com/jayshilj/Team3_CSYE7245_Spring2021/blob/main/Final%20Project/Architecture%20Final%20AWS_FAST.jpg" width="100%" style="border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);" />
</p>

## 🚀 Recent Intelligence Upgrades  

FAST has evolved into a comprehensive **Agentic Financial Intelligence Platform**. Recent updates have integrated state-of-the-art AI and robust data pipelines:

*   **🤖 Agentic Research Bot**: A multi-model conversational AI module natively integrated into the dashboard. Features a dynamic setup allowing seamless toggling between **Google Gemini (2.5 Flash)** and **Perplexity (Sonar)**. It injects real-time market context (price, metrics, news) to provide grounded, expert-level analysis.
    *   **Dynamic API Key Input**: Input your API keys safely through the sidebar at runtime without exposing credentials in the codebase.
    *   **Contextual Grounding**: Live scraping from Yahoo Finance and FinViz is dynamically formatted and injected as a system prompt, enabling the bot to converse with up-to-the-minute stock facts and figures.
*   **📈 Advanced Forecasting**: Leveraging lightweight, high-accuracy statistical forecasting via `statsmodels` (Holt-Winters Exponential Smoothing). It captures complex trends and periodic seasonal spikes (e.g., holiday search trends) with interactive **Plotly** visualizations.
*   **💬 Real-Time Social Sentiment**: A robust data pipeline for **StockTwits & Reddit**. Features real-time VADER NLP sentiment charting and aggressive algorithms for rendering clean, dense financial WordClouds.
*   **💎 Premium UI/UX & Developer Sidebar**: A modernized, glassmorphic design system powered by `ui_theme.py`:
    *   **Glassmorphic Design Tokens**: Custom CSS injected globally to support blur backdrops, fine-bordered cards, custom text inputs, buttons, and high-contrast tables.
    *   **Typography**: Clean global overrides applying Google Font's **Inter** typeface across the entire viewport.
    *   **Developer Info Sidebar Panel**: Clean, interactive developer biography card, providing links to portfolios, GitHub repositories, and LinkedIn profiles directly within the app context.
    *   **Re-engineered Metric Strips**: Custom layout wrappers converting native metrics into sleek, border-framed panels displaying Market Cap, P/E, Dividend Yield, and Beta.

---

## ✨ Key Features

- **Real-time Market Dashboard**: Track prices, daily moves, and volume with interactive technical indicators (SMA 20/50).
- **Advanced Forecasting Engine**: Triple-model support for robust predictive analytics:
    *   **Holt-Winters Exponential Smoothing**: Captures level, trend, and seasonality natively. Best suited for predictable periodic cycles like Google Search trend seasonality.
    *   **SARIMA (Seasonal Autoregressive Integrated Moving Average)**: Employs statistical lag correlation and seasonal integration for structured, mathematically rigorous prediction.
    *   **Monte Carlo (Geometric Brownian Motion)**: Projects organic, stochastic price patterns by running multi-path simulations based on historical drift and asset volatility, yielding dynamic confidence intervals.
- **Google Trends Integration**: Analyze and forecast search interest for any keyword with seasonal awareness.
- **Multi-Source Sentiment Engine & NLP Pipeline**: Live sentiment intelligence aggregated from major social channels and news feeds:
    *   **Financial News (FinViz)**: Real-time scraping of financial news headlines mapped directly to the active stock ticker, complete with interactive distribution pie charts and color-coded sentiment indicators.
    *   **Social Channels (Reddit & StockTwits)**: High-throughput, rate-limit aware scrapers querying the latest discussion threads and investor posts.
    *   **VADER NLP Processor**: Classifies social commentary via the Valence Aware Dictionary and sEntiment Reasoner, yielding precise positivity, negativity, and compound sentiment scores.
    *   **High-Density WordClouds**: Dynamic text processing that filters out stopwords, noise, and generic symbols to generate clean visual representation of hot keywords and investor buzz.
- **Financial Audio Transcription**: (Legacy Support) Infrastructure for transcribing and summarizing financial meeting audio via Amazon Transcribe.
- **Model-Agnostic LLM Interface**: Securely use Gemini or Perplexity for deep-dive research without hardcoded API keys.

---

## 🏗 Architecture

The platform leverages a hybrid cloud architecture designed for scalability and real-time responsiveness:

- **Frontend**: Streamlit with custom CSS and Plotly.
- **Intelligence**: Google Gemini (via `google-genai`), Perplexity API, and VADER Sentiment.
- **Data Sources**: Yahoo Finance (`yfinance`), Google Trends (`pytrends`), StockTwits, and Reddit.
- **Cloud Infrastructure (Vision)**:
    *   **AWS Lambda**: Serverless microservices to execute periodic scrapers for Reddit and StockTwits feeds.
    *   **Amazon S3**: High-durability data lake hosting raw scraped JSON sentiment payloads.
    *   **AWS Glue**: ETL pipeline that catalogs schemas and aggregates data into partitioned Parquet stores.
    *   **Amazon Redshift**: Columnar data warehouse enabling performant historical query analytics on long-term market trends.

---

## 🛠 Getting Started

### Prerequisites

- **Python 3.9+** (Tested on Python 3.10 and 3.12)
- **Git**
- **LLM API Credentials** (Optional): A Google Gemini API Key or a Perplexity Sonar API Key.

### Installation & Environment Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jayshilj/FAST-Stock-Analysis-WebApp.git
    cd FAST-Stock-Analysis-WebApp
    ```

2.  **Establish Environment Variables** (Optional, to bypass sidebar manual entry):
    Create a `.env` file or export variables in your shell:
    ```bash
    # On Windows (PowerShell)
    $env:GEMINI_API_KEY="your-gemini-key"
    $env:PERPLEXITY_API_KEY="your-perplexity-key"
    
    # On Linux/macOS
    export GEMINI_API_KEY="your-gemini-key"
    export PERPLEXITY_API_KEY="your-perplexity-key"
    ```

3.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Activate on Windows:
    .\venv\Scripts\activate
    # Activate on Linux/macOS:
    source venv/bin/activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

```text
.
├── .streamlit/             # Streamlit configuration
├── Datasets/               # Static datasets (e.g., SP500.csv)
├── Images/                 # Project assets and diagrams
├── app.py                  # Main application entry point
├── ui_theme.py             # Premium design system and CSS injection
├── requirements.txt        # Project dependencies
└── README.md               # You are here
```

---

## 📝 Authors

<b>[Jayshil Jain](https://www.linkedin.com/in/jayshiljain/)</b>
<b>[Sagar Shah](https://www.linkedin.com/in/shahsagar95/)</b>
<b>[Akash M Dubey](https://www.linkedin.com/in/akashmdubey/)</b>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
