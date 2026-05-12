# FAST - Financial Analytics with Stock Prediction and Timeseries Forecasting

<p align="center">
  <img src="https://github.com/jayshilj/Team3_CSYE7245_Spring2021/blob/main/Final%20Project/Architecture%20Final%20AWS_FAST.jpg" width="100%" style="border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);" />
</p>

## 🚀 Recent Intelligence Upgrades  

FAST has evolved into a comprehensive **Agentic Financial Intelligence Platform**. Recent updates have integrated state-of-the-art AI and robust data pipelines:

*   **🤖 Agentic Research Bot**: A multi-model conversational AI module natively integrated into the dashboard. Features a dynamic setup allowing seamless toggling between **Google Gemini (2.5 Flash)** and **Perplexity (Sonar)**. It injects real-time market context (price, metrics, news) to provide grounded, expert-level analysis.
*   **📈 Advanced Forecasting**: Leveraging lightweight, high-accuracy statistical forecasting via `statsmodels` (Holt-Winters Exponential Smoothing). It captures complex trends and periodic seasonal spikes (e.g., holiday search trends) with interactive **Plotly** visualizations.
*   **💬 Real-Time Social Sentiment**: A robust data pipeline for **StockTwits & Reddit**. Features real-time VADER NLP sentiment charting and aggressive algorithms for rendering clean, dense financial WordClouds.
*   **💎 Premium UI/UX**: A modernized, glassmorphic Streamlit interface with custom Inter-based typography, interactive metric strips, and institutional-grade charting.

---

## ✨ Key Features

- **Real-time Market Dashboard**: Track prices, daily moves, and volume with interactive technical indicators (SMA 20/50).
- **Advanced Forecasting Engine**: Triple-model support featuring **Holt-Winters**, **SARIMA**, and **Monte Carlo (Geometric Brownian Motion)** for both smooth trends and organic, stochastic price patterns.
- **Google Trends Integration**: Analyze and forecast search interest for any keyword with seasonal awareness.
- **Sentiment Analysis**: Live news sentiment tracking from FinViz, visualized with interactive pie charts and sentiment badges.
- **Financial Audio Transcription**: (Legacy Support) Infrastructure for transcribing and summarizing financial meeting audio via Amazon Transcribe.
- **Model-Agnostic LLM Interface**: Securely use Gemini or Perplexity for deep-dive research without hardcoded API keys.

---

## 🏗 Architecture

The platform leverages a hybrid cloud architecture designed for scalability and real-time responsiveness:

- **Frontend**: Streamlit with custom CSS and Plotly.
- **Intelligence**: Google Gemini (via `google-genai`), Perplexity API, and VADER Sentiment.
- **Data Sources**: Yahoo Finance (`yfinance`), Google Trends (`pytrends`), StockTwits, and Reddit.
- **Cloud Infrastructure (Vision)**: AWS Lambda for scraping, S3 for storage, Glue for ETL, and Redshift for historical data warehousing.

---

## 🛠 Getting Started

### Prerequisites

- **Python 3.9+** (Recommended)
- **Git**
- **API Keys** (Optional, for Research Bot): Gemini API Key or Perplexity API Key.

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jayshilj/FAST-Stock-Analysis-WebApp.git
    cd FAST-Stock-Analysis-WebApp
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
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
