# Data Sources and Sentiment Analysis Engine

The **FAST Stock Analysis WebApp** aggregates financial, corporate, and social media data from multiple public data providers. This document summarizes those data sources, query mechanisms, and the sentiment analysis pipeline.

---

## 1. Yahoo Finance (via `yfinance`)

* **Endpoint / Library**: Unofficial python wrapper library `yfinance`.
* **Purpose**:
  * **Historical Prices**: Downloads daily OHLCV (Open, High, Low, Close, Volume) data for any selected time window.
  * **Company Metadata**: Retrieves key fundamental stats (sector, industry, company description, PEG ratio, forward P/E, profit margins, enterprise-to-EBITDA).
* **Caching**: Data is cached locally with `@st.cache_data` using the ticker and timeframe as cache keys to minimize network latency and prevent Yahoo Finance rate limits.

---

## 2. FinViz Scraper

* **Endpoint**: `https://finviz.com/quote.ashx?t={ticker}`
* **Purpose**:
  * **News Feed**: Extracts the latest financial news headlines, dates, and sources.
  * **Insider Trading**: Parses the insider transaction table listing transactions by executives/directors (buying/selling shares, relationship, transaction value, date).
* **Implementation Details**: Uses Python's `urllib.request` and `BeautifulSoup` to parse standard HTML elements, disguised with a browser User-Agent header to avoid bot blocking.

---

## 3. Reddit API

* **Endpoint**: `https://www.reddit.com/r/wallstreetbets+stocks+investing/search.json?q={ticker}` (via unauthenticated JSON) or Python Reddit API Wrapper (`praw`).
* **Purpose**: Crawls subreddits dedicated to stock discussions (`r/wallstreetbets`, `r/stocks`, `r/investing`) for posts matching the target symbol.
* **Fallback Policy**: The app prefers `praw` credentials if configured, falling back gracefully to public JSON searches if keys are missing.

---

## 4. StockTwits API

* **Endpoint**: `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json`
* **Purpose**: Queries StockTwits for real-time messages, user sentiment classifications (bullish/bearish tags), and message timestamps.
* **Rate Limits**: The public StockTwits API has a rate limit of 200 requests per hour per IP. Caching is set to 5 minutes to mitigate rate-limiting.

---

## 5. Sentiment Processing Pipeline (VADER)

Once social media content (Reddit titles/bodies, StockTwits messages) and news headlines are fetched, the text is processed by a sentiment analyzer:

* **Engine**: `nltk.sentiment.vader.SentimentIntensityAnalyzer` (Valence Aware Dictionary and Sentiment Reasoner).
* **Pipeline**:
  1. Concatenate text streams (excluding noise/generic keywords).
  2. Compute polarity scores: **Negative**, **Neutral**, **Positive**, and **Compound**.
  3. Map the compound score to a semantic label:
     * **Bullish/Positive**: $\text{Compound} \ge 0.05$
     * **Bearish/Negative**: $\text{Compound} \le -0.05$
     * **Neutral**: $-0.05 < \text{Compound} < 0.05$
  4. Display aggregate sentiment indicators alongside gauge charts in the UI.
