# Technical Indicators Mathematical Reference

This reference documentation describes the core technical analysis indicators implemented in [indicators.py](file:///c:/Users/jaysh/OpenSourceContributions/FAST-Stock-Analysis-WebApp/indicators.py), including equations, parameters, and typical trading signals.

---

## 1. Relative Strength Index (RSI)

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.

* **Function**: `compute_rsi(prices, window=14)`
* **Mathematical Formula**:
  $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
  $$\text{RS} = \frac{\text{Smoothed Gain}}{\text{Smoothed Loss}}$$
  * Gains ($U$) and losses ($D$) are calculated from daily price changes.
  * Smoothing uses an exponential weighted moving average (EWM) with center-of-mass parameter `com = window - 1` (equivalent to Welles Wilder's smoothing technique).

* **Interpretation**:
  * **Overbought (>= 70)**: Suggests that the security may be overvalued or primed for a trend reversal/corrective pullback.
  * **Oversold (<= 30)**: Indicates that the security may be undervalued or oversold, potentially setting up for a rebound.

---

## 2. Average True Range (ATR)

Average True Range (ATR) is a volatility indicator that measures market range over a specific time period.

* **Function**: `compute_atr(high, low, close, window=14)`
* **Mathematical Formula**:
  $$\text{True Range (TR)} = \max\left(\text{High} - \text{Low}, \left|\text{High} - \text{Close}_{\text{prev}}\right|, \left|\text{Low} - \text{Close}_{\text{prev}}\right|\right)$$
  $$\text{ATR} = \text{EWM}(\text{TR}, \text{span}=\text{window})$$
  * The True Range accounts for any opening gaps between trading sessions.

* **Interpretation**:
  * **High Volatility**: Elevated ATR values signify large daily price ranges, typical during market trends or sudden news releases.
  * **Low Volatility**: Decreasing ATR represents quiet, consolidation phases. Often used to determine stop-loss placement (e.g., $2 \times \text{ATR}$).

---

## 3. Moving Average Convergence Divergence (MACD)

MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.

* **Function**: `compute_macd(prices, fast=12, slow=26, signal=9)`
* **Mathematical Formula**:
  $$\text{MACD Line} = \text{EMA}(\text{prices}, \text{span}=\text{fast}) - \text{EMA}(\text{prices}, \text{span}=\text{slow})$$
  $$\text{Signal Line} = \text{EMA}(\text{MACD Line}, \text{span}=\text{signal})$$
  $$\text{Histogram} = \text{MACD Line} - \text{Signal Line}$$

* **Interpretation**:
  * **Bullish Crossover**: MACD Line crosses above the Signal Line, indicating upward momentum.
  * **Bearish Crossover**: MACD Line crosses below the Signal Line, indicating downward momentum.
  * **Zero Line Crossover**: MACD crossing above 0 confirms a rising medium-term trend.

---

## 4. Stochastic Oscillator

The Stochastic Oscillator compares a security's closing price to its price range over a specific period.

* **Function**: `compute_stochastic(high, low, close, k_window=14, d_window=3)`
* **Mathematical Formula**:
  $$\%K = 100 \times \frac{\text{Close} - \text{Lowest Low}(k\_window)}{\text{Highest High}(k\_window) - \text{Lowest Low}(k\_window)}$$
  $$\%D = \text{SMA}(\%K, \text{d\_window})$$

* **Interpretation**:
  * **Overbought (>= 80)**: Indicates the asset is trading near the top of its high-low range.
  * **Oversold (<= 20)**: Indicates the asset is trading near the bottom of its high-low range.
  * **Crossover Signals**: Buy when $\%K$ crosses above $\%D$ below 20; sell when $\%K$ crosses below $\%D$ above 80.

---

## 5. Bollinger Bands

Bollinger Bands consist of a middle band (SMA) and two outer standard-deviation envelopes.

* **Function**: `compute_bollinger(prices, window=20, n_std=2.0)`
* **Mathematical Formula**:
  $$\text{Middle Band (SMA)} = \text{Rolling Mean}(\text{prices}, \text{window})$$
  $$\text{Upper Band} = \text{SMA} + \left(n\_std \times \sigma\right)$$
  $$\text{Lower Band} = \text{SMA} - \left(n\_std \times \sigma\right)$$
  $$\%B = \frac{\text{Close} - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}$$
  $$\text{BandWidth} = \frac{\text{Upper Band} - \text{Lower Band}}{\text{SMA}} \times 100$$
  * Where $\sigma$ is the rolling standard deviation of prices over the given window.

* **Interpretation**:
  * **Volatility Expansion/Contraction**: Bands widen when volatility increases and contract during consolidation (squeeze).
  * **%B Position**: $\%B > 1.0$ means the price is above the upper band; $\%B < 0.0$ means the price is below the lower band.
  * **BandWidth Squeeze**: A historically low BandWidth is often followed by a large price breakout.

---

## 6. Simple Moving Average (SMA)

The Simple Moving Average (SMA) is a trend-following indicator that calculates the average price of a security over a specified number of periods.

* **Function**: `compute_sma(prices, window=20)`
* **Mathematical Formula**:
  $$\text{SMA} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Price}_{t-i}$$
  * Where $N$ is the rolling window size.

* **Interpretation**:
  * **Trend Direction**: A rising SMA indicates an uptrend, while a falling SMA indicates a downtrend.
  * **Support/Resistance**: Longer-term SMAs (e.g., 50-day or 200-day) often act as dynamic support or resistance levels.

---

## 7. Exponential Moving Average (EMA)

The Exponential Moving Average (EMA) is a type of moving average that places a greater weight and significance on the most recent data points.

* **Function**: `compute_ema(prices, window=20)`
* **Mathematical Formula**:
  $$\text{EMA}_t = \alpha \times \text{Price}_t + (1 - \alpha) \times \text{EMA}_{t-1}$$
  $$\alpha = \frac{2}{N + 1}$$
  * Where $N$ is the smoothing window size, and $\alpha$ is the multiplier.

* **Interpretation**:
  * **Reactivity**: EMA responds more quickly to recent price changes than SMA, making it useful for capturing short-term trends and momentum shifts.
---

## 8. Volume Weighted Average Price (VWAP)

VWAP is the ratio of cumulative (Typical Price × Volume) to cumulative Volume. It represents the average price weighted by trading activity.

* **Function**: `compute_vwap(high, low, close, volume)`
* **Mathematical Formula**:
  $$\text{Typical Price (TP)} = \frac{\text{High} + \text{Low} + \text{Close}}{3}$$
  $$\text{VWAP} = \frac{\sum_{i=1}^{t} \text{TP}_i \times \text{Volume}_i}{\sum_{i=1}^{t} \text{Volume}_i}$$

* **Interpretation**:
  * **Bullish signal**: Price trading above VWAP indicates buyers are in control.
  * **Bearish signal**: Price trading below VWAP indicates sellers are dominant.
  * **Fair value**: VWAP is frequently used by institutional traders as a benchmark for execution quality.
  * **Support/Resistance**: VWAP often acts as a dynamic intraday support or resistance level.

---

## 9. On-Balance Volume (OBV)

OBV is a cumulative volume-based momentum indicator that relates volume flow to price changes.

* **Function**: `compute_obv(close, volume)`
* **Mathematical Formula**:
  $$\text{OBV}_t = \text{OBV}_{t-1} +
  \begin{cases}
  +\text{Volume}_t & \text{if Close}_t > \text{Close}_{t-1} \\
  -\text{Volume}_t & \text{if Close}_t < \text{Close}_{t-1} \\
  0 & \text{if Close}_t = \text{Close}_{t-1}
  \end{cases}$$

* **Interpretation**:
  * **Trend confirmation**: Rising OBV alongside rising price confirms an uptrend.
  * **Divergence**: OBV rising while price falls (or vice versa) signals a potential reversal.
  * **Breakout validation**: A breakout accompanied by increasing OBV is considered more reliable.

---

## 10. Commodity Channel Index (CCI)

CCI measures how far the Typical Price has deviated from its rolling average, normalised by the mean absolute deviation.

* **Function**: `compute_cci(high, low, close, window=20, constant=0.015)`
* **Mathematical Formula**:
  $$\text{TP} = \frac{\text{High} + \text{Low} + \text{Close}}{3}$$
  $$\text{CCI} = \frac{\text{TP} - \text{SMA}(\text{TP}, N)}{0.015 \times \text{MAD}(\text{TP}, N)}$$
  * Where $\text{MAD} = \frac{1}{N}\sum_{i=0}^{N-1}|\text{TP}_{t-i} - \text{Mean}(\text{TP})|$ is the Mean Absolute Deviation.
  * The Lambert constant 0.015 ensures roughly 70-80% of CCI values fall within the ±100 range for normal distributions.

* **Interpretation**:
  * **Overbought (> +100)**: Asset may be extended to the upside; consider caution on new longs.
  * **Oversold (< -100)**: Asset may be oversold; potential for a mean-reversion rally.
  * **Zero crossing**: A CCI crossing above 0 from below signals emerging upward momentum.