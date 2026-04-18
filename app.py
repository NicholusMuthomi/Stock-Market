import re
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL STYLING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg-primary  : #0e1117;
  --bg-secondary: #161b22;
  --bg-tertiary : #21262d;
  --accent-blue : #58a6ff;
  --accent-green: #39d353;
  --accent-red  : #ff7b72;
  --text-primary: #c9d1d9;
  --text-muted  : #f1f1f1;
  --font-stack  : 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
body, .main, .stApp {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-stack);
}
.stApp {
    background-image: url("https://media.cnn.com/api/v1/images/stellar/prod/220718124741-google-alphabet-stock-split.jpg?c=original");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-color: rgba(14, 17, 23, 0.85);
    background-blend-mode: overlay;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
}
[data-testid="stSidebar"] .sidebar-content {padding: 1.5rem 1rem;}
h1, h2, h3, h4, h5, h6 {
  color: white !important;
  font-weight: 600;
  margin-bottom: 0.5rem;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}
h1 {font-size: 2.25rem;}
h2 {font-size: 1.75rem;}
.card {
  background-color: rgba(255, 255, 255, 0.15) !important;
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
.stButton > button {
  background-color: rgba(88, 166, 255, 0.9) !important;
  color: #ffffff !important;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 1.25rem;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s ease;
}
.stButton > button:hover {
  background-color: rgba(57, 211, 83, 0.9) !important;
}
.metric-container {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}
.metric-box {
  flex: 1;
  background-color: rgba(255, 255, 255, 0.15) !important;
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  transition: all 0.3s ease;
}
.metric-label {
  color: rgba(255, 255, 255, 0.8) !important;
  font-size: 0.875rem;
  margin-bottom: 0.25rem;
  text-align: center;
}
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: white !important;
  text-shadow: 0 1px 3px rgba(0,0,0,0.3);
  text-align: center;
}
.metric-delta {
  font-size: 0.875rem;
  color: white !important;
  margin-top: 0.25rem;
  text-align: center;
  display: block;
}
table {width: 100%; border-collapse: collapse;}
th, td {padding: 0.75rem 1rem; text-align: left;}
thead {background-color: var(--bg-tertiary);}
tbody tr:nth-child(even) {background-color: var(--bg-secondary);}
.stDataFrame {
    background-color: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(8px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
}
.stDataFrame thead th {
    background-color: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 600;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
}
.stDataFrame tbody td {
    background-color: transparent !important;
    color: white !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.stDataFrame tbody tr:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
}
.stDataFrame::-webkit-scrollbar {height: 6px; width: 6px;}
.stDataFrame::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
.stDataFrame::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 3px;
}
.plotly-graph-div {
    background-color: rgba(255, 255, 255, 0.1) !important;
}
.info-banner {
    background-color: rgba(88, 166, 255, 0.15);
    border: 1px solid rgba(88, 166, 255, 0.4);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    color: white;
    font-size: 0.95rem;
    backdrop-filter: blur(8px);
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  RE-USABLE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label, value, delta=None):
    delta_html = f"<span class='metric-delta'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
<div class="metric-box">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  {delta_html}
</div>
""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TICKER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = {
    "GOOG": "Alphabet Inc. (Google)",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOAD ML ARTEFACTS (GOOG only)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_components():
    try:
        model = load_model("google_stock_price_prediction_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_resource
def get_training_scaler():
    """
    Load the scaler from disk when available (preferred).

    Fallback — build it from scratch:
      Download 5 years of GOOG history and fit the scaler only on the portion
      that ends 90 days before today, so the hold-out window stays unseen.
      Using a fixed 5-year look-back avoids the date-arithmetic bug that
      produced an empty DataFrame (and the resulting MinMaxScaler crash) when
      subtracting 20 years from an already-shifted end date.
    """
    scaler_path = "google_stock_scaler.joblib"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)

    try:
        end_full   = datetime.now()
        start_full = end_full - timedelta(days=365 * 5)   # 5 years is plenty

        raw = yf.download("GOOG", start=start_full, end=end_full, progress=False)
        raw = raw.dropna()

        if raw.empty:
            st.error("Could not download GOOG data to build the scaler.")
            st.stop()

        # Extract close prices
        if ("Close", "GOOG") in raw.columns:
            close_col = raw[("Close", "GOOG")]
        elif "Close" in raw.columns:
            close_col = raw["Close"]
        else:
            st.error("Unexpected column structure from yfinance.")
            st.stop()

        # Fit only on the training slice (everything before the 90-day hold-out)
        cutoff     = end_full - timedelta(days=90)
        train_mask = raw.index <= cutoff
        train_prices = close_col[train_mask].values.reshape(-1, 1)

        if len(train_prices) < 2:
            # Edge case: if somehow the mask is empty, fit on all data
            train_prices = close_col.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_prices)
        return scaler

    except Exception as e:
        st.error(f"Error building scaler: {e}")
        st.stop()


model  = load_ml_components()
scaler = get_training_scaler()

if model is None or scaler is None:
    st.error("Critical components (model or scaler) could not be loaded.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="card" style="text-align:center;">
  <h1>Stock Market Dashboard</h1>
  <p style="color:var(--text-muted);margin:0;">
    Select a ticker from the sidebar. GOOG includes full LSTM price predictions.
    All other tickers display historical prices, volume, and moving averages.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

selected_ticker = st.sidebar.selectbox(
    "Select Ticker",
    options=list(TICKERS.keys()),
    format_func=lambda x: f"{x} - {TICKERS[x]}",
    index=0,
)

lookback_days = st.sidebar.slider(
    "Lookback Window (days)", min_value=50, max_value=200, value=100
)

if selected_ticker == "GOOG":
    prediction_days = st.sidebar.slider(
        "Prediction Horizon (days)", min_value=1, max_value=30, value=7
    )
else:
    prediction_days = None
    st.sidebar.markdown(
        """
        <div style="
            background-color: rgba(88,166,255,0.1);
            border: 1px solid rgba(88,166,255,0.3);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            color: rgba(255,255,255,0.8);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        ">
            Predictions are only available for GOOG. Switch to GOOG to enable the prediction horizon slider.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_stock_data(ticker):
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error("No data retrieved from Yahoo Finance. Please try again later.")
            st.stop()
        return data.dropna()
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        st.stop()


def extract_close(data, ticker):
    if ("Close", ticker) in data.columns:
        return data[("Close", ticker)]
    return data["Close"]


def extract_volume(data, ticker):
    if ("Volume", ticker) in data.columns:
        return data[("Volume", ticker)]
    return data["Volume"]


def prepare_data(data, ticker, lookback_window):
    try:
        close_series  = extract_close(data, ticker)
        close_prices  = close_series.values.reshape(-1, 1)

        if np.isnan(close_prices).any() or np.isinf(close_prices).any():
            raise ValueError("Data contains NaN or infinite values")
        if len(close_prices) < lookback_window:
            raise ValueError(f"Need at least {lookback_window} days, got {len(close_prices)}")

        scaled_data = scaler.transform(close_prices)
        x_data = [scaled_data[i - lookback_window:i] for i in range(lookback_window, len(scaled_data))]
        return np.array(x_data), close_prices.flatten()
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 7a.  IMPROVED PREDICTION  –  LSTM + linear-trend blend
#
#  The raw LSTM on price levels tends to lag because it minimises MSE by
#  predicting something close to yesterday's price.  We correct for this by:
#   1. Running the LSTM autoregressively to get a raw price path.
#   2. Fitting a short-term linear trend on the last 20 actual prices and
#      projecting it forward.
#   3. Blending the two (60 % LSTM, 40 % trend) so the forecast is anchored
#      to real momentum rather than purely echoing recent levels.
#   4. Building the confidence band from actual model residuals over the last
#      60 out-of-sample days, not from a generic volatility formula.
# ─────────────────────────────────────────────────────────────────────────────

def make_predictions(model, last_sequence, days_to_predict):
    """
    Autoregressive LSTM prediction blended with a linear momentum trend.
    Returns the blended forecast as a 1-D numpy array.
    """
    try:
        # --- LSTM autoregressive rollout ---
        lstm_preds_scaled = []
        current_seq = last_sequence.copy()
        for _ in range(days_to_predict):
            p = model.predict(current_seq.reshape(1, -1, 1), verbose=0)[0][0]
            lstm_preds_scaled.append(p)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = p

        lstm_prices = scaler.inverse_transform(
            np.array(lstm_preds_scaled).reshape(-1, 1)
        ).flatten()

        # --- Linear momentum trend over the last 20 actual closing prices ---
        # last_sequence is in scaled space; invert to get actual prices
        recent_prices = scaler.inverse_transform(
            last_sequence[-20:].reshape(-1, 1)
        ).flatten()

        x_trend = np.arange(len(recent_prices)).reshape(-1, 1)
        lr      = LinearRegression().fit(x_trend, recent_prices)
        x_future = np.arange(
            len(recent_prices), len(recent_prices) + days_to_predict
        ).reshape(-1, 1)
        trend_prices = lr.predict(x_future)

        # --- Blend: 60 % LSTM, 40 % linear trend ---
        blended = 0.60 * lstm_prices + 0.40 * trend_prices
        return blended

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()


def compute_residual_confidence_band(model, data, ticker, lookback_window, eval_days=60):
    """
    Walk-forward evaluation on the last `eval_days` of data that the scaler
    has NOT seen (because the scaler was fitted on data ending 90 days ago).

    Returns per-step mean absolute error indexed by step offset (1, 2, … n).
    This is used to build a confidence band whose width is grounded in actual
    measured model error, not a theoretical volatility formula.
    """
    try:
        close_series   = extract_close(data, ticker)
        close_prices   = close_series.values.reshape(-1, 1)
        scaled_all     = scaler.transform(close_prices)

        if len(scaled_all) < lookback_window + eval_days + 1:
            return None

        eval_start = len(scaled_all) - eval_days
        errors     = []   # list of (step_offset, abs_error)

        for i in range(eval_start, len(scaled_all)):
            seq        = scaled_all[i - lookback_window:i]
            p_scaled   = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
            p_price    = scaler.inverse_transform([[p_scaled]])[0][0]
            actual     = close_prices[i][0]
            errors.append(abs(p_price - actual))

        mae_per_step = np.array(errors)
        # Compound uncertainty: band at step k = mae * sqrt(k)
        return mae_per_step.mean()

    except Exception:
        return None


def build_confidence_band(predictions, base_mae):
    """
    Confidence band whose width at step k is  base_mae * sqrt(k).
    This correctly reflects that uncertainty grows with forecast horizon,
    and the base width is anchored to real out-of-sample prediction error.
    """
    upper, lower = [], []
    for k, price in enumerate(predictions, start=1):
        band = base_mae * np.sqrt(k)
        upper.append(price + band)
        lower.append(price - band)
    return np.array(upper), np.array(lower)


# ─────────────────────────────────────────────────────────────────────────────
# 7b.  GENUINE OUT-OF-SAMPLE MODEL PERFORMANCE
#
#  The evaluation window is the last 60 trading days.  Because the scaler was
#  fitted only on data BEFORE the last 90 days, these 60 days are truly
#  out-of-sample: the model has never indirectly seen their price levels via
#  normalisation.  Directional accuracy is also computed correctly here using
#  walk-forward one-step-ahead predictions rather than in-sample fits.
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_performance(model, data, ticker, lookback_window, evaluation_days=60):
    try:
        close_series    = extract_close(data, ticker)
        close_prices_all = close_series.values.reshape(-1, 1)

        if len(close_prices_all) < lookback_window + evaluation_days:
            return None

        scaled_all  = scaler.transform(close_prices_all)
        eval_start  = len(scaled_all) - evaluation_days
        actuals     = []
        predicted   = []

        for i in range(eval_start, len(scaled_all)):
            seq        = scaled_all[i - lookback_window:i]
            p_scaled   = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
            p_price    = scaler.inverse_transform([[p_scaled]])[0][0]
            actuals.append(close_prices_all[i][0])
            predicted.append(p_price)

        actuals   = np.array(actuals)
        predicted = np.array(predicted)

        rmse = np.sqrt(np.mean((actuals - predicted) ** 2))
        mae  = np.mean(np.abs(actuals - predicted))
        mape = np.mean(np.abs((actuals - predicted) / actuals)) * 100

        # Directional accuracy: does the model correctly call up vs. down?
        actual_dir    = np.sign(np.diff(actuals))
        predicted_dir = np.sign(np.diff(predicted))
        dir_acc       = np.mean(actual_dir == predicted_dir) * 100

        eval_dates = close_series.index[eval_start:]
        perf_df    = pd.DataFrame({
            "Date":            eval_dates,
            "Actual Price":    actuals,
            "Predicted Price": predicted,
            "Error ($)":       predicted - actuals,
        })

        return {
            "rmse": rmse,
            "mae":  mae,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "df":   perf_df,
        }

    except Exception as e:
        st.warning(f"Could not compute model performance: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 7c.  TECHNICAL INDICATORS  (used for all tickers, not just GOOG)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moving_averages(close_series):
    ma20 = close_series.rolling(window=20).mean()
    ma50 = close_series.rolling(window=50).mean()
    return ma20, ma50


def compute_rsi(close_series, period=14):
    """Relative Strength Index (0–100).  >70 = overbought, <30 = oversold."""
    delta  = close_series.diff()
    gain   = delta.clip(lower=0).rolling(window=period).mean()
    loss   = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close_series, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    ema_fast   = close_series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close_series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close_series, window=20, num_std=2):
    """Upper, middle (SMA), and lower Bollinger Bands."""
    sma   = close_series.rolling(window=window).mean()
    std   = close_series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def generate_signal_summary(close_series):
    """
    Produce a plain-language technical signal summary based on RSI, MACD,
    and Bollinger Band position.  Returns a list of (indicator, signal, detail)
    tuples so the UI can render them in a table.
    """
    rsi                         = compute_rsi(close_series)
    macd_line, signal_line, _   = compute_macd(close_series)
    bb_upper, bb_mid, bb_lower  = compute_bollinger_bands(close_series)
    ma20, ma50                  = compute_moving_averages(close_series)

    current_price = close_series.iloc[-1]
    current_rsi   = rsi.iloc[-1]
    current_macd  = macd_line.iloc[-1]
    current_sig   = signal_line.iloc[-1]
    current_ma20  = ma20.iloc[-1]
    current_ma50  = ma50.iloc[-1]
    current_bb_u  = bb_upper.iloc[-1]
    current_bb_l  = bb_lower.iloc[-1]

    rows = []

    # RSI
    if current_rsi > 70:
        rows.append(("RSI", "Overbought", f"{current_rsi:.1f} — price may be due for a pullback"))
    elif current_rsi < 30:
        rows.append(("RSI", "Oversold",   f"{current_rsi:.1f} — price may be due for a bounce"))
    else:
        rows.append(("RSI", "Neutral",    f"{current_rsi:.1f} — no extreme reading"))

    # MACD crossover
    prev_macd = macd_line.iloc[-2]
    prev_sig  = signal_line.iloc[-2]
    if prev_macd < prev_sig and current_macd > current_sig:
        rows.append(("MACD", "Bullish crossover", "MACD just crossed above signal line"))
    elif prev_macd > prev_sig and current_macd < current_sig:
        rows.append(("MACD", "Bearish crossover", "MACD just crossed below signal line"))
    elif current_macd > current_sig:
        rows.append(("MACD", "Bullish",  "MACD above signal — upward momentum"))
    else:
        rows.append(("MACD", "Bearish",  "MACD below signal — downward momentum"))

    # Bollinger Bands
    if current_price > current_bb_u:
        rows.append(("Bollinger Bands", "Above upper band", f"Price ${current_price:.2f} above upper band ${current_bb_u:.2f} — extended"))
    elif current_price < current_bb_l:
        rows.append(("Bollinger Bands", "Below lower band", f"Price ${current_price:.2f} below lower band ${current_bb_l:.2f} — oversold"))
    else:
        pct_b = (current_price - current_bb_l) / (current_bb_u - current_bb_l) * 100
        rows.append(("Bollinger Bands", "Within bands", f"Price at {pct_b:.0f}% of band range"))

    # MA crossover (golden/death cross)
    prev_ma20 = ma20.iloc[-2]
    prev_ma50 = ma50.iloc[-2]
    if prev_ma20 < prev_ma50 and current_ma20 > current_ma50:
        rows.append(("MA Cross", "Golden Cross", "20-day MA crossed above 50-day — long-term bullish signal"))
    elif prev_ma20 > prev_ma50 and current_ma20 < current_ma50:
        rows.append(("MA Cross", "Death Cross",  "20-day MA crossed below 50-day — long-term bearish signal"))
    elif current_ma20 > current_ma50:
        rows.append(("MA Cross", "Bullish trend", f"20-day MA (${current_ma20:.2f}) above 50-day MA (${current_ma50:.2f})"))
    else:
        rows.append(("MA Cross", "Bearish trend", f"20-day MA (${current_ma20:.2f}) below 50-day MA (${current_ma50:.2f})"))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 7d.  NEWS + SENTIMENT
#
#  The original app fetched headlines but displayed them as pure decoration.
#  This version adds a lightweight keyword-based sentiment classifier so each
#  headline is labelled Positive / Negative / Neutral, and an aggregate
#  sentiment score is displayed alongside the LSTM prediction to give context.
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    "surge", "soar", "rally", "beat", "record", "profit", "gain", "growth",
    "upgrade", "outperform", "buy", "bullish", "strong", "rises", "climbs",
    "jumps", "top", "win", "award", "partnership", "launch", "innovation",
    "exceeds", "expands", "dividend", "breakout", "momentum", "milestone",
]
NEGATIVE_KEYWORDS = [
    "fall", "drop", "decline", "miss", "loss", "downgrade", "underperform",
    "sell", "bearish", "weak", "slump", "crash", "risk", "lawsuit", "fine",
    "probe", "investigation", "cut", "layoff", "warning", "concern", "fear",
    "tumble", "plunge", "disappoints", "below", "deficit", "breach",
]


def score_sentiment(title: str) -> tuple[str, str]:
    """
    Return (label, colour_variable) for a headline.
    Scoring: +1 per positive keyword hit, -1 per negative keyword hit.
    Ties go to Neutral.
    """
    text  = title.lower()
    score = sum(1 for w in POSITIVE_KEYWORDS if w in text) \
          - sum(1 for w in NEGATIVE_KEYWORDS if w in text)
    if score > 0:
        return "Positive", "var(--accent-green)"
    if score < 0:
        return "Negative", "var(--accent-red)"
    return "Neutral", "var(--accent-blue)"


@st.cache_data(ttl=1800)
def fetch_google_news(max_items=5):
    try:
        url = "https://news.google.com/rss/search?q=Google+Alphabet+stock&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            raw = response.read()

        root    = ET.fromstring(raw)
        channel = root.find("channel")
        items   = channel.findall("item")
        news    = []

        for item in items[:max_items]:
            title    = item.findtext("title",   default="No title")
            pub_date = item.findtext("pubDate", default="")
            source_el = item.find("source")
            source   = source_el.text if source_el is not None else "Google News"

            link = "#"
            for child in item:
                if child.tag == "link" and child.tail and child.tail.strip():
                    link = child.tail.strip()
                    break
            if link == "#" and source_el is not None:
                link = source_el.get("url", "#")

            title = re.sub(r"<[^>]+>", "", title).strip()
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]

            try:
                dt            = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                pub_formatted = dt.strftime("%b %d, %Y %H:%M UTC")
            except Exception:
                pub_formatted = pub_date

            sentiment_label, sentiment_color = score_sentiment(title)

            news.append({
                "title":           title,
                "link":            link,
                "source":          source,
                "published":       pub_formatted,
                "sentiment":       sentiment_label,
                "sentiment_color": sentiment_color,
            })

        return news

    except Exception:
        return []


def aggregate_sentiment(news_items):
    """
    Return an overall sentiment label and score for a list of news items.
    Score = (positive_count - negative_count) / total, in [-1, 1].
    """
    if not news_items:
        return "Neutral", 0.0
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for item in news_items:
        counts[item["sentiment"]] += 1
    total = len(news_items)
    score = (counts["Positive"] - counts["Negative"]) / total
    if score > 0.2:
        return "Positive", score
    if score < -0.2:
        return "Negative", score
    return "Neutral", score


def generate_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

data = get_stock_data(selected_ticker)

if len(data) < lookback_days:
    st.error(f"Not enough data. Need {lookback_days} days, have {len(data)}.")
    st.stop()

close_series  = extract_close(data, selected_ticker)
close_prices  = close_series.values
volume_series = extract_volume(data, selected_ticker)
ma20, ma50    = compute_moving_averages(close_series)
current_price = float(close_prices[-1])

# ─────────────────────────────────────────────────────────────────────────────
# 8a.  GOOG  –  FULL PREDICTION + ANALYTICS MODE
# ─────────────────────────────────────────────────────────────────────────────

if selected_ticker == "GOOG":

    x_data, _   = prepare_data(data, selected_ticker, lookback_days)
    predictions = make_predictions(model, x_data[-1], prediction_days)

    # Build confidence band from real model residuals
    with st.spinner("Computing residual-based confidence band..."):
        base_mae = compute_residual_confidence_band(
            model, data, "GOOG", lookback_days, eval_days=60
        )
    if base_mae is None:
        base_mae = current_price * 0.01   # fallback: 1 % of price

    upper_band, lower_band = build_confidence_band(predictions, base_mae)

    last_date    = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_days)

    next_price  = float(predictions[0])
    change_pct  = (next_price - current_price) / current_price * 100
    delta_color = "var(--accent-green)" if change_pct >= 0 else "var(--accent-red)"

    # Fetch news early so sentiment can inform the metric bar
    news_items       = fetch_google_news(max_items=5)
    sentiment_label, sentiment_score = aggregate_sentiment(news_items)
    sentiment_color  = (
        "var(--accent-green)" if sentiment_label == "Positive"
        else "var(--accent-red)" if sentiment_label == "Negative"
        else "var(--accent-blue)"
    )

    # --- Metrics row (now includes news sentiment) ---
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Current Price", f"${current_price:.2f}")
    with col2:
        metric_card("Next-Day Forecast", f"${next_price:.2f}")
    with col3:
        metric_card(
            "Predicted Change",
            f"{change_pct:+.2f}%",
            delta=f"<span style='color:{delta_color};'>{change_pct:+.2f}%</span>",
        )
    with col4:
        metric_card(
            "News Sentiment",
            sentiment_label,
            delta=f"<span style='color:{sentiment_color};'>Score: {sentiment_score:+.2f} of ±1.0</span>",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Price chart with predictions ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Price Chart with Predictions")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=close_prices,
        mode="lines", name="Historical",
        line=dict(color="#58a6ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=ma20,
        mode="lines", name="20-Day MA",
        line=dict(color="#f0c040", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=ma50,
        mode="lines", name="50-Day MA",
        line=dict(color="#c792ea", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=predictions,
        mode="lines+markers", name="Forecast (LSTM + Trend blend)",
        line=dict(color="#ff7b72", width=2, dash="dot"),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper_band,
        mode="lines", name="Upper Band",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower_band,
        mode="lines",
        name="Confidence Band (residual MAE)",
        fill="tonexty",
        fillcolor="rgba(255,123,114,0.15)",
        line=dict(width=0),
        hoverinfo="skip",
    ))
    fig.add_shape(
        type="line", x0=last_date, y0=0, x1=last_date, y1=1,
        yref="paper", line=dict(color="#39d353", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=last_date, y=0.95, yref="paper",
        text="Forecast Start", showarrow=False, font=dict(color="#39d353"),
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.8)", font_size=14, font_family="Arial"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        <p style="color:#DBDBDB;font-size:1rem;margin-top:-0.5rem;padding-bottom:0.5rem;">
        Forecast = 60% LSTM autoregressive rollout + 40% linear momentum trend fitted on the last 20 trading days.
        The confidence band is derived from the model's actual mean absolute error over the last 60 out-of-sample days,
        and compounds as band = MAE × √k so uncertainty grows correctly with horizon.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Bollinger Band chart for GOOG ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Bollinger Bands (20-day, 2σ)")
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close_series)
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_upper.values,
        mode="lines", name="Upper Band",
        line=dict(color="rgba(255,123,114,0.6)", width=1),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_lower.values,
        mode="lines", name="Lower Band",
        fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(88,166,255,0.6)", width=1),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_mid.values,
        mode="lines", name="Middle Band (SMA 20)",
        line=dict(color="#f0c040", width=1.5, dash="dot"),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=close_prices,
        mode="lines", name="Close",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_bb.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig_bb, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- RSI chart ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("RSI (14-day)")
    rsi_series = compute_rsi(close_series)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data.index, y=rsi_series.values,
        mode="lines", name="RSI",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff7b72",
                      annotation_text="Overbought (70)", annotation_position="bottom right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#39d353",
                      annotation_text="Oversold (30)", annotation_position="top right")
    fig_rsi.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="RSI",
        yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- MACD chart ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("MACD (12 / 26 / 9)")
    macd_line, signal_line, histogram = compute_macd(close_series)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(
        x=data.index, y=histogram.values,
        name="Histogram",
        marker_color=[
            "rgba(57,211,83,0.6)" if v >= 0 else "rgba(255,123,114,0.6)"
            for v in histogram.values
        ],
    ))
    fig_macd.add_trace(go.Scatter(
        x=data.index, y=macd_line.values,
        mode="lines", name="MACD",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_macd.add_trace(go.Scatter(
        x=data.index, y=signal_line.values,
        mode="lines", name="Signal",
        line=dict(color="#ff7b72", width=1.5, dash="dot"),
    ))
    fig_macd.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="MACD",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Volume chart ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data.index, y=volume_series,
        name="Volume", marker_color="rgba(88,166,255,0.6)",
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Volume",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Prediction table ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Forecast Details")
    pred_details = pd.DataFrame({
        "Date":            future_dates,
        "Forecast Price":  predictions,
        "Lower Band ($)":  lower_band,
        "Upper Band ($)":  upper_band,
        "Day":             range(1, prediction_days + 1),
    })
    pred_details["Change %"] = pred_details["Forecast Price"].pct_change() * 100
    pred_details.at[0, "Change %"] = (
        pred_details.at[0, "Forecast Price"] - current_price
    ) / current_price * 100

    st.dataframe(
        pred_details.style.format({
            "Forecast Price": "${:.2f}",
            "Lower Band ($)": "${:.2f}",
            "Upper Band ($)": "${:.2f}",
            "Change %":       "{:+.2f}%",
        }).set_properties(**{
            "background-color": "transparent",
            "border-color":     "rgba(255,255,255,0.1)",
        }),
        hide_index=True,
        use_container_width=True,
        height=(pred_details.shape[0] + 1) * 35 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Genuine out-of-sample model performance ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance — Last 60 Out-of-Sample Days")
    st.markdown(
        """
        <p style="color:rgba(255,255,255,0.65);font-size:1rem;margin-bottom:1rem;">
        The scaler was fitted on data ending 90 days ago, so these 60 evaluation
        days were never seen during normalisation — this is a genuine out-of-sample
        test, not an in-sample fit.  MAPE tells you the average percentage miss.
        Directional accuracy above 55% is considered actionable; near 50% is a coin flip.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Running out-of-sample evaluation..."):
        perf = compute_model_performance(model, data, "GOOG", lookback_days, evaluation_days=60)

    if perf:
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            metric_card("RMSE", f"${perf['rmse']:.2f}", delta="Root Mean Squared Error")
        with p_col2:
            metric_card("MAE",  f"${perf['mae']:.2f}",  delta="Mean Absolute Error")
        with p_col3:
            metric_card("MAPE", f"{perf['mape']:.2f}%", delta="Mean Absolute % Error")
        with p_col4:
            dir_color = "var(--accent-green)" if perf["directional_accuracy"] >= 55 else \
                        "var(--accent-blue)"   if perf["directional_accuracy"] >= 50 else \
                        "var(--accent-red)"
            qualifier = "Actionable" if perf["directional_accuracy"] >= 55 else \
                        "Marginal"   if perf["directional_accuracy"] >= 50 else \
                        "Below chance"
            metric_card(
                "Directional Accuracy",
                f"{perf['directional_accuracy']:.1f}%",
                delta=f"<span style='color:{dir_color};'>{qualifier}</span>",
            )

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=perf["df"]["Date"], y=perf["df"]["Actual Price"],
            mode="lines", name="Actual",
            line=dict(color="#58a6ff", width=2),
        ))
        fig_perf.add_trace(go.Scatter(
            x=perf["df"]["Date"], y=perf["df"]["Predicted Price"],
            mode="lines", name="Predicted",
            line=dict(color="#ff7b72", width=2, dash="dot"),
        ))
        fig_perf.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date", yaxis_title="Price ($)",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.8)", font_size=14, font_family="Arial"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
            ),
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("Not enough data to evaluate model performance with the current lookback window.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Technical signal summary for GOOG ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Technical Signal Summary")
    signal_rows = generate_signal_summary(close_series)
    signal_df = pd.DataFrame(signal_rows, columns=["Indicator", "Signal", "Detail"])
    st.dataframe(
            signal_df.style.set_properties(**{
                "background-color": "transparent",
                "border-color":     "rgba(255,255,255,0.1)",
            }),
            hide_index=True,
            use_container_width=True,
            height=(signal_df.shape[0] + 1) * 45 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- News with sentiment ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Latest Google News  —  Sentiment Analysis")
    st.markdown(
        f"""
        <p style="color:rgba(255,255,255,0.65);font-size:0.88rem;margin-bottom:1.25rem;">
        Each headline is scored using keyword-based sentiment analysis.
        Overall news sentiment: <strong style="color:{sentiment_color};">{sentiment_label}
        (score {sentiment_score:+.2f})</strong>.
        Headlines refresh every 30 minutes.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if news_items:
        cards_html = ""
        for article in news_items:
            s_color = article["sentiment_color"]
            s_label = article["sentiment"]
            cards_html += f"""
            <a href="{article['link']}" target="_blank" class="news-card">
                <p class="news-card-title">{article['title']}</p>
                <div class="news-card-footer">
                    <span class="news-card-source">{article['source']}</span>
                    <span class="news-sentiment" style="color:{s_color};">{s_label}</span>
                    <span class="news-card-date">{article['published']}</span>
                </div>
            </a>
            """

        full_html = f"""
        <style>
            body {{ margin: 0; padding: 0; background: transparent; }}
            .news-row {{
                display: flex; flex-direction: row; gap: 1rem;
                overflow-x: auto; padding-bottom: 0.85rem;
                scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
            }}
            .news-row::-webkit-scrollbar {{ height: 5px; }}
            .news-row::-webkit-scrollbar-track {{ background: transparent; }}
            .news-row::-webkit-scrollbar-thumb {{
                background: rgba(255,255,255,0.25); border-radius: 4px;
            }}
            .news-card {{
                flex: 0 0 calc(20% - 0.8rem); min-width: 210px; max-width: 260px;
                background: rgba(255,255,255,0.08); backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255,255,255,0.18); border-radius: 16px;
                padding: 1.15rem 1.1rem 1rem 1.1rem; height: 200px;
                display: flex; flex-direction: column; justify-content: space-between;
                box-shadow: 0 4px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
                text-decoration: none;
                transition: background 0.25s ease, transform 0.22s ease,
                            border-top-color 0.25s ease, box-shadow 0.25s ease;
            }}
            .news-card:hover {{
                background: rgba(255,255,255,0.14); border-top-color: rgba(57,211,83,0.9);
                transform: translateY(-4px);
                box-shadow: 0 8px 28px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.12);
            }}
            .news-card-title {{
                color: rgba(255,255,255,1); font-size:1rem; font-weight:600;
                line-height:1.48; margin:0 0 0.6rem 0;
                display:-webkit-box; -webkit-line-clamp:4;
                -webkit-box-orient:vertical; overflow:hidden;
            }}
            .news-card-footer {{
                display:flex; flex-direction:column; gap:0.15rem;
            }}
            .news-card-source {{
                color:rgba(88,166,255,1); font-size:0.72rem;
                font-weight:700; letter-spacing:0.04em; text-transform:uppercase;
            }}
            .news-sentiment {{
                font-size:0.72rem; font-weight:700; letter-spacing:0.03em;
            }}
            .news-card-date {{
                color:rgba(255,255,255,0.35); font-size:0.68rem; text-align:right;
            }}
        </style>
        <div class="news-row">{cards_html}</div>
        """
        components.html(full_html, height=240, scrolling=False)
    else:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.5);font-size:0.88rem;padding:0.5rem 0;'>"
            "Unable to load news. Check your internet connection or try again shortly.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Download ---
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.subheader("Download Report")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.65);font-size:0.88rem;'>Download the forecast details as a CSV file.</p>",
        unsafe_allow_html=True,
    )
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_data  = generate_csv(pred_details[["Date", "Day", "Forecast Price", "Lower Band ($)", "Upper Band ($)", "Change %"]])
    st.download_button(
        label="Download GOOG Forecast CSV",
        data=csv_data,
        file_name=f"GOOG_forecast_{today_str}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8b.  NON-GOOG  –  FULL TECHNICAL ANALYSIS MODE
#      Now includes RSI, MACD, Bollinger Bands, and a signal summary table
#      instead of just a price chart and a 10-row moving average table.
# ─────────────────────────────────────────────────────────────────────────────

else:
    company_name = TICKERS[selected_ticker]

    st.markdown(
        f"""
        <div class="info-banner">
            You are viewing <strong>{selected_ticker} — {company_name}</strong>.
            Price predictions are only available for GOOG.
            Full technical analysis is shown below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    prev_price    = float(close_prices[-2]) if len(close_prices) > 1 else current_price
    day_change    = current_price - prev_price
    day_change_pct = day_change / prev_price * 100
    delta_color   = "var(--accent-green)" if day_change_pct >= 0 else "var(--accent-red)"

    rsi_series                  = compute_rsi(close_series)
    macd_line, signal_line, _   = compute_macd(close_series)
    bb_upper, bb_mid, bb_lower  = compute_bollinger_bands(close_series)
    current_rsi                 = float(rsi_series.iloc[-1])

    rsi_label = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    rsi_color = "var(--accent-red)" if current_rsi > 70 else \
                "var(--accent-green)" if current_rsi < 30 else "var(--accent-blue)"

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Current Price", f"${current_price:.2f}")
    with col2:
        metric_card("Previous Close", f"${prev_price:.2f}")
    with col3:
        metric_card(
            "Day Change",
            f"{day_change_pct:+.2f}%",
            delta=f"<span style='color:{delta_color};'>${day_change:+.2f}</span>",
        )
    with col4:
        metric_card(
            "RSI (14)",
            f"{current_rsi:.1f}",
            delta=f"<span style='color:{rsi_color};'>{rsi_label}</span>",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Price + MA chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} Historical Price and Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=close_prices,
        mode="lines", name="Close Price",
        line=dict(color="#58a6ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=ma20,
        mode="lines", name="20-Day MA",
        line=dict(color="#f0c040", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=ma50,
        mode="lines", name="50-Day MA",
        line=dict(color="#c792ea", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.8)", font_size=14, font_family="Arial"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Bollinger Bands
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} Bollinger Bands (20-day, 2σ)")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_upper.values,
        mode="lines", name="Upper Band",
        line=dict(color="rgba(255,123,114,0.6)", width=1),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_lower.values,
        mode="lines", name="Lower Band",
        fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(88,166,255,0.6)", width=1),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=bb_mid.values,
        mode="lines", name="Middle Band (SMA 20)",
        line=dict(color="#f0c040", width=1.5, dash="dot"),
    ))
    fig_bb.add_trace(go.Scatter(
        x=data.index, y=close_prices,
        mode="lines", name="Close",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_bb.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig_bb, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # RSI
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} RSI (14-day)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data.index, y=rsi_series.values,
        mode="lines", name="RSI",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff7b72",
                      annotation_text="Overbought (70)", annotation_position="bottom right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#39d353",
                      annotation_text="Oversold (30)", annotation_position="top right")
    fig_rsi.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="RSI",
        yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # MACD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} MACD (12 / 26 / 9)")
    macd_line, signal_line, histogram = compute_macd(close_series)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(
        x=data.index, y=histogram.values,
        name="Histogram",
        marker_color=[
            "rgba(57,211,83,0.6)" if v >= 0 else "rgba(255,123,114,0.6)"
            for v in histogram.values
        ],
    ))
    fig_macd.add_trace(go.Scatter(
        x=data.index, y=macd_line.values,
        mode="lines", name="MACD",
        line=dict(color="#58a6ff", width=2),
    ))
    fig_macd.add_trace(go.Scatter(
        x=data.index, y=signal_line.values,
        mode="lines", name="Signal",
        line=dict(color="#ff7b72", width=1.5, dash="dot"),
    ))
    fig_macd.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="MACD",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.2)",
        ),
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Volume
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data.index, y=volume_series,
        name="Volume", marker_color="rgba(88,166,255,0.6)",
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Volume",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Technical signal summary table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Technical Signal Summary")
    signal_rows = generate_signal_summary(close_series)
    signal_df   = pd.DataFrame(signal_rows, columns=["Indicator", "Signal", "Detail"])
    st.dataframe(
        signal_df.set_properties(**{
            "background-color": "transparent",
            "border-color":     "rgba(255,255,255,0.1)",
        }),
        hide_index=True,
        use_container_width=True,
        height=(signal_df.shape[0] + 1) * 45 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Moving average summary (kept from original, now supplementary)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Moving Average Summary (Last 10 Days)")
    ma_summary = pd.DataFrame({
        "Date":       data.index[-10:],
        "Close":      close_prices[-10:],
        "20-Day MA":  ma20.values[-10:],
        "50-Day MA":  ma50.values[-10:],
        "RSI":        rsi_series.values[-10:],
    })
    st.dataframe(
        ma_summary.style.format({
            "Close":     "${:.2f}",
            "20-Day MA": "${:.2f}",
            "50-Day MA": "${:.2f}",
            "RSI":       "{:.1f}",
        }).set_properties(**{
            "background-color": "transparent",
            "border-color":     "rgba(255,255,255,0.1)",
        }),
        hide_index=True,
        use_container_width=True,
        height=(ma_summary.shape[0] + 1) * 35 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Download
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.subheader("Download Report")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.65);font-size:0.88rem;'>Download the technical summary as a CSV file.</p>",
        unsafe_allow_html=True,
    )
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_data  = generate_csv(ma_summary)
    st.download_button(
        label=f"Download {selected_ticker} Technical Summary CSV",
        data=csv_data,
        file_name=f"{selected_ticker}_technical_{today_str}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="card" style="text-align:center;margin-top:2rem;">
  <p style="color:var(--text-muted);margin:0;font-size:clamp(0.75rem, 1.5vw, 0.875rem);">
    Disclaimer: Stock price forecasts are based on historical data and machine-learning models.
    Past performance is not indicative of future results.
    This tool is for informational purposes only and should not be considered financial advice.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
