import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# 1.  GLOBAL STYLING
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

st.markdown(
    """
<style>
/* --- Color palette & fonts --- */
:root {
  --bg-primary  : #0e1117;
  --bg-secondary: #161b22;
  --bg-tertiary : #21262d;
  --accent-blue : #58a6ff;
  --accent-green: #39d353;
  --accent-red  : #ff7b72;
  --text-primary: #c9d1d9;
  --text-muted  : #8b949e;
  --font-stack  : 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- Global overrides --- */
body, .main, .stApp {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-stack);
}

/* --- Sidebar --- */
.css-1d391kg {background-color: var(--bg-secondary);}
.sidebar .sidebar-content {padding: 1.5rem 1rem;}

/* --- Headers --- */
h1, h2, h3, h4, h5, h6 {
  color: var(--text-primary);
  font-weight: 600;
  margin-bottom: 0.5rem;
}
h1 {font-size: 2.25rem;}
h2 {font-size: 1.75rem;}

/* --- Cards / containers --- */
.card {
  background-color: var(--bg-secondary);
  border: 1px solid var(--bg-tertiary);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

/* --- Buttons --- */
.stButton > button {
  background-color: var(--accent-blue);
  color: #ffffff;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 1.25rem;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s ease;
}
.stButton > button:hover {
  background-color: var(--accent-green);
}

/* --- Metric boxes --- */
.metric-container {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}
.metric-box {
  flex: 1;
  background-color: var(--bg-secondary);
  border: 1px solid var(--bg-tertiary);
  border-radius: 8px;
  padding: 1rem 1.25rem;
}
.metric-label {
  color: var(--text-muted);
  font-size: 0.875rem;
}
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
}
.metric-delta {
  font-size: 0.875rem;
}

/* --- Tables --- */
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  padding: 0.75rem 1rem;
  text-align: left;
}
thead {
  background-color: var(--bg-tertiary);
}
tbody tr:nth-child(even) {background-color: var(--bg-secondary);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* --- Background image --- */
.stApp {
    background-image: url("https://media.cnn.com/api/v1/images/stellar/prod/220718124741-google-alphabet-stock-split.jpg?c=original");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-color: rgba(14, 17, 23, 0.85);
    background-blend-mode: overlay;
}

/* --- Cards --- */
.card {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* --- Sidebar --- */
.css-1d391kg {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* --- Charts --- */
.plotly-graph-div {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* --- Tables --- */
.stDataFrame {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* --- Text contrast --- */
h1, h2, h3, h4, h5, h6, .metric-value, .metric-label {
    color: white !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

/* --- Buttons --- */
.stButton > button {
    background-color: rgba(88, 166, 255, 0.9) !important;
    color: white !important;
}
.stButton > button:hover {
    background-color: rgba(57, 211, 83, 0.9) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* --- Tables --- */
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

.stDataFrame::-webkit-scrollbar {
    height: 6px;
    width: 6px;
}
.stDataFrame::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
.stDataFrame::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 3px;
}
</style>
""",
    unsafe_allow_html=True,
)

# 2.  RE-USABLE COMPONENTS

def metric_card(label, value, delta=None):
    """Render a single metric box."""
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

st.markdown(
    """
<style>
.metric-box {
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
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: white !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.metric-delta {
    font-size: 0.875rem;
    color: white !important;
    margin-top: 0.25rem;
}

/* --- Info banner for non-GOOG tickers --- */
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

# 3.  TICKER CONFIGURATION

TICKERS = {
    "GOOG": "Alphabet Inc. (Google)",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
}

GOOG_ONLY_PREDICTION = True  # Only GOOG has a trained model

# 4.  LOAD ML ARTEFACTS (GOOG only)

@st.cache_resource
def load_ml_components():
    try:
        model = load_model("google_stock_price_prediction_model.keras", compile=False)
        model.save("google_stock_price_prediction_model.keras", save_format="keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def get_training_scaler():
    """
    Reproduce the scaler exactly as the notebook did:
    - Download 20 years of GOOG Close prices
    - Fit MinMaxScaler(feature_range=(0, 1)) on that single column
    This guarantees the scaler always expects exactly 1 feature.
    """
    try:
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        training_data = yf.download("GOOG", start=start, end=end, progress=False)
        training_data = training_data.dropna()

        if ("Close", "GOOG") in training_data.columns:
            close_prices = training_data[("Close", "GOOG")]
        else:
            close_prices = training_data["Close"]

        price_data = close_prices.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(price_data)
        return scaler
    except Exception as e:
        st.error(f"Error building scaler: {e}")
        st.stop()

model = load_ml_components()
scaler = get_training_scaler()

# 5.  HEADER
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

# 6.  SIDEBAR
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

# 7.  DATA PIPELINE

@st.cache_data
def get_stock_data(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error("No data retrieved from Yahoo Finance. Please try again later.")
            st.stop()

        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        st.stop()


def extract_close(data, ticker):
    """Safely extract the Close price series regardless of column structure."""
    if (("Close", ticker)) in data.columns:
        return data[("Close", ticker)]
    return data["Close"]


def extract_volume(data, ticker):
    """Safely extract the Volume series regardless of column structure."""
    if (("Volume", ticker)) in data.columns:
        return data[("Volume", ticker)]
    return data["Volume"]


def prepare_data(data, ticker, lookback_window):
    try:
        close_series = extract_close(data, ticker)
        close_prices = close_series.values.reshape(-1, 1)

        if np.isnan(close_prices).any() or np.isinf(close_prices).any():
            raise ValueError("Data contains NaN or infinite values")

        if len(close_prices) < lookback_window:
            raise ValueError(
                f"Not enough data. Need at least {lookback_window} days, but got {len(close_prices)}"
            )

        scaled_data = scaler.transform(close_prices)

        x_data = []
        for i in range(lookback_window, len(scaled_data)):
            x_data.append(scaled_data[i - lookback_window : i])

        return np.array(x_data), close_prices.flatten()

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        st.stop()


def make_predictions(model, last_sequence, days_to_predict):
    try:
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days_to_predict):
            next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)[0][0]
            predictions.append(next_pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred

        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()


def compute_moving_averages(close_series):
    ma20 = close_series.rolling(window=20).mean()
    ma50 = close_series.rolling(window=50).mean()
    return ma20, ma50


def compute_model_performance(model, data, ticker, lookback_window, evaluation_days=60):
    """
    Run the model against the last `evaluation_days` of actual data and
    return RMSE, MAE, directional accuracy, and a comparison DataFrame.
    """
    try:
        close_series = extract_close(data, ticker)
        close_prices_all = close_series.values.reshape(-1, 1)

        if len(close_prices_all) < lookback_window + evaluation_days:
            return None

        scaled_all = scaler.transform(close_prices_all)

        actuals = []
        predicted = []

        eval_start = len(scaled_all) - evaluation_days

        for i in range(eval_start, len(scaled_all)):
            seq = scaled_all[i - lookback_window : i]
            pred_scaled = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
            pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
            actual_price = close_prices_all[i][0]
            actuals.append(actual_price)
            predicted.append(pred_price)

        actuals = np.array(actuals)
        predicted = np.array(predicted)

        rmse = np.sqrt(np.mean((actuals - predicted) ** 2))
        mae = np.mean(np.abs(actuals - predicted))

        actual_direction = np.sign(np.diff(actuals))
        predicted_direction = np.sign(np.diff(predicted))
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

        eval_dates = close_series.index[eval_start:]
        performance_df = pd.DataFrame({
            "Date": eval_dates,
            "Actual Price": actuals,
            "Predicted Price": predicted,
            "Error ($)": predicted - actuals,
        })

        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
            "df": performance_df,
        }

    except Exception as e:
        st.warning(f"Could not compute model performance: {e}")
        return None


def generate_csv(df):
    """Convert a DataFrame to a UTF-8 CSV bytes object for download."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(ttl=1800)
def fetch_google_news(max_items=5):
    """
    Pull the latest Google-related headlines from Google News RSS.
    TTL of 1800 seconds means headlines refresh every 30 minutes.
    """
    try:
        url = "https://news.google.com/rss/search?q=Google+Alphabet+stock&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            raw = response.read()

        root = ET.fromstring(raw)
        channel = root.find("channel")
        items = channel.findall("item")

        news = []
        for item in items[:max_items]:
            title = item.findtext("title", default="No title")
            link = item.findtext("link", default="#")
            pub_date = item.findtext("pubDate", default="")
            source_el = item.find("source")
            source = source_el.text if source_el is not None else "Google News"

            # Google News RSS appends " - Source" at the end of titles
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]

            try:
                dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                pub_formatted = dt.strftime("%b %d, %Y %H:%M UTC")
            except Exception:
                pub_formatted = pub_date

            news.append({
                "title": title,
                "link": link,
                "source": source,
                "published": pub_formatted,
            })

        return news

    except Exception:
        return []


def compute_confidence_band(predictions, close_prices, window=30):
    """
    Compute upper and lower confidence bounds around the prediction line.
    Uses 1-sigma of recent daily return volatility, compounding over time.
    """
    recent_prices = close_prices[-window:]
    daily_returns = np.diff(recent_prices) / recent_prices[:-1]
    volatility = np.std(daily_returns)

    upper = []
    lower = []
    for i, price in enumerate(predictions):
        band = price * volatility * np.sqrt(i + 1)
        upper.append(price + band)
        lower.append(price - band)

    return np.array(upper), np.array(lower)


# 8.  MAIN DASHBOARD

data = get_stock_data(selected_ticker)

if len(data) < lookback_days:
    st.error(
        f"Not enough data available. Need at least {lookback_days} days, but only have {len(data)} days."
    )
    st.stop()

close_series = extract_close(data, selected_ticker)
close_prices = close_series.values
volume_series = extract_volume(data, selected_ticker)
ma20, ma50 = compute_moving_averages(close_series)
current_price = float(close_prices[-1])

# 8a. GOOG FULL PREDICTION MODE
if selected_ticker == "GOOG":

    x_data, _ = prepare_data(data, selected_ticker, lookback_days)
    predictions = make_predictions(model, x_data[-1], prediction_days)
    upper_band, lower_band = compute_confidence_band(predictions, close_prices)

    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

    next_price = float(predictions[0])
    change_pct = (next_price - current_price) / current_price * 100
    delta_color = "var(--accent-green)" if change_pct >= 0 else "var(--accent-red)"

    # Metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Current Price", f"${current_price:.2f}")
    with col2:
        metric_card("Next-Day Prediction", f"${next_price:.2f}")
    with col3:
        metric_card(
            "Predicted Change",
            f"{change_pct:+.2f}%",
            delta=f"<span style='color:{delta_color};'>{change_pct:+.2f}%</span>",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Price chart with predictions
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
        mode="lines+markers", name="Predicted",
        line=dict(color="#ff7b72", width=2, dash="dot"),
        marker=dict(size=6),
    ))

    # Confidence band — upper bound (invisible line, fills down to lower)
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper_band,
        mode="lines",
        name="Upper Band",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Lower bound with fill to upper — creates the shaded region
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower_band,
        mode="lines",
        name="Confidence Band (1 sigma)",
        fill="tonexty",
        fillcolor="rgba(255,123,114,0.15)",
        line=dict(width=0),
        hoverinfo="skip",
    ))

    fig.add_shape(
        type="line",
        x0=last_date, y0=0, x1=last_date, y1=1,
        yref="paper",
        line=dict(color="#39d353", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=last_date, y=0.95, yref="paper",
        text="Prediction Start", showarrow=False,
        font=dict(color="#39d353"),
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        <p style="color:rgba(255,255,255,0.55);font-size:0.82rem;margin-top:-0.5rem;padding-bottom:0.5rem;">
        The shaded region represents a 1-sigma confidence band based on 30-day price volatility.
        It widens over time because uncertainty compounds with each additional day of prediction.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Volume chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data.index, y=volume_series,
        name="Volume",
        marker_color="rgba(88,166,255,0.6)",
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Volume",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Details")
    pred_details = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions,
        "Day": range(1, prediction_days + 1),
    })
    pred_details["Change %"] = pred_details["Predicted Price"].pct_change() * 100
    pred_details.at[0, "Change %"] = (
        pred_details.at[0, "Predicted Price"] - current_price
    ) / current_price * 100

    st.dataframe(
        pred_details.style.format(
            {"Predicted Price": "${:.2f}", "Change %": "{:+.2f}%"}
        ).set_properties(**{
            "background-color": "transparent",
            "border-color": "rgba(255,255,255,0.1)",
        }),
        hide_index=True,
        use_container_width=True,
        height=(pred_details.shape[0] + 1) * 35 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Model Performance Section ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance (Last 60 Days)")
    st.markdown(
        """
        <p style="color:rgba(255,255,255,0.65);font-size:0.88rem;margin-bottom:1rem;">
        These metrics show how the LSTM model performed against actual GOOG prices
        over the last 60 trading days. Lower RMSE and MAE indicate better accuracy.
        Directional accuracy measures how often the model correctly predicted
        whether the price would rise or fall.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Evaluating model performance..."):
        perf = compute_model_performance(model, data, "GOOG", lookback_days, evaluation_days=60)

    if perf:
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            metric_card("RMSE", f"${perf['rmse']:.2f}", delta="Root Mean Squared Error")
        with p_col2:
            metric_card("MAE", f"${perf['mae']:.2f}", delta="Mean Absolute Error")
        with p_col3:
            dir_color = "var(--accent-green)" if perf["directional_accuracy"] >= 50 else "var(--accent-red)"
            metric_card(
                "Directional Accuracy",
                f"{perf['directional_accuracy']:.1f}%",
                delta=f"<span style='color:{dir_color};'>{'Above' if perf['directional_accuracy'] >= 50 else 'Below'} 50% baseline</span>",
            )

        # Actual vs Predicted chart
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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

    # --- Latest News Section ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Latest Google News")
    st.markdown(
        """
        <p style="color:rgba(255,255,255,0.65);font-size:0.88rem;margin-bottom:1.25rem;">
        Recent headlines related to Google and Alphabet. Headlines refresh every 30 minutes.
        Stock prices are directly influenced by news, so use these as context alongside the predictions above.
        </p>
        """,
        unsafe_allow_html=True,
    )

    news_items = fetch_google_news(max_items=5)

    if news_items:
        news_html = """
        <style>
        .news-row {
            display: flex;
            flex-direction: row;
            gap: 1rem;
            overflow-x: auto;
            padding-bottom: 0.85rem;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.2) transparent;
        }
        .news-row::-webkit-scrollbar {
            height: 5px;
        }
        .news-row::-webkit-scrollbar-track {
            background: transparent;
        }
        .news-row::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.25);
            border-radius: 4px;
        }
        .news-card {
            flex: 0 0 calc(20% - 0.8rem);
            min-width: 210px;
            max-width: 260px;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-top: 3px solid rgba(88, 166, 255, 0.85);
            border-radius: 16px;
            padding: 1.15rem 1.1rem 1rem 1.1rem;
            height: 190px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow:
                0 4px 16px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.08);
            text-decoration: none;
            transition:
                background 0.25s ease,
                transform 0.22s ease,
                border-top-color 0.25s ease,
                box-shadow 0.25s ease;
        }
        .news-card:hover {
            background: rgba(255, 255, 255, 0.14);
            border-top-color: rgba(57, 211, 83, 0.9);
            transform: translateY(-4px);
            box-shadow:
                0 8px 28px rgba(0, 0, 0, 0.45),
                inset 0 1px 0 rgba(255,255,255,0.12);
        }
        .news-card-title {
            color: rgba(255, 255, 255, 0.95);
            font-size: 0.865rem;
            font-weight: 600;
            line-height: 1.48;
            margin: 0 0 0.6rem 0;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }
        .news-card-footer {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .news-card-source {
            color: rgba(88, 166, 255, 1);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .news-card-date {
            color: rgba(255, 255, 255, 0.35);
            font-size: 0.68rem;
        }
        </style>
        <div class="news-row">
        """

        for article in news_items:
            news_html += f"""
            <a href="{article['link']}" target="_blank" class="news-card">
                <p class="news-card-title">{article['title']}</p>
                <div class="news-card-footer">
                    <span class="news-card-source">{article['source']}</span>
                    <span class="news-card-date">{article['published']}</span>
                </div>
            </a>
            """

        news_html += "</div>"
        st.markdown(news_html, unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div style="color:rgba(255,255,255,0.5);font-size:0.88rem;padding:0.5rem 0;">
                Unable to load news at this time. Please check your internet connection or try again shortly.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Download Button (GOOG) ---
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.subheader("Download Report")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.65);font-size:0.88rem;'>Download the prediction details as a CSV file.</p>",
        unsafe_allow_html=True,
    )
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_data = generate_csv(pred_details[["Date", "Day", "Predicted Price", "Change %"]])
    st.download_button(
        label="Download GOOG Predictions CSV",
        data=csv_data,
        file_name=f"GOOG_predictions_{today_str}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# 8b. NON-GOOG ANALYSIS ONLY MODE
else:
    company_name = TICKERS[selected_ticker]

    # Info banner
    st.markdown(
        f"""
        <div class="info-banner">
            You are viewing <strong>{selected_ticker} - {company_name}</strong>.
            Price predictions are only available for GOOG, as the LSTM model was trained exclusively on Google stock data.
            Shown below are historical prices, moving averages, and trading volume.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics (no prediction, just current snapshot)
    prev_price = float(close_prices[-2]) if len(close_prices) > 1 else current_price
    day_change_pct = (current_price - prev_price) / prev_price * 100
    delta_color = "var(--accent-green)" if day_change_pct >= 0 else "var(--accent-red)"

    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Current Price", f"${current_price:.2f}")
    with col2:
        metric_card("Previous Close", f"${prev_price:.2f}")
    with col3:
        metric_card(
            "Day Change",
            f"{day_change_pct:+.2f}%",
            delta=f"<span style='color:{delta_color};'>{day_change_pct:+.2f}%</span>",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Price chart with moving averages
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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

    # Volume chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"{selected_ticker} Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=data.index, y=volume_series,
        name="Volume",
        marker_color="rgba(88,166,255,0.6)",
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date", yaxis_title="Volume",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Moving average summary table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Moving Average Summary")
    ma_summary = pd.DataFrame({
        "Date": data.index[-10:],
        "Close Price": close_prices[-10:],
        "20-Day MA": ma20.values[-10:],
        "50-Day MA": ma50.values[-10:],
    })
    st.dataframe(
        ma_summary.style.format({
            "Close Price": "${:.2f}",
            "20-Day MA": "${:.2f}",
            "50-Day MA": "${:.2f}",
        }).set_properties(**{
            "background-color": "transparent",
            "border-color": "rgba(255,255,255,0.1)",
        }),
        hide_index=True,
        use_container_width=True,
        height=(ma_summary.shape[0] + 1) * 35 + 3,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Download Button (non-GOOG) ---
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.subheader("Download Report")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.65);font-size:0.88rem;'>Download the moving average summary as a CSV file.</p>",
        unsafe_allow_html=True,
    )
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_data = generate_csv(ma_summary)
    st.download_button(
        label=f"Download {selected_ticker} Summary CSV",
        data=csv_data,
        file_name=f"{selected_ticker}_summary_{today_str}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# 9.  DISCLAIMER
st.markdown(
    """
<div class="card" style="text-align:center;margin-top:2rem;">
  <p style="color:var(--text-muted);margin:0;font-size:clamp(0.75rem, 1.5vw, 0.875rem);">
    Disclaimer: Stock price predictions are based on historical data and trained models.
    Past performance is not indicative of future results.
    This tool is for informational purposes only and should not be considered as financial advice.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
