import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# 1.  GLOBAL STYLING
st.set_page_config(page_title="Google Stock Predictor", layout="wide")

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

/* Make cards slightly transparent to show background */
.card {
    background-color: rgba(22, 27, 34, 0.9);
}

/* Make sidebar slightly transparent */
.css-1d391kg {
    background-color: rgba(22, 27, 34, 0.9) !important;
}
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

/* --- Table header --- */
.stDataFrame thead th {
    background-color: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 600;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* --- Table cells --- */
.stDataFrame tbody td {
    background-color: transparent !important;
    color: white !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* --- Hover effects --- */
.stDataFrame tbody tr:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
}

/* --- Scrollbar styling --- */
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
/* --- Metric Cards --- */
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
</style>
""",
    unsafe_allow_html=True,
)

# 3.  LOAD ML ARTEFACTS
@st.cache_resource
def load_ml_components():
    try:
        model = load_model("google_stock_price_prediction_model.keras", compile=False)
        model.save("google_stock_price_prediction_model.keras", save_format="keras")
        scaler = joblib.load("stock_price_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_ml_components()

# Detect how many features the scaler expects
N_FEATURES = scaler.n_features_in_

# 4.  HEADER
st.markdown(
    """
<div class="card" style="text-align:center;">
  <h1>Google Stock Price Predictor</h1>
  <p style="color:var(--text-muted);margin:0;">
     This application uses a trained LSTM neural network to predict Google (GOOG) stock prices based on historical data.
    The model was trained on 20 years of daily stock data from Yahoo Finance.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# 5.  SIDEBAR
st.sidebar.header("Settings")
lookback_days = st.sidebar.slider(
    "Lookback Window (days)", min_value=50, max_value=200, value=100
)
prediction_days = st.sidebar.slider(
    "Prediction Horizon (days)", min_value=1, max_value=30, value=7
)

# 6.  DATA PIPELINE
@st.cache_data
def get_stock_data():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        data = yf.download("GOOG", start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error("No data retrieved from Yahoo Finance. Please try again later.")
            st.stop()

        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        st.stop()


def build_features(data, n_features):
    """
    Build a feature matrix that matches the number of features the scaler
    was trained on. Features are added in the same order used in the notebook:
      1. Close
      2. MA_50
      3. MA_100
      4. RSI_14
      5. BB_width (BB_upper - BB_lower, window=20)
      6. Volume_ratio (Volume / Volume_MA5)
    If n_features > 6, additional features (MA_200, Daily_Return,
    Volatility_30d) are appended to reach the required count.
    """
    # Handle MultiIndex columns produced by newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", "GOOG")]
        volume = data[("Volume", "GOOG")]
    else:
        close = data["Close"]
        volume = data["Volume"]

    df = pd.DataFrame(index=data.index)
    df["Close"] = close
    df["MA_50"] = close.rolling(window=50).mean()
    df["MA_100"] = close.rolling(window=100).mean()
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    bb = BollingerBands(close, window=20)
    df["BB_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    volume_ma5 = volume.rolling(5).mean()
    df["Volume_ratio"] = volume / volume_ma5

    # If the scaler was trained on more than 6 features, append extras
    if n_features >= 7:
        df["MA_200"] = close.rolling(window=200).mean()
    if n_features >= 8:
        df["Daily_Return"] = close.pct_change()
    if n_features >= 9:
        df["Volatility_30d"] = df["Daily_Return"].rolling(window=30).std()

    # Keep only as many columns as the scaler expects
    feature_cols = list(df.columns)[:n_features]
    df = df[feature_cols]

    # Drop rows with any NaN (caused by rolling windows)
    df = df.dropna()

    return df


def prepare_data(data, lookback_window, n_features):
    try:
        feature_df = build_features(data, n_features)

        if len(feature_df) < lookback_window:
            raise ValueError(
                f"Not enough data after feature engineering. "
                f"Need at least {lookback_window} rows, got {len(feature_df)}."
            )

        feature_array = feature_df.values  # shape: (T, n_features)

        # Validate before transforming
        if np.isnan(feature_array).any() or np.isinf(feature_array).any():
            raise ValueError("Feature matrix contains NaN or infinite values.")

        scaled = scaler.transform(feature_array)  # shape: (T, n_features)

        # Build sequences
        x_data = []
        for i in range(lookback_window, len(scaled)):
            x_data.append(scaled[i - lookback_window : i])  # shape: (lookback, n_features)

        x_array = np.array(x_data)  # shape: (N, lookback, n_features)

        # Close prices aligned to the same rows as x_data targets
        close_prices = feature_df["Close"].values

        return x_array, close_prices

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        st.stop()


def make_predictions(model, last_sequence, days_to_predict, n_features):
    """
    Iteratively predict future close prices.
    last_sequence shape: (lookback, n_features)
    For each step, predict the next close (column 0), then shift the window
    forward by updating column 0 of the new timestep with the predicted value
    and padding the remaining feature columns with the last known values.
    Finally, inverse-transform using only column 0 of the scaler.
    """
    try:
        predictions_scaled = []
        current_seq = last_sequence.copy()  # (lookback, n_features)

        for _ in range(days_to_predict):
            # Predict expects shape (1, lookback, n_features)
            next_pred_scaled = model.predict(
                current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]),
                verbose=0,
            )[0][0]

            predictions_scaled.append(next_pred_scaled)

            # Build new row: use last known values for auxiliary features,
            # update close (column 0) with the new prediction
            new_row = current_seq[-1].copy()
            new_row[0] = next_pred_scaled

            # Shift window forward
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1] = new_row

        # Inverse-transform only the close-price column
        # Build a dummy array with n_features columns, zeros everywhere except col 0
        dummy = np.zeros((len(predictions_scaled), n_features))
        dummy[:, 0] = predictions_scaled
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()


# 7.  MAIN DASHBOARD
data = get_stock_data()

if len(data) < lookback_days + 200:  # 200 for MA_200 rolling window
    st.error(
        f"Not enough data available. Need at least {lookback_days + 200} days of history."
    )
    st.stop()

x_data, close_prices = prepare_data(data, lookback_days, N_FEATURES)
predictions = make_predictions(model, x_data[-1], prediction_days, N_FEATURES)

last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

# --- Metrics row ---
st.markdown('<div class="metric-container">', unsafe_allow_html=True)
current_price = float(close_prices[-1])
next_price = float(predictions[0])
change_pct = (next_price - current_price) / current_price * 100
delta_color = "var(--accent-green)" if change_pct >= 0 else "var(--accent-red)"

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

# --- Chart ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Price Chart")

history_df = pd.DataFrame(
    {"Date": data.index[-len(close_prices):], "Close": close_prices, "Type": "Historical"}
)
prediction_df = pd.DataFrame(
    {"Date": future_dates, "Close": predictions, "Type": "Predicted"}
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_df["Date"],
        y=history_df["Close"],
        mode="lines",
        name="Historical",
        line=dict(color="#58a6ff", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=prediction_df["Date"],
        y=prediction_df["Close"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#ff7b72", width=2, dash="dot"),
        marker=dict(size=6),
    )
)

fig.add_shape(
    type="line",
    x0=last_date,
    y0=0,
    x1=last_date,
    y1=1,
    yref="paper",
    line=dict(color="#39d353", width=2, dash="dash"),
)
fig.add_annotation(
    x=last_date,
    y=0.95,
    yref="paper",
    text="Prediction Start",
    showarrow=False,
    font=dict(color="#39d353"),
)

fig.update_layout(
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    hoverlabel=dict(
        bgcolor="rgba(30,30,30,0.8)",
        font_size=14,
        font_family="Arial",
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
    ),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction table ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Prediction Details")
pred_details = pd.DataFrame(
    {
        "Date": future_dates,
        "Predicted Price": predictions,
        "Day": range(1, prediction_days + 1),
    }
)
pred_details["Change %"] = pred_details["Predicted Price"].pct_change() * 100
pred_details.at[0, "Change %"] = (
    pred_details.at[0, "Predicted Price"] - current_price
) / current_price * 100

st.dataframe(
    pred_details.style
        .format({"Predicted Price": "${:.2f}", "Change %": "{:+.2f}%"})
        .applymap(lambda x: "color: white")
        .set_properties(**{
            "background-color": "transparent",
            "border-color": "rgba(255,255,255,0.1)",
        }),
    hide_index=True,
    use_container_width=True,
    height=(pred_details.shape[0] + 1) * 35 + 3,
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Disclaimer ---
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
