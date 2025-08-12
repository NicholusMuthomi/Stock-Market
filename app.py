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
    model_path = os.path.join(os.path.dirname(__file__), "google_stock_price_prediction_model.h5")
    model = load_model(model_path)
    scaler = joblib.load("stock_price_scaler.pkl")
    return model, scaler

model, scaler = load_ml_components()

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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    return yf.download("GOOG", start=start_date, end=end_date)

def prepare_data(data, lookback_window):
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    x_data = []
    for i in range(lookback_window, len(scaled_data)):
        x_data.append(scaled_data[i - lookback_window : i])
    return np.array(x_data), close_prices.flatten()

def make_predictions(model, last_sequence, days_to_predict):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days_to_predict):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))[0][0]
        predictions.append(next_pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# 7.  MAIN DASHBOARD
data = get_stock_data()
x_data, close_prices = prepare_data(data, lookback_days)
predictions = make_predictions(model, x_data[-1], prediction_days)

last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

# --- Metrics row ---
st.markdown('<div class="metric-container">', unsafe_allow_html=True)
current_price = float(data["Close"].iloc[-1])
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

# --- Chart
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Price Chart")

history_df = pd.DataFrame(
    {"Date": data.index, "Close": close_prices, "Type": "Historical"}
)
prediction_df = pd.DataFrame(
    {"Date": future_dates, "Close": predictions, "Type": "Predicted"}
)
combined_df = pd.concat([history_df, prediction_df])

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
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    hoverlabel=dict(
        bgcolor='rgba(30,30,30,0.8)',
        font_size=14,
        font_family="Arial"
    ),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.1)',
        linecolor='rgba(255,255,255,0.2)'
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.1)',
        linecolor='rgba(255,255,255,0.2)'
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='rgba(255,255,255,0.2)'
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)  # close the card AFTER the chart

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
            'background-color': 'transparent',
            'border-color': 'rgba(255,255,255,0.1)'
        }),
    hide_index=True,
    use_container_width=True,
    height=(pred_details.shape[0] + 1) * 35 + 3
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
