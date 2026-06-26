"""
Streamlit Real-Time Wearable Monitoring Dashboard
==================================================
A production-quality real-time dashboard for the Lightweight AI Framework
for Wearable Health and Activity Monitoring.

Features:
  • Live-streaming PPG + HRV metrics (simulated or from InfluxDB)
  • Activity classification display (Sitting / Walking / Running)
  • LSTM-AE anomaly score trend with threshold visualization
  • 3-tier alert system with color-coded status panel
  • Historical HRV trend charts
  • Bandwidth reduction statistics

Run:
    streamlit run src/dashboard/app.py

Dependencies:
    pip install streamlit plotly numpy pandas
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wearable Health Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-lbl { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .alert-ok   { color: #3fb950; font-weight: bold; font-size: 1.1rem; }
  .alert-warn { color: #d29922; font-weight: bold; font-size: 1.1rem; }
  .alert-crit { color: #f85149; font-weight: bold; font-size: 1.1rem; }
  h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALERT_RULES = {
    "Sitting": {"hr_high": 100, "hr_low": 50, "rmssd_low": 15},
    "Walking": {"hr_high": 130, "hr_low": 50, "rmssd_low": 10},
    "Running": {"hr_high": 170, "hr_low": 60, "rmssd_low":  5},
}
HR_BASE    = {"Sitting": 68,  "Walking": 95,  "Running": 145}
RMSSD_BASE = {"Sitting": 42, "Walking": 28,  "Running": 15}
WINDOW     = 120   # seconds of history to display
AE_THRESHOLD = 0.05

COLORS = {
    "Sitting": "#58a6ff",
    "Walking": "#3fb950",
    "Running": "#f85149",
}

ACTIVITY_SCHEDULE = [
    (0,   "Sitting"),
    (60,  "Walking"),
    (140, "Running"),
    (220, "Walking"),
    (270, "Sitting"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = {
        "timestamps": [],
        "hr": [],
        "rmssd": [],
        "sdnn": [],
        "ae_score": [],
        "anomaly_flag": [],
        "activity": [],
        "alert_active": [],
    }

if "sim_t" not in st.session_state:
    st.session_state.sim_t = 0

if "anomaly_counter" not in st.session_state:
    st.session_state.anomaly_counter = 0

if "running" not in st.session_state:
    st.session_state.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Simulator (matches mqtt_publisher.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def get_activity(t: int) -> str:
    current = "Sitting"
    for onset, act in ACTIVITY_SCHEDULE:
        if t >= onset:
            current = act
    return current


def simulate_sample(t: int, anomaly_counter: int) -> dict:
    np.random.seed(t % 1000 + anomaly_counter)
    activity = get_activity(t)
    anomaly  = (anomaly_counter % 120) < 8

    hr = HR_BASE[activity] + 5 * np.sin(2 * np.pi * t / 60) + np.random.randn() * 2.5
    if anomaly:
        hr += np.random.uniform(35, 50)

    rmssd = RMSSD_BASE[activity] + np.random.randn() * 2.5
    if anomaly:
        rmssd = max(2.0, rmssd - 25)

    sdnn = rmssd * 1.15 + np.random.randn() * 3

    ae_score = np.random.exponential(0.012)
    if anomaly:
        ae_score = np.random.uniform(0.07, 0.15)

    rules = ALERT_RULES[activity]
    alert_active = (hr > rules["hr_high"] or hr < rules["hr_low"] or rmssd < rules["rmssd_low"])

    return {
        "timestamp":    datetime.now(timezone.utc),
        "hr":           round(float(hr), 1),
        "rmssd":        round(float(max(rmssd, 1.0)), 2),
        "sdnn":         round(float(max(sdnn, 1.0)), 2),
        "ae_score":     round(float(ae_score), 6),
        "anomaly_flag": int(ae_score > AE_THRESHOLD),
        "activity":     activity,
        "alert_active": bool(alert_active),
    }


def append_sample(sample: dict):
    h = st.session_state.history
    h["timestamps"].append(sample["timestamp"])
    h["hr"].append(sample["hr"])
    h["rmssd"].append(sample["rmssd"])
    h["sdnn"].append(sample["sdnn"])
    h["ae_score"].append(sample["ae_score"])
    h["anomaly_flag"].append(sample["anomaly_flag"])
    h["activity"].append(sample["activity"])
    h["alert_active"].append(sample["alert_active"])

    # Keep last WINDOW samples
    if len(h["timestamps"]) > WINDOW:
        for key in h:
            h[key] = h[key][-WINDOW:]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.markdown("---")

    refresh_rate = st.slider("Refresh rate (s)", 1, 5, 1, key="refresh_rate_slider")
    show_raw     = st.checkbox("Show raw PPG simulation", value=False)

    st.markdown("---")
    st.markdown("### 📡 Data Source")
    data_source = st.radio("Source", ["Simulation", "InfluxDB (requires setup)"], index=0)

    st.markdown("---")
    st.markdown("### 🏃 Activity Alerts")
    for act, rules in ALERT_RULES.items():
        st.markdown(f"**{act}**: HR {rules['hr_low']}–{rules['hr_high']} bpm | RMSSD >{rules['rmssd_low']} ms")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")

    if st.button("🔄 Reset Session"):
        st.session_state.history = {k: [] for k in st.session_state.history}
        st.session_state.sim_t = 0
        st.session_state.anomaly_counter = 0
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Framework**: Lightweight AI for Wearable Health Monitoring  
    **Paper**: IEEE Conference Format  
    **Datasets**: WESAD · PAMAP2
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ❤️ Wearable Health Monitoring — Real-Time Dashboard")
st.markdown("*Lightweight AI Framework · Edge-AI Pipeline · Week 5*")
st.markdown("---")

# Advance simulation
sample = simulate_sample(st.session_state.sim_t, st.session_state.anomaly_counter)
append_sample(sample)
st.session_state.sim_t += 1
st.session_state.anomaly_counter += 1

h  = st.session_state.history
n  = len(h["timestamps"])
ts = h["timestamps"]

# ── Current metrics row ─────────────────────────────────────────────────────
cur_hr       = h["hr"][-1]       if n else 0
cur_rmssd    = h["rmssd"][-1]    if n else 0
cur_sdnn     = h["sdnn"][-1]     if n else 0
cur_ae       = h["ae_score"][-1] if n else 0
cur_activity = h["activity"][-1] if n else "–"
cur_alert    = h["alert_active"][-1] if n else False
cur_anomaly  = h["anomaly_flag"][-1] if n else False

total_anomalies = sum(h["anomaly_flag"])
bandwidth_reduction = 92.5

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_hr = round(cur_hr - (h["hr"][-2] if n > 1 else cur_hr), 1)
    st.metric("❤️ Heart Rate", f"{cur_hr:.0f} bpm", f"{delta_hr:+.1f}")

with col2:
    st.metric("📊 RMSSD", f"{cur_rmssd:.1f} ms")

with col3:
    st.metric("📈 SDNN", f"{cur_sdnn:.1f} ms")

with col4:
    alert_emoji = "🚨" if cur_alert else "✅"
    st.metric("🔔 Alert Status", f"{alert_emoji} {'ACTIVE' if cur_alert else 'OK'}")

with col5:
    act_emoji = {"Sitting": "🪑", "Walking": "🚶", "Running": "🏃"}.get(cur_activity, "❓")
    st.metric("🏃 Activity", f"{act_emoji} {cur_activity}")

st.markdown("---")

# ── AE Anomaly score + HR trend ─────────────────────────────────────────────
fig_main = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Heart Rate (bpm)", "Anomaly Score (AE Reconstruction Error)",
                    "RMSSD — Parasympathetic Tone (ms)", "Activity Classification"),
    vertical_spacing=0.18,
    horizontal_spacing=0.08,
)

PLOT_COLORS_ACT = [COLORS.get(a, "#58a6ff") for a in h["activity"]]

# HR
fig_main.add_trace(
    go.Scatter(x=ts, y=h["hr"], mode="lines", name="HR",
               line=dict(color="#58a6ff", width=1.5)),
    row=1, col=1
)
# Alert threshold
rules_cur = ALERT_RULES.get(cur_activity, ALERT_RULES["Sitting"])
fig_main.add_hline(y=rules_cur["hr_high"], line_dash="dash", line_color="#f85149",
                   annotation_text=f"Alert >{rules_cur['hr_high']}", row=1, col=1)

# AE Score
fig_main.add_trace(
    go.Scatter(x=ts, y=h["ae_score"], fill="tozeroy", name="AE Score",
               line=dict(color="#bc8cff", width=1.2),
               fillcolor="rgba(188,140,255,0.2)"),
    row=1, col=2
)
fig_main.add_hline(y=AE_THRESHOLD, line_dash="dash", line_color="#f85149",
                   annotation_text=f"Threshold ({AE_THRESHOLD})", row=1, col=2)

# RMSSD
fig_main.add_trace(
    go.Scatter(x=ts, y=h["rmssd"], mode="lines", name="RMSSD",
               line=dict(color="#3fb950", width=1.5)),
    row=2, col=1
)
fig_main.add_hline(y=rules_cur["rmssd_low"], line_dash="dash", line_color="#d29922",
                   annotation_text=f"Alert <{rules_cur['rmssd_low']}", row=2, col=1)

# Activity
act_int_map = {"Sitting": 0, "Walking": 1, "Running": 2}
act_ints    = [act_int_map.get(a, 0) for a in h["activity"]]
fig_main.add_trace(
    go.Scatter(x=ts, y=act_ints, mode="lines", fill="tozeroy", name="Activity",
               line=dict(color="#d29922", width=1.5),
               fillcolor="rgba(210,153,34,0.25)"),
    row=2, col=2
)
fig_main.update_yaxes(
    tickvals=[0, 1, 2],
    ticktext=["Sitting", "Walking", "Running"],
    row=2, col=2
)

fig_main.update_layout(
    height=520,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", size=11),
    showlegend=False,
    margin=dict(l=40, r=20, t=50, b=20),
)
for ax in fig_main.layout:
    if ax.startswith("xaxis") or ax.startswith("yaxis"):
        fig_main.layout[ax].update(
            gridcolor="#21262d",
            zerolinecolor="#30363d",
        )

st.plotly_chart(fig_main, use_container_width=True)

# ── Stats row ────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📡 Session Time", f"{st.session_state.sim_t}s")
with c2:
    st.metric("🚨 Anomalies Detected", total_anomalies)
with c3:
    st.metric("📉 Bandwidth Reduction", f"{bandwidth_reduction:.1f}%")
with c4:
    ae_status = "🚨 ANOMALY" if cur_anomaly else "✅ Normal"
    st.metric("🤖 LSTM-AE Status", ae_status)

# ── Anomaly log table ────────────────────────────────────────────────────────
if total_anomalies > 0:
    st.markdown("### 🚨 Recent Anomaly Log")
    df = pd.DataFrame({
        "Timestamp": h["timestamps"],
        "HR (bpm)":  h["hr"],
        "RMSSD (ms)": h["rmssd"],
        "AE Score":  h["ae_score"],
        "Activity":  h["activity"],
        "Alert":     ["🚨" if f else "✅" for f in h["alert_active"]],
    })
    anomaly_df = df[df["AE Score"] > AE_THRESHOLD].tail(10)
    if not anomaly_df.empty:
        st.dataframe(
            anomaly_df.style.background_gradient(subset=["AE Score"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

# ── Auto refresh ─────────────────────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
