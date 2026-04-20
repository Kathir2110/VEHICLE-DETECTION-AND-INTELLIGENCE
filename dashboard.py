import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")

st.title("🚦 Smart Traffic Intelligence Dashboard")

# Load CSV
try:
    df = pd.read_csv("traffic_log.csv")
except:
    st.error("traffic_log.csv not found. Run main.py first.")
    st.stop()

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour

# ---------------- TOTAL VEHICLES ----------------
total_vehicles = len(df)

st.metric("🚗 Total Vehicles Detected", total_vehicles)

# ---------------- TYPE-WISE COUNT ----------------
st.subheader("Vehicle Type Distribution")

type_counts = df["Vehicle_Type"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.write(type_counts)

with col2:
    fig1, ax1 = plt.subplots()
    type_counts.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("Vehicle Type Distribution")
    st.pyplot(fig1)

# ---------------- LANE-WISE COUNT ----------------
st.subheader("Lane-wise Vehicle Count")

lane_counts = df["Lane"].value_counts()

col3, col4 = st.columns(2)

with col3:
    st.write(lane_counts)

with col4:
    fig2, ax2 = plt.subplots()
    lane_counts.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Count")
    ax2.set_title("Lane Distribution")
    st.pyplot(fig2)

# ---------------- HOURLY ANALYSIS ----------------
st.subheader("Peak Hour Analysis")

hourly_count = df.groupby("Hour").size()

peak_hour = hourly_count.idxmax()
peak_count = hourly_count.max()

st.success(f"🔥 Peak Traffic Hour: {peak_hour}:00 with {peak_count} vehicles")

fig3, ax3 = plt.subplots()
hourly_count.plot(kind="line", marker="o", ax=ax3)
ax3.set_ylabel("Vehicles")
ax3.set_title("Hourly Traffic Trend")
st.pyplot(fig3)