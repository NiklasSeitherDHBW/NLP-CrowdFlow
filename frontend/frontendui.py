import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Mockup-Daten erstellen
def generate_mock_data():
    dates = pd.date_range(start="2022-01-01", end=datetime.today(), freq="D")
    data = {
        "Date": dates,
        "Price": np.random.uniform(20000, 60000, len(dates)),
        "Volume": np.random.uniform(1000, 10000, len(dates)),
        "Change": np.random.choice(["Positive", "Negative", "Neutral"], len(dates)),
    }
    return pd.DataFrame(data)

# Mock-Daten laden
df = generate_mock_data()

# Streamlit App
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .block-container {padding: 1rem;}
    .css-1d391kg {padding: 1rem;}
    .box {
        border: 1px solid #d3d3d3;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .scroll-box {
        height: 400px;
        overflow-y: scroll;
        border: 1px solid #d3d3d3;
        padding: 1rem;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4C8 Kursanalyse Dashboard")

# Oben: Dropdowns und Datumsbereichsauswahl
col1, col2 = st.columns([1, 1])

with col1:
    selected_crypto = st.selectbox("Dropdown - Kryptowährung", ["Bitcoin", "Ethereum", "Ripple"])

with col2:
    date_range = st.date_input(
        "Auswahl Zeitraum",
        [datetime.today() - timedelta(days=30), datetime.today()],
        min_value=datetime(2022, 1, 1),
        max_value=datetime.today(),
    )

# Daten filtern
filtered_data = df[(df["Date"] >= pd.Timestamp(date_range[0])) & (df["Date"] <= pd.Timestamp(date_range[1]))]

# Mittig: Kerzendiagramm
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=filtered_data["Date"],
    open=filtered_data["Price"] - np.random.uniform(0, 1000, len(filtered_data)),
    high=filtered_data["Price"] + np.random.uniform(0, 1000, len(filtered_data)),
    low=filtered_data["Price"] - np.random.uniform(0, 2000, len(filtered_data)),
    close=filtered_data["Price"],
)])
fig_candlestick.update_layout(
    title="Kursverlauf (Kerzendiagramm)",
    xaxis_title="Datum",
    yaxis_title="Preis",
    height=300
)

# Unten: Balkendiagramm
change_colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
filtered_data["Color"] = filtered_data["Change"].map(change_colors)
fig_bar = go.Figure(data=[go.Bar(
    x=filtered_data["Date"],
    y=filtered_data["Volume"],
    marker_color=filtered_data["Color"],
)])
fig_bar.update_layout(
    title="Volumen (Positiv, Negativ, Neutral)",
    xaxis_title="Datum",
    yaxis_title="Volumen",
    height=300
)

# Layout mit optischen Kästen und Scrollbar
col3, col4 = st.columns([3, 1])

with col3:
    #st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.plotly_chart(fig_candlestick, use_container_width=True)
    #st.markdown("</div>", unsafe_allow_html=True)

    #st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    #st.markdown("</div>", unsafe_allow_html=True)

with col4:
    #st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
    st.write("### Informationen")
    for _ in range(30):
        st.text("Beispieltext ...")
    #st.markdown("</div>", unsafe_allow_html=True)
