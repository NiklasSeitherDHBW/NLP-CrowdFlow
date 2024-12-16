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
    .custom-box {
        border: 2px solid #4CAF50; /* Rahmenfarbe */
        padding: 15px; /* Abstand innerhalb des Rahmens */
        border-radius: 10px; /* Abgerundete Ecken */
        background-color: #f9f9f9; /* Hintergrundfarbe */
        margin-bottom: 20px; /* Abstand nach unten */
    }
    .scroll-box {
        border: 1px solid #cccccc;
        padding: 10px;
        width: 100%; /* Kastenbreite */
        height: 700px; /* Angepasste Höhe, bündig mit Achsenbeschriftung */
        overflow-y: scroll; /* Eigenes Scrollen aktivieren */ 
        background-color: #f9f9f9;
    }
    .custom-title {
        font-size: 1.5rem; /* Größere Schriftgröße für Überschriften */
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
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

# Aggregation auf wöchentlicher Basis (anpassbar)
filtered_data["Week"] = filtered_data["Date"].dt.to_period("W").dt.start_time
weekly_sentiment = filtered_data.groupby("Week")["Change"].value_counts(normalize=True).unstack(fill_value=0)

# Gestapeltes Balkendiagramm für Sentiment-Verlauf
fig_sentiment = go.Figure()
fig_sentiment.add_trace(go.Bar(
    x=weekly_sentiment.index,
    y=weekly_sentiment["Positive"],
    name="Positive",
    marker_color="green"
))
fig_sentiment.add_trace(go.Bar(
    x=weekly_sentiment.index,
    y=weekly_sentiment["Neutral"],
    name="Neutral",
    marker_color="gray"
))
fig_sentiment.add_trace(go.Bar(
    x=weekly_sentiment.index,
    y=weekly_sentiment["Negative"],
    name="Negative",
    marker_color="red"
))
fig_sentiment.update_layout(
    barmode="stack",  # Gestapeltes Balkendiagramm
    title="Sentiment-Verlauf",
    xaxis_title="Zeitraum",
    yaxis_title="Anteil",
    height=320,
    legend_title="Sentiment"
)

# Mittig: Kerzendiagramm
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=filtered_data["Date"],
    open=filtered_data["Price"] - np.random.uniform(0, 1000, len(filtered_data)),
    high=filtered_data["Price"] + np.random.uniform(0, 1000, len(filtered_data)),
    low=filtered_data["Price"] - np.random.uniform(0, 2000, len(filtered_data)),
    close=filtered_data["Price"],
)])
fig_candlestick.update_layout(
    title="Kursverlauf", 
    xaxis_title="Datum",
    yaxis_title="Preis",
    height=280
)

# Layout mit optischen Kästen und Scrollbar
col3, col4 = st.columns([3, 1])

with col3:
    # Horizontale Linie vor Sentiment-Diagramm
    st.markdown("""
            <div class="custom-title">Kursverlauf (Kerzendiagramm)</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_candlestick, use_container_width=True)
    st.markdown("""
        <hr class="horizontal-line">
    """, unsafe_allow_html=True)  # Horizontale Linie

    # Sentiment-Diagramm
    st.markdown("""
        <div class="custom-title">Sentiment-Verlauf</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col4:
    # Überschrift oberhalb der Scroll-Box
    st.markdown("""
        <div class="custom-title">Informationen</div>
    """, unsafe_allow_html=True)
    # Scroll-Kasten hinzufügen
    st.markdown("""
        <div class="scroll-box">
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan, metus ultrices eleifend gravida, nulla nunc varius lectus, nec rutrum justo nibh eu lectus. Ut vulputate semper dui. Fusce erat odio, sollicitudin vel erat vel, interdum mattis neque. Subheadings can go here.</p>
            <p>More content: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis vehicula ex sed justo tempor, quis facilisis libero viverra. Integer vitae libero ac risus egestas placerat eu ac felis. Donec fermentum id sapien at congue.</p>
            <p>Even more content: Proin faucibus arcu quis ante. Cras justo odio, dapibus ac facilisis in, egestas eget quam. Nullam quis risus eget urna mollis ornare vel eu leo.</p>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan, metus ultrices eleifend gravida, nulla nunc varius lectus, nec rutrum justo nibh eu lectus. Ut vulputate semper dui. Fusce erat odio, sollicitudin vel erat vel, interdum mattis neque. Subheadings can go here.</p>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan, metus ultrices eleifend gravida, nulla nunc varius lectus, nec rutrum justo nibh eu lectus. Ut vulputate semper dui. Fusce erat odio, sollicitudin vel erat vel, interdum mattis neque. Subheadings can go here.</p>
            <p>More content: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis vehicula ex sed justo tempor, quis facilisis libero viverra. Integer vitae libero ac risus egestas placerat eu ac felis. Donec fermentum id sapien at congue.</p>
            <p>Even more content: Proin faucibus arcu quis ante. Cras justo odio, dapibus ac facilisis in, egestas eget quam. Nullam quis risus eget urna mollis ornare vel eu leo.</p>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan, metus ultrices eleifend gravida, nulla nunc varius lectus, nec rutrum justo nibh eu lectus. Ut vulputate semper dui. Fusce erat odio, sollicitudin vel erat vel, interdum mattis neque. Subheadings can go here.</p>
        </div>
    """, unsafe_allow_html=True)
