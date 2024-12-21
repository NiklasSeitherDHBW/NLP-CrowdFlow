import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Mockup-Daten erstellen


def generate_mock_data():
    dates = pd.date_range(start="2022-01-01", end=datetime.today(), freq="h")
    data = {
        "Date": dates,
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
        /* background-color: #f9f9f9; */
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

symbols = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Ripple": "XRP-USD"
}

with col1:
    selected_crypto = st.selectbox(
        "Kryptowährung auswählen:", list(symbols.keys()))

with col2:
    date_range = st.date_input(
        "Auswahl Zeitraum",
        [datetime.today() - timedelta(days=30), datetime.today()],
        min_value=datetime(2022, 1, 1),
        max_value=datetime.today(),
    )

# Daten filtern
start_date, end_date = date_range

filtered_data = df[(df["Date"] >= pd.Timestamp(start_date))
                   & (df["Date"] <= pd.Timestamp(end_date))]
filtered_data.set_index("Date", inplace=True)

# Aggregation auf wöchentlicher Basis (anpassbar)
print(filtered_data)
diff = end_date - start_date
print(diff.days)

# Group by the appropriate frequency and aggregate the "Change" column
if diff.days <= 1:
    grouped_data = filtered_data.groupby(pd.Grouper(freq='h'))[
        "Change"].value_counts()
if diff.days <= 3:
    grouped_data = filtered_data.groupby(pd.Grouper(freq='2h'))[
        "Change"].value_counts()
elif diff.days <= 7:
    grouped_data = filtered_data.groupby(pd.Grouper(freq='4h'))[
        "Change"].value_counts()
else:
    grouped_data = filtered_data.groupby(pd.Grouper(freq='D'))[
        "Change"].value_counts()

print("Jdaskdjnaskjdkjsakj=============================================================================")
print(grouped_data)
grouped_data["Positive"] = 0
grouped_data["Negative"] = 0
grouped_data["Neutral"] = 0
print(grouped_data)
print("2222222222222222222=============================================================================")
grouped_data = grouped_data.unstack(fill_value=0)

print(grouped_data)
print("Jdaskdjnaskjdkjsakj=============================================================================")

fig_sentiment = go.Figure()
fig_sentiment.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data["Positive"],
    name="Positive",
    marker_color="green"
))
fig_sentiment.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data["Neutral"],
    name="Neutral",
    marker_color="gray"
))
fig_sentiment.add_trace(go.Bar(
    x=grouped_data.index,
    y=grouped_data["Negative"],
    name="Negative",
    marker_color="red"
))
fig_sentiment.update_layout(
    barmode="stack",  # Stacked bar chart
    title="Sentiment Over Time",
    xaxis_title="Date",
    yaxis_title="Proportion"
)

# Layout mit optischen Kästen und Scrollbar
col3, col4 = st.columns([3, 1])

with col3:
    # Horizontale Linie vor Sentiment-Diagramm
    st.markdown(
        """<div class="custom-title">Kursverlauf (Kerzendiagramm)</div>""",
        unsafe_allow_html=True
    )

    # Daten abrufen und Multi-Index bereinigen
    if len(date_range) == 2:
        start_date, end_date = date_range

        data = yf.download(symbols[selected_crypto], interval="60m",
                           start=start_date, end=end_date, multi_level_index=False)
        diff = end_date - start_date

        # data = data.loc[start_date:end_date]
        print(diff.days)

        if diff.days <= 1:
            # Group by hours if the difference is less than a day
            data = data.groupby(pd.Grouper(level=0, freq='h')).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        if diff.days <= 3:
            # Group by hours if the difference is less than a day
            data = data.groupby(pd.Grouper(level=0, freq='2h')).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        elif diff.days < 7:
            # Group by days if the difference is less than a week but more than a day
            data = data.groupby(pd.Grouper(level=0, freq='4h')).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        else:
            # Group by weeks if the difference is more than a week
            data = data.groupby(pd.Grouper(level=0, freq='D')).agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

        # Multi-Index bereinigen
        data = data.reset_index()  # Multi-Index auflösen

        print(data.head(10))

        # Plotly Candlestick-Chart erstellen
        fig = go.Figure(data=[go.Candlestick(
            x=data['Date'] if 'Date' in data.columns else data['Datetime'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])

        fig.update_layout(
            title=f"{selected_crypto} Kursverlauf",
            xaxis_title="Datum",
            yaxis_title="Preis in USD",
            xaxis_rangeslider_visible=False
        )

        figBar = go.Figure(data=[go.Bar(
            x=data['Date'] if 'Date' in data.columns else data['Datetime'],
            y=data['Volume']
        )])
        figBar.update_layout(
            title=f"{selected_crypto} Volume",
            xaxis_title="Datum",
            yaxis_title="Volumen in USD",
            xaxis_rangeslider_visible=False
        )

        # Chart anzeigen
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(figBar, use_container_width=True)
    else:
        st.warning("Bitte wähle ein gültiges Start- und Enddatum aus.")

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
