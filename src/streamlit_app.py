from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import os
import urllib.request
import nltk

CRYPTO_SYMBOLS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Ripple": "XRP-USD"}
SENTIMENTS = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {"positive": "green", "neutral": "gray", "negative": "red"}


def generate_mock_news(model):
    df = pd.read_csv("res/input/cryptonews.csv")
    
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    
    max_date = df["date"].max()
    today = pd.Timestamp.now()
    time_difference = (today - max_date).days
    offset_days = time_difference
    
    df["date"] = df["date"] + timedelta(days=offset_days)
    
    df = df.drop(columns=["sentiment"])
    
    X_pred = df.drop(columns=["date", "source", "url"])
    
    df["sentiment"] = model.pipeline_balanced.predict(X_pred)
    
    return df


def display_news(news_df):
    for _, row in news_df.iterrows():
        sentiment_color = SENTIMENT_COLORS[row["sentiment"]]

        st.markdown(
            f"""
            <div>
                <h3 style="margin: 0;">{row['title']}</h3>
                <p style="margin: 5px 0;">{row['text'][:100]}...</p>
                <a href="{row['url']}" style="color: #1a73e8; text-decoration: none;">Read more...</a>
            </div>
            <div style="position: absolute; top: 15px; right: 15px; background-color: {sentiment_color}; color: white; 
                        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                {row['sentiment']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")


def get_frequency(start_date, end_date):
    """Determine the frequency for grouping data based on the date range."""
    diff_days = (end_date - start_date).days
    if diff_days <= 1:
        return "h"
    elif diff_days <= 3:
        return "2h"
    elif diff_days <= 7:
        return "4h"
    else:
        return "D"


def group_sentiment_data(df, start_date, end_date):
    """Group data based on the selected date range."""
    filtered_data = df[
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    ]
    filtered_data.set_index("date", inplace=True)
    freq = get_frequency(start_date, end_date)
    grouped = (
        filtered_data.groupby(pd.Grouper(freq=freq))["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )
    return grouped


@st.cache_data
def download_crypto_data(symbol, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance."""
    data = yf.download(
        symbol, interval="60m", start=start_date, end=end_date, multi_level_index=False
    )

    freq = get_frequency(start_date, end_date)
    data = data.groupby(pd.Grouper(level=0, freq=freq)).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )

    return data


def plot_candlestick_with_separate_volume(data):
    """Create a candlestick chart with a separate subplot for volume."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color="lightblue",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title = "Candlestick chart with Volume",
        xaxis=dict(
            rangeslider=dict(visible=False)  # Disable the range slider
        ),
        yaxis=dict(title="Price (USD)", side="left"),
        yaxis2=dict(title="Volume", side="left"),
        height=650,
        showlegend=False,
    )

    return fig


def plot_sentiment(data):
    """Create a stacked bar chart for sentiment analysis."""
    fig = go.Figure()

    for sentiment in SENTIMENTS:
        y_values = data[sentiment] if sentiment in data else [0] * len(data)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=y_values,
                name=sentiment,
                marker_color=SENTIMENT_COLORS[sentiment],
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Counts",
        legend={"orientation": "h"},
        height=400,
    )

    return fig


def apply_custom_styles():
    """Apply custom CSS styles."""
    st.markdown(
        """
        <style>
        .block-container {padding: 1rem;}
        .custom-title {font-size: 1.5rem; font-weight: bold; text-align: center; margin-bottom: 10px;}
        .scroll-box {border: 1px solid #ccc; padding: 10px; height: 700px; overflow-y: auto;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    model_path = "res/models/nltk_rf.joblib"
    model_url = "https://drive.usercontent.google.com/download?id=1bHl_4G_0W63Q3kj0h1tUijnvtY7BITcn&export=download&authuser=0&confirm=t&uuid=3f546199-99ec-4ff7-9142-0d66ec3084e2&at=APvzH3oWTi5WpBYW23RsVWQNEg27%3A1734888877135"

    if not os.path.exists(model_path):
        st.warning("Downloading the model file because it couldn't be found locally.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
    
    return joblib.load(model_path)


def main():
    st.set_page_config(layout="wide")
    apply_custom_styles()
    st.title("\U0001f4c8 Cryptocurrency Analysis Dashboard")

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")

    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox(
            "Select Cryptocurrency:", list(CRYPTO_SYMBOLS.keys())
        )
    with col2:
        date_range = st.date_input(
            "Select Date Range:",
            [datetime.today() - timedelta(days=30), datetime.today()],
            min_value=datetime(2022, 1, 1),
            max_value=datetime.today(),
        )
    if len(date_range) != 2:
        st.warning("Please select a valid date range.")
        return

    start_date, end_date = date_range

    crypto_data = download_crypto_data(
        CRYPTO_SYMBOLS[selected_crypto], start_date, end_date
    )

    news_df = generate_mock_news(model)
    filtered_news = news_df[
        (news_df["date"] >= pd.Timestamp(start_date))
        & (news_df["date"] <= pd.Timestamp(end_date))
    ]
    sentiment_data = group_sentiment_data(filtered_news, start_date, end_date)

    col3, col4 = st.columns([2, 1])
    with col3:
        st.subheader("Charts")
        with st.container(height=1100, border=True):
            st.plotly_chart(
                plot_candlestick_with_separate_volume(crypto_data),
                use_container_width=True,
            )
            st.plotly_chart(plot_sentiment(sentiment_data), use_container_width=True)

    with col4:
        st.subheader("Information")
        with st.container(border=True, height=1100):
            display_news(filtered_news)


if __name__ == "__main__":
    main()
