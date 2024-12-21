import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import random

# Constants
CRYPTO_SYMBOLS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Ripple": "XRP-USD"
}

# Helper Functions
def generate_mock_data():
    """Generate mock data for sentiment analysis."""
    dates = pd.date_range(start="2022-01-01", end=datetime.today(), freq="h")
    data = {
        "Date": dates,
        "Change": np.random.choice(["Positive", "Negative", "Neutral"], len(dates))
    }
    return pd.DataFrame(data)


def generate_mock_news():
    dates = pd.date_range(start="2022-01-01", end=datetime.today(), freq="h")
    num_entries = len(dates)
    data = {
        "date": dates,
        "title": [f"News Title {i}" for i in range(1, num_entries + 1)],
        "content": [f"This is a preview of content for news {i}. It gives some details about the topic." for i in range(1, num_entries + 1)],
        "url": [f"https://example.com/news{i}" for i in range(1, num_entries + 1)],
        "source": [random.choice(["Source A", "Source B", "Source C"]) for _ in range(num_entries)],
        "sentiment": [random.choice(["Positive", "Neutral", "Negative"]) for _ in range(num_entries)]
    }
    return pd.DataFrame(data)


def display_news(news_df):
    # Create a scrollable container
    for _, row in news_df.iterrows():
        sentiment_color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}[row["sentiment"]]
        
        # HTML structure to display title, content, and sentiment label
        st.markdown(
            f"""
            <div>
                <h3 style="margin: 0;">{row['title']}</h3>
                <p style="margin: 5px 0;">{row['content'][:100]}...</p>
                <a href="{row['url']}" style="color: #1a73e8; text-decoration: none;">Read more...</a>
            </div>
            <div style="position: absolute; top: 15px; right: 15px; background-color: {sentiment_color}; color: white; 
                        padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                {row['sentiment']}
            </div>
            """,
            unsafe_allow_html=True
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


def group_data(df, start_date, end_date):
    """Group data based on the selected date range."""
    filtered_data = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    filtered_data.set_index("Date", inplace=True)
    freq = get_frequency(start_date, end_date)
    grouped = filtered_data.groupby(pd.Grouper(freq=freq))["Change"].value_counts().unstack(fill_value=0)
    return grouped


def download_crypto_data(symbol, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance."""
    data = yf.download(symbol, interval="60m",
                           start=start_date, end=end_date, multi_level_index=False)

    freq = get_frequency(start_date, end_date)
    data = data.groupby(pd.Grouper(level=0, freq=freq)).agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    
    return data


def plot_candlestick_with_separate_volume(data):
    """Create a candlestick chart with a separate subplot for volume."""
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,  # Adjust space between subplots
        row_heights=[0.7, 0.3],  # Allocate more space for candlesticks
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ), row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="lightblue",  # Light blue for volume bars
    ), row=2, col=1)

    # Update layout to hide the range slider
    fig.update_layout(
        title=f"Candlestick chart with Volume",
        xaxis=dict(
            rangeslider=dict(visible=False)  # Disable the range slider
        ),
        yaxis=dict(title="Price (USD)", side="left"),
        yaxis2=dict(title="Volume", side="left"),
        height=650,  # Adjust chart height for better visibility
        showlegend=False,  # Remove legend if unnecessary
    )

    return fig


def plot_sentiment(data):
    """Create a stacked bar chart for sentiment analysis."""
    fig = go.Figure()
    for sentiment, color in zip(["Positive", "Neutral", "Negative"], ["green", "gray", "red"]):
        fig.add_trace(go.Bar(x=data.index, y=data.get(sentiment, 0), name=sentiment, marker_color=color))
    fig.update_layout(barmode="stack", title="Sentiment Over Time", xaxis_title="Date", yaxis_title="Counts", legend={"orientation": "h"}, height=400)
    return fig


# Streamlit App Layout
def apply_custom_styles():
    """Apply custom CSS styles."""
    st.markdown("""
        <style>
        .block-container {padding: 1rem;}
        .custom-title {font-size: 1.5rem; font-weight: bold; text-align: center; margin-bottom: 10px;}
        .scroll-box {border: 1px solid #ccc; padding: 10px; height: 700px; overflow-y: auto;}
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide")
    apply_custom_styles()
    st.title("\U0001F4C8 Cryptocurrency Analysis Dashboard")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("Select Cryptocurrency:", list(CRYPTO_SYMBOLS.keys()))
    with col2:
        date_range = st.date_input("Select Date Range:", [datetime.today() - timedelta(days=30), datetime.today()],
                                   min_value=datetime(2022, 1, 1), max_value=datetime.today())
    if len(date_range) != 2:
        st.warning("Please select a valid date range.")
        return
    
    start_date, end_date = date_range

    # Mock Sentiment Analysis
    df = generate_mock_data()
    sentiment_data = group_data(df, start_date, end_date)

    # Cryptocurrency Data
    crypto_data = download_crypto_data(CRYPTO_SYMBOLS[selected_crypto], start_date, end_date)

    # Layout
    col3, col4 = st.columns([2, 1])
    with col3:
        st.subheader("Charts")
        with st.container(height=1100, border=True):
            st.plotly_chart(plot_candlestick_with_separate_volume(crypto_data), use_container_width=True)
            st.plotly_chart(plot_sentiment(sentiment_data), use_container_width=True)

    with col4:
        st.subheader("Information")
        with st.container(border=True, height=1100):
            news_df = generate_mock_news()
            filtered_news = news_df[(news_df["date"] >= pd.Timestamp(start_date)) & (news_df["date"] <= pd.Timestamp(end_date))]
            display_news(filtered_news)


if __name__ == "__main__":
    main()
