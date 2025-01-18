from datetime import datetime, timedelta

import joblib
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

CRYPTO_SYMBOLS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Ripple": "XRP-USD"}
SENTIMENTS = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {"positive": "green", "neutral": "gray", "negative": "red"}


def retrieve_news(selected_crypto, model):
    crypto_news = yf.Ticker(CRYPTO_SYMBOLS[selected_crypto]).get_news(1000)

    df = pd.DataFrame(
        [
            {
                "date": item["content"].get("pubDate", ""),
                "title": item["content"].get("title", ""),
                "text": item["content"].get("summary", ""),
                "url": item["clickThroughUrl"]["url"] if item.get("clickThroughUrl") else "",
            }
            for item in crypto_news
        ]
    )

    df["date"] = pd.to_datetime(df["date"])

    X_pred = df.drop(columns=["date"])

    df["sentiment"] = model.predict(X_pred, True)

    return df


def display_news(news_df):
    df = news_df.sort_values("date", ascending=False)

    for _, row in df.iterrows():
        sentiment_color = SENTIMENT_COLORS[row["sentiment"]]

        # Truncate the text for display
        text = row["text"][:400]
        if len(row["text"]) > 400:
            text += "..."

        # Include a container for title and sentiment, making sure the sentiment does not overlap
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; flex-grow: 1;">{row['title']}</h3>
                <div style="background-color: {sentiment_color}; color: white; 
                            padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                    {row['sentiment']}
                </div>
            </div>
            <small style="color: #555; font-style: italic;">Published on: {row['date']}</small>
            <p style="margin: 10px 0;">{text}</p>
            <a href="{row['url']}" style="color: #1a73e8; text-decoration: none;">Read more...</a>
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
        filtered_data.groupby(pd.Grouper(level=0, freq=freq))["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )
    return grouped


def download_crypto_data(symbol, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance."""
    if pd.Timedelta(end_date - start_date).days < 8:
        interval = "1h"
    elif pd.Timedelta(end_date - start_date).days < 60:
        interval = "60m"
    else:
        interval = "1d"

    data = (
        yf.Ticker(symbol)
        .history(interval=interval, start=start_date, end=end_date)
        .reset_index()
    )

    date_col = "Datetime" if "Datetime" in data.columns else "Date"    
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)

    freq = get_frequency(start_date, end_date)
    data = data.groupby(pd.Grouper(level=0, freq=freq)).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )

    return data


def plot_candlestick_with_separate_volume(data, min_date, max_date):
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
        title="Candlestick chart with Volume",
        xaxis=dict(
            range=[min_date, max_date],  # Set the range of x-axis
            rangeslider=dict(visible=False)  # Disable the range slider
        ),
        yaxis=dict(title="Price (USD)", side="left"),
        yaxis2=dict(title="Volume", side="left"),
        height=650,
        showlegend=False,
    )

    return fig


def plot_sentiment(data, min_date, max_date):
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

    max_y_value = data.sum(axis=1).max()  # Calculate maximum sum of counts for scaling

    fig.update_layout(
        barmode="stack",
        title="Sentiment Over Time",
        xaxis=dict(range=[min_date, max_date]),
        xaxis_title="Date",
        yaxis=dict(
            title="Counts",
            tickmode="linear" if max_y_value <= 10 else "auto",
            tick0=0,
            dtick=1 if max_y_value <= 10 else None,
            tickformat=".0f",
            range=[0, max_y_value + 1],
        ),
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


def main():
    st.set_page_config(layout="wide")
    apply_custom_styles()
    st.title("\U0001f4c8 Cryptocurrency Analysis Dashboard")

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

    start_date, end_date = [
        pd.Timestamp(date).tz_localize("UTC") for date in date_range
    ]

    crypto_data = download_crypto_data(
        CRYPTO_SYMBOLS[selected_crypto], start_date, end_date
    )

    model = joblib.load("res/models/nltk_rf_w2v_1.joblib")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")

    news_df = retrieve_news(selected_crypto, model)
    filtered_news = news_df[
        (news_df["date"] >= pd.Timestamp(start_date))
        & (news_df["date"] <= pd.Timestamp(end_date))
    ]
    sentiment_data = group_sentiment_data(filtered_news, start_date, end_date)

    common_index = crypto_data.index.union(sentiment_data.index)
    crypto_data = crypto_data.reindex(common_index, method=None)
    sentiment_data = sentiment_data.reindex(common_index, fill_value=0)

    min_date = min(crypto_data.index.min(), sentiment_data.index.min())
    max_date = max(crypto_data.index.max(), sentiment_data.index.max())

    col3, col4 = st.columns([3, 2])
    with col3:
        st.subheader("Charts")
        with st.container(height=1100, border=True):
            st.plotly_chart(
                plot_candlestick_with_separate_volume(crypto_data, min_date, max_date),
                use_container_width=True,
            )
            st.plotly_chart(plot_sentiment(sentiment_data, min_date, max_date), use_container_width=True)

    with col4:
        st.subheader("Information")
        with st.container(border=True, height=1100):
            display_news(filtered_news)


if __name__ == "__main__":
    main()
