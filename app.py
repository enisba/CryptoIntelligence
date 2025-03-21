import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from crypto_predictor import CryptoPredictor

# Set page config
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource
def get_predictor():
    return CryptoPredictor()

predictor = get_predictor()

# App title
st.title("📈 Cryptocurrency Price Predictor")
st.markdown("### With Feedback Loop Learning System")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "Binance Coin": "BNBUSDT",
    "Ripple": "XRPUSDT",
    "Cardano": "ADAUSDT",
    "Solana": "SOLUSDT",
    "Dogecoin": "DOGEUSDT",
    "Polkadot": "DOTUSDT",
    "Avalanche": "AVAXUSDT",
    "Chainlink": "LINKUSDT"
}
selected_crypto_name = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
selected_crypto = crypto_options[selected_crypto_name]

# Time interval selection
interval_options = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d",
    "1 Week": "1w"
}
selected_interval_name = st.sidebar.selectbox("Select Time Interval", list(interval_options.keys()))
selected_interval = interval_options[selected_interval_name]

# Historical data limit
limit_options = {
    "100 data points": 100,
    "200 data points": 200,
    "500 data points": 500,
    "1000 data points": 1000
}
selected_limit_name = st.sidebar.selectbox("Select Data Amount", list(limit_options.keys()))
selected_limit = limit_options[selected_limit_name]

# Button to fetch data and make prediction
if st.sidebar.button("Fetch Data & Predict"):
    with st.spinner("Fetching and processing cryptocurrency data..."):
        # Fetch and process data
        df = predictor.fetch_and_process_data(selected_crypto, selected_interval, selected_limit)
        
        if df is not None and not df.empty:
            st.session_state['df'] = df
            st.session_state['symbol'] = selected_crypto
            st.session_state['last_updated'] = datetime.now()
            
            # Train model
            with st.spinner("Training prediction model..."):
                predictor.train_prediction_model(df)
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                hourly_pred, daily_pred = predictor.make_predictions(df, selected_crypto)
                
                if hourly_pred is not None and daily_pred is not None:
                    st.session_state['hourly_pred'] = hourly_pred
                    st.session_state['daily_pred'] = daily_pred
                    st.session_state['has_prediction'] = True
                else:
                    st.error("Failed to generate predictions. Please try again.")
        else:
            st.error("Failed to fetch data. Please check your network connection or try a different cryptocurrency.")

# Auto refresh data toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
refresh_interval = st.sidebar.slider("Refresh interval (minutes)", 1, 60, 5)

# Explanation section
with st.sidebar.expander("About the Feedback Loop"):
    st.markdown("""
    The prediction model improves over time by comparing past predictions with actual prices.
    
    **How it works:**
    1. Make price predictions
    2. Store predictions with timestamps
    3. Retrieve actual prices when they become available
    4. Calculate prediction accuracy
    5. Adjust the model based on historical accuracy
    6. Use improved model for future predictions
    
    This creates a continuous learning cycle that improves prediction accuracy over time.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Display price chart
    st.subheader("Price Chart & Predictions")
    
    if 'df' in st.session_state and 'has_prediction' in st.session_state and st.session_state['has_prediction']:
        # Create visualization
        fig = predictor.create_visualization(
            st.session_state['df'],
            st.session_state.get('hourly_pred'),
            st.session_state.get('daily_pred')
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Failed to create visualization.")
    else:
        st.info("Please fetch data to see the price chart and predictions.")

with col2:
    # Display current data and predictions
    st.subheader("Current Data & Predictions")
    
    if 'df' in st.session_state and 'has_prediction' in st.session_state and st.session_state['has_prediction']:
        # Display current price
        current_price = st.session_state['df']['close'].iloc[-1]
        st.metric(
            label=f"Current {selected_crypto_name} Price",
            value=f"${current_price:.2f}"
        )
        
        # Display hourly prediction
        hourly_pred = st.session_state['hourly_pred']
        hourly_change = (hourly_pred - current_price) / current_price * 100
        st.metric(
            label="1 Hour Prediction",
            value=f"${hourly_pred:.2f}",
            delta=f"{hourly_change:.2f}%"
        )
        
        # Display daily prediction
        daily_pred = st.session_state['daily_pred']
        daily_change = (daily_pred - current_price) / current_price * 100
        st.metric(
            label="24 Hour Prediction",
            value=f"${daily_pred:.2f}",
            delta=f"{daily_change:.2f}%"
        )
        
        # Display last updated time
        if 'last_updated' in st.session_state:
            st.text(f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("Please fetch data to see the current price and predictions.")

# Technical indicators tab
st.subheader("Technical Indicators")
if 'df' in st.session_state:
    # Technical indicators chart
    tech_chart = predictor.create_technical_charts(st.session_state['df'])
    if tech_chart is not None:
        st.plotly_chart(tech_chart, use_container_width=True)
    else:
        st.warning("Failed to create technical indicators chart.")
else:
    st.info("Please fetch data to see technical indicators.")

# Prediction accuracy tab
st.subheader("Prediction Accuracy & History")
if 'symbol' in st.session_state:
    # Get accuracy metrics
    accuracy_metrics = predictor.get_prediction_accuracy(st.session_state['symbol'])
    
    # Display metrics
    if accuracy_metrics['sample_size'] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hourly_acc = accuracy_metrics.get('hourly_accuracy')
            if hourly_acc is not None:
                st.metric(
                    label="Hourly Prediction Accuracy",
                    value=f"{hourly_acc*100:.2f}%"
                )
            else:
                st.metric(label="Hourly Prediction Accuracy", value="No data")
        
        with col2:
            daily_acc = accuracy_metrics.get('daily_accuracy')
            if daily_acc is not None:
                st.metric(
                    label="Daily Prediction Accuracy",
                    value=f"{daily_acc*100:.2f}%"
                )
            else:
                st.metric(label="Daily Prediction Accuracy", value="No data")
        
        with col3:
            st.metric(
                label="Verified Predictions",
                value=str(accuracy_metrics['sample_size'])
            )
        
        # Accuracy charts
        accuracy_chart, hourly_compare, daily_compare = predictor.create_accuracy_charts(st.session_state['symbol'])
        
        if accuracy_chart is not None:
            st.plotly_chart(accuracy_chart, use_container_width=True)
        
        if hourly_compare is not None:
            st.plotly_chart(hourly_compare, use_container_width=True)
        
        if daily_compare is not None:
            st.plotly_chart(daily_compare, use_container_width=True)
    else:
        st.info("No verified predictions yet. The system needs to collect actual price data before calculating accuracy.")
else:
    st.info("Please fetch data to see prediction accuracy metrics and history.")

# Auto-refresh functionality
if auto_refresh and 'symbol' in st.session_state:
    # Calculate time since last update
    if 'last_updated' in st.session_state:
        time_diff = (datetime.now() - st.session_state['last_updated']).total_seconds() / 60
        if time_diff >= refresh_interval:
            st.warning("Auto-refreshing data...")
            # Fetch and process data
            df = predictor.fetch_and_process_data(st.session_state['symbol'], selected_interval, selected_limit)
            
            if df is not None and not df.empty:
                st.session_state['df'] = df
                st.session_state['last_updated'] = datetime.now()
                
                # Make predictions
                hourly_pred, daily_pred = predictor.make_predictions(df, st.session_state['symbol'])
                
                if hourly_pred is not None and daily_pred is not None:
                    st.session_state['hourly_pred'] = hourly_pred
                    st.session_state['daily_pred'] = daily_pred
                    st.session_state['has_prediction'] = True
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("Cryptocurrency Price Predictor with Feedback Loop Learning System")
st.markdown("Data provided by Binance API")
