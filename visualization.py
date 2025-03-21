import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class Visualizer:
    def create_price_chart(self, df, hourly_pred=None, daily_pred=None):
        """Create price chart with predictions."""
        if df is None or df.empty:
            return None
        
        try:
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                opacity=0.7
            ))
            
            # Add moving averages if available
            if 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ))
            
            if 'EMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['EMA_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            # Add Bollinger Bands if available
            if all(band in df.columns for band in ['BB_high', 'BB_mid', 'BB_low']):
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_high'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(250, 0, 0, 0.3)', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_low'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(250, 0, 0, 0.3)', width=1),
                    fill='tonexty', 
                    fillcolor='rgba(250, 0, 0, 0.05)'
                ))
            
            # Add predictions if available
            if hourly_pred is not None and daily_pred is not None:
                last_time = df['timestamp'].iloc[-1]
                future_times = [last_time + timedelta(hours=1), last_time + timedelta(days=1)]
                future_prices = [hourly_pred, daily_pred]
                
                fig.add_trace(go.Scatter(
                    x=future_times,
                    y=future_prices,
                    mode='markers+lines',
                    name='Predictions',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8, symbol='star', color='red')
                ))
            
            # Update layout
            fig.update_layout(
                title='Cryptocurrency Price Chart with Predictions',
                xaxis_title='Date',
                yaxis_title='Price (USDT)',
                xaxis_rangeslider_visible=False,
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating price chart: {e}")
            return None
    
    def create_technical_indicators_chart(self, df):
        """Create chart of technical indicators."""
        if df is None or df.empty:
            return None
        
        try:
            # Create figure with subplots
            fig = go.Figure()
            
            # RSI subplot
            if 'RSI' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ))
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=70,
                    x1=df['timestamp'].iloc[-1],
                    y1=70,
                    line=dict(color='red', dash='dash', width=1)
                )
                
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=30,
                    x1=df['timestamp'].iloc[-1],
                    y1=30,
                    line=dict(color='green', dash='dash', width=1)
                )
                
                # Update layout
                fig.update_layout(
                    title='Relative Strength Index (RSI)',
                    xaxis_title='Date',
                    yaxis_title='RSI',
                    height=300,
                    yaxis=dict(range=[0, 100])
                )
            
            return fig
        
        except Exception as e:
            print(f"Error creating technical indicators chart: {e}")
            return None
    
    def create_prediction_accuracy_chart(self, predictions_history):
        """Create chart showing prediction accuracy over time."""
        if not predictions_history:
            return None
        
        try:
            # Extract verified predictions with accuracy data
            verified = [p for p in predictions_history if p.get('verified', False)]
            
            if not verified:
                return None
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(verified)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create figure
            fig = go.Figure()
            
            # Hourly prediction accuracy
            if 'hourly_accuracy' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['hourly_accuracy'],
                    mode='lines+markers',
                    name='Hourly Prediction Accuracy',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
            
            # Daily prediction accuracy
            if 'daily_accuracy' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_accuracy'],
                    mode='lines+markers',
                    name='Daily Prediction Accuracy',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))
            
            # Update layout
            fig.update_layout(
                title='Prediction Accuracy Over Time',
                xaxis_title='Date',
                yaxis_title='Accuracy (1.0 = 100%)',
                height=400,
                yaxis=dict(range=[0, 1.1]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating prediction accuracy chart: {e}")
            return None
    
    def create_prediction_vs_actual_chart(self, predictions_history):
        """Create chart comparing predicted vs actual prices."""
        if not predictions_history:
            return None
        
        try:
            # Extract verified predictions with actual prices
            verified = [p for p in predictions_history if p.get('verified', False)]
            
            if not verified:
                return None
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(verified)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create hourly predictions chart
            hourly_fig = go.Figure()
            
            if 'hourly_pred' in df.columns and 'hourly_actual' in df.columns:
                hourly_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['hourly_pred'],
                    mode='lines+markers',
                    name='Hourly Prediction',
                    line=dict(color='blue', width=2)
                ))
                
                hourly_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['hourly_actual'],
                    mode='lines+markers',
                    name='Actual Price (1h later)',
                    line=dict(color='black', width=2)
                ))
                
                hourly_fig.update_layout(
                    title='Hourly Predictions vs Actual Prices',
                    xaxis_title='Date',
                    yaxis_title='Price (USDT)',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
            
            # Create daily predictions chart
            daily_fig = go.Figure()
            
            if 'daily_pred' in df.columns and 'daily_actual' in df.columns:
                daily_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_pred'],
                    mode='lines+markers',
                    name='Daily Prediction',
                    line=dict(color='green', width=2)
                ))
                
                daily_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_actual'],
                    mode='lines+markers',
                    name='Actual Price (24h later)',
                    line=dict(color='black', width=2)
                ))
                
                daily_fig.update_layout(
                    title='Daily Predictions vs Actual Prices',
                    xaxis_title='Date',
                    yaxis_title='Price (USDT)',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
            
            return hourly_fig, daily_fig
        
        except Exception as e:
            print(f"Error creating prediction vs actual chart: {e}")
            return None, None
