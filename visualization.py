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
                title='Kripto Para Fiyat Grafiği ve Tahminler',
                xaxis_title='Tarih',
                yaxis_title='Fiyat (USDT)',
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
            # Create figure with multiple subplots
            from plotly.subplots import make_subplots
            
            # Create a 3x2 subplot grid with specific heights
            fig = make_subplots(
                rows=4, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=(
                    "Fiyat ve Hareketli Ortalamalar", 
                    "Göreceli Güç Endeksi (RSI)", 
                    "MACD (Hareketli Ortalama Yakınsama/Iraksama)", 
                    "Stokastik Osilatör"
                )
            )
            
            # Add price and moving averages to first plot
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Fiyat",
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add Bollinger Bands
            if all(band in df.columns for band in ['BB_high', 'BB_mid', 'BB_low']):
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['BB_high'],
                        mode='lines',
                        name='Bollinger Üst',
                        line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['BB_mid'],
                        mode='lines',
                        name='Bollinger Orta',
                        line=dict(color='rgba(0, 0, 250, 0.5)', width=1),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['BB_low'],
                        mode='lines',
                        name='Bollinger Alt',
                        line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 250, 0.1)',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add Moving Averages
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue', width=1.5),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='green', width=1.5),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if 'SMA_200' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['SMA_200'],
                        mode='lines',
                        name='SMA 200',
                        line=dict(color='red', width=1.5),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            if 'EMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['EMA_20'],
                        mode='lines',
                        name='EMA 20',
                        line=dict(color='orange', width=1.5, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add RSI to second subplot
            if 'RSI_14' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['RSI_14'],
                        mode='lines',
                        name='RSI 14',
                        line=dict(color='purple', width=1.5),
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
                # Add additional RSI periods if available
                if 'RSI_7' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['RSI_7'],
                            mode='lines',
                            name='RSI 7',
                            line=dict(color='blue', width=1, dash='dot'),
                            showlegend=True
                        ),
                        row=2, col=1
                    )
                
                if 'RSI_28' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['RSI_28'],
                            mode='lines',
                            name='RSI 28',
                            line=dict(color='orange', width=1, dash='dash'),
                            showlegend=True
                        ),
                        row=2, col=1
                    )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=70,
                    x1=df['timestamp'].iloc[-1],
                    y1=70,
                    line=dict(color='red', dash='dash', width=1),
                    row=2, col=1
                )
                
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=30,
                    x1=df['timestamp'].iloc[-1],
                    y1=30,
                    line=dict(color='green', dash='dash', width=1),
                    row=2, col=1
                )
                
                # Add a midline at 50
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=50,
                    x1=df['timestamp'].iloc[-1],
                    y1=50,
                    line=dict(color='grey', dash='dot', width=1),
                    row=2, col=1
                )
                
                # Set RSI range
                fig.update_yaxes(range=[0, 100], row=2, col=1)
            
            # Add MACD to third subplot
            if all(x in df.columns for x in ['MACD', 'MACD_signal', 'MACD_hist']):
                # MACD Line
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=1.5),
                        showlegend=True
                    ),
                    row=3, col=1
                )
                
                # Signal Line
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['MACD_signal'],
                        mode='lines',
                        name='Sinyal',
                        line=dict(color='red', width=1.5),
                        showlegend=True
                    ),
                    row=3, col=1
                )
                
                # Histogram
                colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['MACD_hist'],
                        name='Histogram',
                        marker_color=colors,
                        showlegend=True
                    ),
                    row=3, col=1
                )
                
                # Add zero line
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=0,
                    x1=df['timestamp'].iloc[-1],
                    y1=0,
                    line=dict(color='grey', dash='dot', width=1),
                    row=3, col=1
                )
            
            # Add Stochastic Oscillator to fourth subplot
            if all(x in df.columns for x in ['STOCH_K', 'STOCH_D']):
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['STOCH_K'],
                        mode='lines',
                        name='%K',
                        line=dict(color='blue', width=1.5),
                        showlegend=True
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['STOCH_D'],
                        mode='lines',
                        name='%D',
                        line=dict(color='red', width=1.5),
                        showlegend=True
                    ),
                    row=4, col=1
                )
                
                # Add overbought/oversold lines
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=80,
                    x1=df['timestamp'].iloc[-1],
                    y1=80,
                    line=dict(color='red', dash='dash', width=1),
                    row=4, col=1
                )
                
                fig.add_shape(
                    type='line',
                    x0=df['timestamp'].iloc[0],
                    y0=20,
                    x1=df['timestamp'].iloc[-1],
                    y1=20,
                    line=dict(color='green', dash='dash', width=1),
                    row=4, col=1
                )
                
                # Set Stochastic range
                fig.update_yaxes(range=[0, 100], row=4, col=1)
            
            # Update layout for the entire figure
            fig.update_layout(
                title='Teknik Göstergeler',
                height=1200,  # Increased height to fit all subplots
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=100, l=50, r=50, b=50)  # Adjust margins
            )
            
            # Hide rangeslider for price chart
            fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
            
            # Add Y-axis titles
            fig.update_yaxes(title_text="Fiyat (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Stokastik", row=4, col=1)
            
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
                    name='Saatlik Tahmin Doğruluğu',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
            
            # Daily prediction accuracy
            if 'daily_accuracy' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_accuracy'],
                    mode='lines+markers',
                    name='Günlük Tahmin Doğruluğu',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))
            
            # Update layout
            fig.update_layout(
                title='Zaman İçinde Tahmin Doğruluğu',
                xaxis_title='Tarih',
                yaxis_title='Doğruluk (1.0 = %100)',
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
                # Calculate whether predictions were too high or too low
                df['hourly_direction'] = (df['hourly_pred'] - df['hourly_actual']).apply(
                    lambda x: 'Yüksek' if x > 0 else 'Düşük' if x < 0 else 'Doğru'
                )
                
                # Add scatter plot for predictions
                hourly_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['hourly_pred'],
                    mode='lines+markers',
                    name='1 Saat Tahmini',
                    line=dict(color='blue', width=2),
                    marker=dict(
                        size=10,
                        color=df['hourly_direction'].map({
                            'Yüksek': 'blue', 
                            'Düşük': 'red',
                            'Doğru': 'green'
                        })
                    )
                ))
                
                # Add scatter plot for actual prices
                hourly_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['hourly_actual'],
                    mode='lines+markers',
                    name='Gerçek Fiyat (1s sonra)',
                    line=dict(color='black', width=2),
                    marker=dict(size=6)
                ))
                
                # Add arrows showing the difference
                for i, row in df.iterrows():
                    if pd.notna(row['hourly_pred']) and pd.notna(row['hourly_actual']):
                        color = 'blue' if row['hourly_pred'] > row['hourly_actual'] else 'red'
                        hourly_fig.add_shape(
                            type="line",
                            x0=row['timestamp'], y0=row['hourly_pred'],
                            x1=row['timestamp'], y1=row['hourly_actual'],
                            line=dict(color=color, width=1, dash="dot"),
                        )
                
                hourly_fig.update_layout(
                    title='1 Saatlik Tahminler - Gerçek Fiyatlar Karşılaştırması',
                    xaxis_title='Tarih',
                    yaxis_title='Fiyat (USDT)',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    annotations=[
                        dict(
                            x=0.02, y=0.98, 
                            xref="paper", yref="paper",
                            text="🔴 Düşük tahmin - 🔵 Yüksek tahmin", 
                            showarrow=False,
                            font=dict(size=12)
                        )
                    ]
                )
            
            # Create daily predictions chart
            daily_fig = go.Figure()
            
            if 'daily_pred' in df.columns and 'daily_actual' in df.columns:
                # Calculate whether predictions were too high or too low
                df['daily_direction'] = (df['daily_pred'] - df['daily_actual']).apply(
                    lambda x: 'Yüksek' if x > 0 else 'Düşük' if x < 0 else 'Doğru'
                )
                
                # Add scatter plot for predictions
                daily_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_pred'],
                    mode='lines+markers',
                    name='24 Saat Tahmini',
                    line=dict(color='green', width=2),
                    marker=dict(
                        size=10,
                        color=df['daily_direction'].map({
                            'Yüksek': 'blue', 
                            'Düşük': 'red',
                            'Doğru': 'green'
                        })
                    )
                ))
                
                # Add scatter plot for actual prices
                daily_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['daily_actual'],
                    mode='lines+markers',
                    name='Gerçek Fiyat (24s sonra)',
                    line=dict(color='black', width=2),
                    marker=dict(size=6)
                ))
                
                # Add arrows showing the difference
                for i, row in df.iterrows():
                    if pd.notna(row['daily_pred']) and pd.notna(row['daily_actual']):
                        color = 'blue' if row['daily_pred'] > row['daily_actual'] else 'red'
                        daily_fig.add_shape(
                            type="line",
                            x0=row['timestamp'], y0=row['daily_pred'],
                            x1=row['timestamp'], y1=row['daily_actual'],
                            line=dict(color=color, width=1, dash="dot"),
                        )
                
                daily_fig.update_layout(
                    title='24 Saatlik Tahminler - Gerçek Fiyatlar Karşılaştırması',
                    xaxis_title='Tarih',
                    yaxis_title='Fiyat (USDT)',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    annotations=[
                        dict(
                            x=0.02, y=0.98, 
                            xref="paper", yref="paper",
                            text="🔴 Düşük tahmin - 🔵 Yüksek tahmin", 
                            showarrow=False,
                            font=dict(size=12)
                        )
                    ]
                )
            
            return hourly_fig, daily_fig
        
        except Exception as e:
            print(f"Error creating prediction vs actual chart: {e}")
            return None, None
