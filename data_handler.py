import os
import json
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    def __init__(self, storage_file="prediction_history.json"):
        """Initialize the DataHandler with storage file path."""
        self.storage_file = storage_file
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Create storage file if it doesn't exist
        if not os.path.exists(storage_file):
            with open(storage_file, 'w') as f:
                json.dump([], f)
    
    def get_binance_data(self, symbol="BTCUSDT", interval="1d", limit=1000):
        """Fetch cryptocurrency data from Binance API."""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            data = requests.get(url).json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                '_', '_', '_', '_', '_', '_'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the given dataframe."""
        if df is None or df.empty:
            return None
            
        try:
            # RSI
            df["RSI"] = ta.rsi(df["close"], length=14)
            
            # MACD
            macd = ta.macd(df["close"], fast=12, slow=26)
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"] = macd["MACDh_12_26_9"]
            
            # Bollinger Bands
            bbands = ta.bbands(df["close"], length=20)
            df["BB_high"] = bbands["BBU_20_2.0"]
            df["BB_mid"] = bbands["BBM_20_2.0"]
            df["BB_low"] = bbands["BBL_20_2.0"]
            
            # Moving Averages
            df["SMA_20"] = ta.sma(df["close"], length=20)
            df["EMA_20"] = ta.ema(df["close"], length=20)
            
            # Momentum
            df["ROC"] = ta.roc(df["close"], length=10)
            
            # Fill NaN values with mean
            df.fillna(df.mean(numeric_only=True), inplace=True)
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return df
    
    def save_prediction(self, symbol, timestamp, actual_price, hourly_pred, daily_pred):
        """Save prediction to storage."""
        try:
            with open(self.storage_file, 'r') as f:
                predictions = json.load(f)
            
            # Add new prediction
            predictions.append({
                'symbol': symbol,
                'timestamp': timestamp.isoformat(),
                'actual_price': float(actual_price),
                'hourly_pred': float(hourly_pred),
                'daily_pred': float(daily_pred),
                'verified': False
            })
            
            with open(self.storage_file, 'w') as f:
                json.dump(predictions, f)
                
        except Exception as e:
            print(f"Error saving prediction: {e}")
    
    def get_predictions_history(self, symbol="BTCUSDT"):
        """Get all historical predictions for a symbol."""
        try:
            with open(self.storage_file, 'r') as f:
                predictions = json.load(f)
            
            # Filter by symbol
            return [p for p in predictions if p['symbol'] == symbol]
        except Exception as e:
            print(f"Error getting prediction history: {e}")
            return []
    
    def update_actual_prices(self, symbol="BTCUSDT"):
        """Update actual prices for past predictions."""
        try:
            with open(self.storage_file, 'r') as f:
                predictions = json.load(f)
            
            if not predictions:
                return
            
            # Get current data
            current_data = self.get_binance_data(symbol=symbol, interval="1h", limit=168)  # Last 7 days hourly
            if current_data is None:
                return
            
            updated = False
            for i, pred in enumerate(predictions):
                if pred['symbol'] == symbol and not pred['verified']:
                    pred_time = datetime.fromisoformat(pred['timestamp'])
                    
                    # Check for hourly prediction (1 hour after prediction time)
                    hourly_check_time = pred_time + timedelta(hours=1)
                    
                    # Check for daily prediction (24 hours after prediction time)
                    daily_check_time = pred_time + timedelta(days=1)
                    
                    # Find closest data point for hourly
                    closest_hourly = current_data.iloc[
                        (current_data['timestamp'] - hourly_check_time).abs().argsort()[:1]
                    ]
                    
                    # Find closest data point for daily
                    closest_daily = current_data.iloc[
                        (current_data['timestamp'] - daily_check_time).abs().argsort()[:1]
                    ]
                    
                    # Only update if the times are close enough (within 30 minutes)
                    if len(closest_hourly) > 0 and abs((closest_hourly['timestamp'].iloc[0] - hourly_check_time).total_seconds()) < 1800:
                        predictions[i]['hourly_actual'] = float(closest_hourly['close'].iloc[0])
                        updated = True
                    
                    if len(closest_daily) > 0 and abs((closest_daily['timestamp'].iloc[0] - daily_check_time).total_seconds()) < 1800:
                        predictions[i]['daily_actual'] = float(closest_daily['close'].iloc[0])
                        updated = True
                    
                    # Mark as verified if both predictions have been checked
                    if 'hourly_actual' in predictions[i] and 'daily_actual' in predictions[i]:
                        predictions[i]['verified'] = True
                        
                        # Calculate accuracy
                        hourly_accuracy = 1 - abs(predictions[i]['hourly_actual'] - predictions[i]['hourly_pred']) / predictions[i]['hourly_actual']
                        daily_accuracy = 1 - abs(predictions[i]['daily_actual'] - predictions[i]['daily_pred']) / predictions[i]['daily_actual']
                        
                        predictions[i]['hourly_accuracy'] = float(hourly_accuracy)
                        predictions[i]['daily_accuracy'] = float(daily_accuracy)
            
            if updated:
                with open(self.storage_file, 'w') as f:
                    json.dump(predictions, f)
        
        except Exception as e:
            print(f"Error updating actual prices: {e}")
    
    def get_accuracy_metrics(self, symbol="BTCUSDT"):
        """Get accuracy metrics for past predictions."""
        try:
            with open(self.storage_file, 'r') as f:
                predictions = json.load(f)
            
            # Filter verified predictions for the symbol
            verified = [p for p in predictions if p['symbol'] == symbol and p.get('verified', False)]
            
            if not verified:
                return {
                    'hourly_accuracy': None,
                    'daily_accuracy': None,
                    'sample_size': 0
                }
            
            hourly_accuracies = [p.get('hourly_accuracy', 0) for p in verified if 'hourly_accuracy' in p]
            daily_accuracies = [p.get('daily_accuracy', 0) for p in verified if 'daily_accuracy' in p]
            
            return {
                'hourly_accuracy': sum(hourly_accuracies) / len(hourly_accuracies) if hourly_accuracies else None,
                'daily_accuracy': sum(daily_accuracies) / len(daily_accuracies) if daily_accuracies else None,
                'sample_size': len(verified)
            }
        
        except Exception as e:
            print(f"Error calculating accuracy metrics: {e}")
            return {
                'hourly_accuracy': None,
                'daily_accuracy': None,
                'sample_size': 0
            }
    
    def get_scaling_factors(self, symbol="BTCUSDT"):
        """Calculate scaling factors based on previous prediction accuracy."""
        metrics = self.get_accuracy_metrics(symbol)
        
        hourly_scaling = 1.0
        daily_scaling = 1.0
        
        # Default scaling factors if no historical data
        if metrics['sample_size'] == 0:
            return hourly_scaling, daily_scaling
        
        # Calculate scaling factors based on historical accuracy
        if metrics['hourly_accuracy'] is not None:
            if metrics['hourly_accuracy'] < 0.9:
                # If predictions tend to be too low
                hourly_errors = []
                with open(self.storage_file, 'r') as f:
                    predictions = json.load(f)
                
                verified = [p for p in predictions if p['symbol'] == symbol and p.get('verified', False)]
                for p in verified:
                    if 'hourly_actual' in p and 'hourly_pred' in p:
                        error = (p['hourly_actual'] - p['hourly_pred']) / p['hourly_actual']
                        hourly_errors.append(error)
                
                if hourly_errors:
                    avg_error = sum(hourly_errors) / len(hourly_errors)
                    hourly_scaling = 1.0 + avg_error
        
        # Same for daily predictions
        if metrics['daily_accuracy'] is not None:
            if metrics['daily_accuracy'] < 0.9:
                daily_errors = []
                with open(self.storage_file, 'r') as f:
                    predictions = json.load(f)
                
                verified = [p for p in predictions if p['symbol'] == symbol and p.get('verified', False)]
                for p in verified:
                    if 'daily_actual' in p and 'daily_pred' in p:
                        error = (p['daily_actual'] - p['daily_pred']) / p['daily_actual']
                        daily_errors.append(error)
                
                if daily_errors:
                    avg_error = sum(daily_errors) / len(daily_errors)
                    daily_scaling = 1.0 + avg_error
        
        # Ensure scaling factors are within reasonable bounds
        hourly_scaling = max(0.8, min(1.2, hourly_scaling))
        daily_scaling = max(0.8, min(1.2, daily_scaling))
        
        return hourly_scaling, daily_scaling
