import os
import json
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pycoingecko import CoinGeckoAPI

class DataHandler:
    def __init__(self, storage_file="prediction_history.json"):
        """Initialize the DataHandler with storage file path."""
        self.storage_file = storage_file
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.cg = CoinGeckoAPI()
        
        self.crypto_mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BNBUSDT": "binancecoin",
            "XRPUSDT": "ripple",
            "ADAUSDT": "cardano",
            "SOLUSDT": "solana",
            "DOGEUSDT": "dogecoin",
            "DOTUSDT": "polkadot",
            "AVAXUSDT": "avalanche-2",
            "LINKUSDT": "chainlink",
            "LTCUSDT": "litecoin",
            "MATICUSDT": "matic-network",
            "UNIUSDT": "uniswap",
            "XLMUSDT": "stellar",
            "ATOMUSDT": "cosmos",
            "TRXUSDT": "tron",
            "XMRUSDT": "monero",
            "EOSUSDT": "eos",
            "VETUSDT": "vechain",
            "ALGOUSDT": "algorand"
        }
        
        if not os.path.exists(storage_file):
            with open(storage_file, 'w') as f:
                json.dump([], f)
    
    def get_binance_data(self, symbol="BTCUSDT", interval="1d", limit=1000):
        """Fetch cryptocurrency data using CoinGecko API instead of Binance."""
        try:
            if symbol in self.crypto_mapping:
                coin_id = self.crypto_mapping[symbol]
            else:
                print(f"Unknown symbol: {symbol}")
                return None
            
            days = 1
            if interval == "1d":
                days = min(limit, 365) 
            elif interval == "1h":
                days = min(limit // 24, 90)  
            else:
                days = 30 
                
            coin_data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days,
                interval='daily' if interval == '1d' else None
            )
            
            prices = coin_data['prices']
            volumes = coin_data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            
            df = df.merge(volume_df, on='timestamp', how='inner')
            
            df['open'] = df['close']
            df['high'] = df['close'] * 1.005  
            df['low'] = df['close'] * 0.995  
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
        except Exception as e:
            print(f"Error fetching data from CoinGecko: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the given dataframe."""
        if df is None or df.empty:
            return None
            
        try:
            df["RSI_14"] = ta.rsi(df["close"], length=14)
            df["RSI_7"] = ta.rsi(df["close"], length=7)
            df["RSI_28"] = ta.rsi(df["close"], length=28)
            
            macd = ta.macd(df["close"], fast=12, slow=26)
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"] = macd["MACDh_12_26_9"]
            
            bbands = ta.bbands(df["close"], length=20)
            df["BB_high"] = bbands["BBU_20_2.0"]
            df["BB_mid"] = bbands["BBM_20_2.0"]
            df["BB_low"] = bbands["BBL_20_2.0"]
            
            df["SMA_20"] = ta.sma(df["close"], length=20)
            df["SMA_50"] = ta.sma(df["close"], length=50)
            df["SMA_200"] = ta.sma(df["close"], length=200)
            df["EMA_20"] = ta.ema(df["close"], length=20)
            df["EMA_50"] = ta.ema(df["close"], length=50)
            
            df["ROC"] = ta.roc(df["close"], length=10)
            df["MOM"] = ta.mom(df["close"], length=14)
            
            stoch = ta.stoch(high=df["high"], low=df["low"], close=df["close"], k=14, d=3)
            df["STOCH_K"] = stoch["STOCHk_14_3_3"]
            df["STOCH_D"] = stoch["STOCHd_14_3_3"]
            
            df["ATR"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)
            
            df["CCI"] = ta.cci(high=df["high"], low=df["low"], close=df["close"], length=20)
            
            df["OBV"] = ta.obv(close=df["close"], volume=df["volume"])
            
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
            
            current_data = self.get_binance_data(symbol=symbol, interval="1h", limit=168) 
            if current_data is None:
                return
            
            updated = False
            for i, pred in enumerate(predictions):
                if pred['symbol'] == symbol and not pred['verified']:
                    pred_time = datetime.fromisoformat(pred['timestamp'])
                    
                    hourly_check_time = pred_time + timedelta(hours=1)
                    daily_check_time = pred_time + timedelta(days=1)
                    closest_hourly = current_data.iloc[
                        (current_data['timestamp'] - hourly_check_time).abs().argsort()[:1]
                    ]
                    
                    closest_daily = current_data.iloc[
                        (current_data['timestamp'] - daily_check_time).abs().argsort()[:1]
                    ]
                    
                    if len(closest_hourly) > 0 and abs((closest_hourly['timestamp'].iloc[0] - hourly_check_time).total_seconds()) < 1800:
                        predictions[i]['hourly_actual'] = float(closest_hourly['close'].iloc[0])
                        updated = True
                    
                    if len(closest_daily) > 0 and abs((closest_daily['timestamp'].iloc[0] - daily_check_time).total_seconds()) < 1800:
                        predictions[i]['daily_actual'] = float(closest_daily['close'].iloc[0])
                        updated = True
                    
                    if 'hourly_actual' in predictions[i] and 'daily_actual' in predictions[i]:
                        predictions[i]['verified'] = True
                        
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
        
        if metrics['sample_size'] == 0:
            return hourly_scaling, daily_scaling
        
        if metrics['hourly_accuracy'] is not None:
            if metrics['hourly_accuracy'] < 0.9:
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
        
        hourly_scaling = max(0.8, min(1.2, hourly_scaling))
        daily_scaling = max(0.8, min(1.2, daily_scaling))
        
        return hourly_scaling, daily_scaling
