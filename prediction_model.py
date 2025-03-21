import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PredictionModel:
    def __init__(self):
        """Initialize the prediction model."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_hourly = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_daily = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        if df is None or df.empty:
            return None
        
        try:
            # Features to use for prediction
            features = [
                'close', 'volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_high', 'BB_mid', 'BB_low', 'SMA_20', 'EMA_20', 'ROC'
            ]
            
            # Check if all features exist in dataframe
            for feature in features:
                if feature not in df.columns:
                    print(f"Warning: {feature} not found in dataframe")
                    if feature in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_high', 'BB_mid', 'BB_low', 'SMA_20', 'EMA_20', 'ROC']:
                        df[feature] = 0  # Add dummy feature
            
            # Filter only needed features
            available_features = [f for f in features if f in df.columns]
            df_features = df[available_features].copy()
            
            # Scale features
            df_scaled = self.scaler.fit_transform(df_features)
            
            return df_scaled
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def train_model(self, df):
        """Train the prediction models with historical data."""
        if df is None or df.empty:
            print("Error: Cannot train model with empty dataframe")
            return False
        
        try:
            df_scaled = self.prepare_features(df)
            if df_scaled is None:
                return False
            
            # Create sequences for time series prediction
            X, y_hourly, y_daily = [], [], []
            
            # Use 30 time steps to predict the next price
            sequence_length = 30
            
            for i in range(len(df_scaled) - sequence_length - 24):  # -24 to allow for daily prediction
                X.append(df_scaled[i:i+sequence_length])
                # Target for hourly prediction (next time step)
                y_hourly.append(df['close'].iloc[i+sequence_length])
                # Target for daily prediction (24 time steps ahead for daily data)
                y_daily.append(df['close'].iloc[i+sequence_length+24])
            
            X = np.array(X)
            y_hourly = np.array(y_hourly)
            y_daily = np.array(y_daily)
            
            # Reshape X for Random Forest (which expects 2D input)
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_test, y_hourly_train, y_hourly_test = train_test_split(
                X_reshaped, y_hourly, test_size=0.2, random_state=42)
            _, _, y_daily_train, y_daily_test = train_test_split(
                X_reshaped, y_daily, test_size=0.2, random_state=42)
            
            # Train models
            self.model_hourly.fit(X_train, y_hourly_train)
            self.model_daily.fit(X_train, y_daily_train)
            
            # Calculate training accuracy
            hourly_score = self.model_hourly.score(X_test, y_hourly_test)
            daily_score = self.model_daily.score(X_test, y_daily_test)
            
            print(f"Model trained. Hourly score: {hourly_score:.4f}, Daily score: {daily_score:.4f}")
            self.is_trained = True
            return True
        
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_prices(self, df, hourly_scaling=1.0, daily_scaling=1.0):
        """Predict next prices using the trained model."""
        if df is None or df.empty:
            print("Error: Cannot predict with empty dataframe")
            return None, None
        
        try:
            df_scaled = self.prepare_features(df)
            if df_scaled is None:
                return None, None
            
            if not self.is_trained:
                print("Warning: Model not trained, using simple estimation")
                # Simple estimation based on last 30 days trend
                last_30_close = df['close'].iloc[-30:].values
                growth_rate = (last_30_close[-1] / last_30_close[0]) ** (1/30) - 1
                
                hourly_prediction = last_30_close[-1] * (1 + growth_rate/24)  # Hourly growth rate
                daily_prediction = last_30_close[-1] * (1 + growth_rate)      # Daily growth rate
                
                # Apply scaling factors from feedback loop
                hourly_prediction *= hourly_scaling
                daily_prediction *= daily_scaling
                
                return hourly_prediction, daily_prediction
            
            # Use the last sequence_length data points for prediction
            sequence_length = 30
            X_live = df_scaled[-sequence_length:].reshape(1, -1)
            
            # Make predictions
            hourly_prediction = self.model_hourly.predict(X_live)[0]
            daily_prediction = self.model_daily.predict(X_live)[0]
            
            # Apply scaling factors from feedback loop
            hourly_prediction *= hourly_scaling
            daily_prediction *= daily_scaling
            
            return hourly_prediction, daily_prediction
        
        except Exception as e:
            print(f"Error predicting prices: {e}")
            last_close = df['close'].iloc[-1]
            return last_close * 1.002, last_close * 1.01  # Fallback values
    
    def predict_with_simple_model(self, df, hourly_scaling=1.0, daily_scaling=1.0):
        """Make simple predictions based on recent trends."""
        if df is None or df.empty:
            print("Error: Cannot predict with empty dataframe")
            return None, None
        
        try:
            # Calculate simple trend
            recent_data = df.tail(30)
            close_values = recent_data['close'].values
            
            # Calculate average daily percentage change
            daily_changes = []
            for i in range(1, len(close_values)):
                daily_changes.append(close_values[i] / close_values[i-1] - 1)
            
            avg_daily_change = np.mean(daily_changes) if daily_changes else 0.01
            
            # Make predictions
            last_price = df['close'].iloc[-1]
            hourly_prediction = last_price * (1 + avg_daily_change/24)  # Hourly change
            daily_prediction = last_price * (1 + avg_daily_change)      # Daily change
            
            # Consider RSI for overbought/oversold conditions
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if rsi > 70:  # Overbought
                    hourly_prediction *= 0.998
                    daily_prediction *= 0.995
                elif rsi < 30:  # Oversold
                    hourly_prediction *= 1.002
                    daily_prediction *= 1.005
            
            # Consider MACD trend
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                signal = df['MACD_signal'].iloc[-1]
                if macd > signal:  # Bullish
                    hourly_prediction *= 1.001
                    daily_prediction *= 1.002
                else:  # Bearish
                    hourly_prediction *= 0.999
                    daily_prediction *= 0.998
            
            # Apply scaling factors from feedback loop
            hourly_prediction *= hourly_scaling
            daily_prediction *= daily_scaling
            
            return hourly_prediction, daily_prediction
        
        except Exception as e:
            print(f"Error in simple prediction: {e}")
            return None, None
