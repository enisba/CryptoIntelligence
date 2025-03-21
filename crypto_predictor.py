import os
import pandas as pd
from datetime import datetime
from data_handler import DataHandler
from prediction_model import PredictionModel
from visualization import Visualizer

class CryptoPredictor:
    def __init__(self):
        """Initialize the CryptoPredictor with necessary components."""
        self.data_handler = DataHandler()
        self.model = PredictionModel()
        self.visualizer = Visualizer()
    
    def fetch_and_process_data(self, symbol="BTCUSDT", interval="1d", limit=1000):
        """Fetch and process cryptocurrency data."""
        # Fetch data from Binance
        df = self.data_handler.get_binance_data(symbol, interval, limit)
        
        if df is None or df.empty:
            return None
        
        # Calculate technical indicators
        df = self.data_handler.calculate_technical_indicators(df)
        
        return df
    
    def train_prediction_model(self, df):
        """Train the prediction model."""
        return self.model.train_model(df)
    
    def make_predictions(self, df, symbol="BTCUSDT"):
        """Make price predictions with feedback loop integration."""
        if df is None or df.empty:
            return None, None
        
        # Update actual prices for previous predictions
        self.data_handler.update_actual_prices(symbol)
        
        # Get scaling factors based on previous prediction accuracy
        hourly_scaling, daily_scaling = self.data_handler.get_scaling_factors(symbol)
        
        # Make predictions
        hourly_pred, daily_pred = self.model.predict_next_prices(df, hourly_scaling, daily_scaling)
        
        # If model-based prediction fails, use simple model
        if hourly_pred is None or daily_pred is None:
            hourly_pred, daily_pred = self.model.predict_with_simple_model(df, hourly_scaling, daily_scaling)
        
        # Save prediction
        current_time = datetime.now()
        current_price = float(df['close'].iloc[-1])
        self.data_handler.save_prediction(symbol, current_time, current_price, hourly_pred, daily_pred)
        
        return hourly_pred, daily_pred
    
    def get_prediction_accuracy(self, symbol="BTCUSDT"):
        """Get accuracy metrics for past predictions."""
        return self.data_handler.get_accuracy_metrics(symbol)
    
    def get_prediction_history(self, symbol="BTCUSDT"):
        """Get historical predictions."""
        return self.data_handler.get_predictions_history(symbol)
    
    def create_visualization(self, df, hourly_pred=None, daily_pred=None):
        """Create visualization for the data and predictions."""
        return self.visualizer.create_price_chart(df, hourly_pred, daily_pred)
    
    def create_technical_charts(self, df):
        """Create charts for technical indicators."""
        return self.visualizer.create_technical_indicators_chart(df)
    
    def create_accuracy_charts(self, symbol="BTCUSDT"):
        """Create charts showing prediction accuracy."""
        predictions_history = self.get_prediction_history(symbol)
        accuracy_chart = self.visualizer.create_prediction_accuracy_chart(predictions_history)
        hourly_compare, daily_compare = self.visualizer.create_prediction_vs_actual_chart(predictions_history)
        
        return accuracy_chart, hourly_compare, daily_compare
