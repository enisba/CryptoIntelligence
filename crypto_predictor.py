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
        df = self.data_handler.get_binance_data(symbol, interval, limit)
        if df is None or df.empty:
            return None
        df = self.data_handler.calculate_technical_indicators(df)
        return df

    def train_prediction_model(self, df):
        """Train the prediction model."""
        return self.model.train_model(df)

    def make_predictions(self, df, symbol="BTCUSDT"):
        """Make price predictions with feedback loop integration."""
        if df is None or df.empty:
            return None, None

        self.data_handler.update_actual_prices(symbol)
        hourly_scaling, daily_scaling = self.data_handler.get_scaling_factors(symbol)
        hourly_pred, daily_pred = self.model.predict_next_prices(df, hourly_scaling, daily_scaling)

        if hourly_pred is None or daily_pred is None:
            hourly_pred, daily_pred = self.model.predict_with_simple_model(df, hourly_scaling, daily_scaling)

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

    def auto_retrain_if_needed(self, df, symbol="BTCUSDT", error_threshold=0.05):
        """Retrain the model if accuracy drops below a threshold."""
        accuracy = self.get_prediction_accuracy(symbol)
        if accuracy and accuracy.get('mean_absolute_percentage_error', 0) > error_threshold:
            print(f"[INFO] Retraining model for {symbol} due to high error.")
            self.train_prediction_model(df)

    def detect_market_trend(self, df):
        """Detect simple market trend based on recent prices."""
        recent = df.tail(10)
        if recent['close'].iloc[-1] > recent['close'].mean():
            return "Uptrend"
        elif recent['close'].iloc[-1] < recent['close'].mean():
            return "Downtrend"
        else:
            return "Sideways"

    def compute_risk_score(self, df):
        """Compute a basic risk score based on price volatility."""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        return round(volatility * 100, 2)  # risk score in percentage

    def feedback_loop(self, symbol="BTCUSDT"):
        """Trigger feedback loop retraining if recent prediction error is high."""
        history = self.get_prediction_history(symbol)
        if history is None or len(history) < 5:
            return
        recent = history.tail(5)
        errors = abs(recent['actual'] - recent['predicted']) / recent['actual']
        mean_error = errors.mean()
        if mean_error > 0.1:
            print("[WARNING] High prediction error. Feedback loop triggered.")
            df = self.fetch_and_process_data(symbol)
            self.train_prediction_model(df)

    def generate_alerts(self, df, hourly_pred, daily_pred):
        """Generate alerts based on predicted price movement."""
        current_price = df['close'].iloc[-1]
        alert_msg = ""
        if hourly_pred > current_price * 1.02:
            alert_msg += "[ALERT] Hourly prediction indicates a strong upward move.\n"
        elif hourly_pred < current_price * 0.98:
            alert_msg += "[ALERT] Hourly prediction indicates a potential drop.\n"
        
        if daily_pred > current_price * 1.05:
            alert_msg += "[ALERT] Daily prediction suggests significant bullish momentum.\n"
        elif daily_pred < current_price * 0.95:
            alert_msg += "[ALERT] Daily prediction suggests possible bearish trend.\n"

        return alert_msg.strip()