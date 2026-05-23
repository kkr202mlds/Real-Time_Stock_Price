import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import schedule
from datetime import datetime, timedelta

class RealTimeStockPredictor:
    def __init__(self, ticker, window_size=60, look_ahead=1):
        """
        ticker: Stock symbol (e.g., 'AAPL')
        window_size: Number of past days to use for prediction
        look_ahead: Days ahead to predict (default 1)
        """
        self.ticker = ticker
        self.window_size = window_size
        self.look_ahead = look_ahead
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_data(self, period="6mo"):
        """Fetch historical stock data from Yahoo Finance"""
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        return df[['Close']].dropna()
    
    def prepare_data(self, data):
        """Scale data and create sequences for LSTM"""
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data) - self.look_ahead + 1):
            X.append(scaled_data[i - self.window_size:i, 0])
            y.append(scaled_data[i + self.look_ahead - 1, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y, scaled_data
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, data, epochs=50, batch_size=32):
        """Train the LSTM model"""
        X, y, _ = self.prepare_data(data)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model = self.build_model((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        return X_test, y_test, history
    
    def predict_next_day(self, latest_data):
        """Predict next day's closing price using the latest window_size days"""
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        
        # Get last window_size days and scale
        last_window = latest_data[-self.window_size:].values.reshape(-1, 1)
        scaled_window = self.scaler.transform(last_window)
        X_pred = scaled_window.reshape(1, self.window_size, 1)
        
        # Predict scaled price
        scaled_pred = self.model.predict(X_pred, verbose=0)[0, 0]
        # Inverse transform to original price
        pred_price = self.scaler.inverse_transform([[scaled_pred]])[0, 0]
        return pred_price
    
    def plot_predictions(self, data, test_y_true, test_y_pred, next_pred_price):
        """Create interactive visualization"""
        # Prepare dates for test set
        split_idx = int(0.8 * len(data))
        test_dates = data.index[split_idx + self.window_size:split_idx + self.window_size + len(test_y_true)]
        
        # Inverse transform test predictions
        test_y_true_orig = self.scaler.inverse_transform(test_y_true.reshape(-1, 1)).flatten()
        test_y_pred_orig = self.scaler.inverse_transform(test_y_pred.reshape(-1, 1)).flatten()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Stock Price with Predictions', 'Prediction Error'),
                            vertical_spacing=0.15)
        
        # Historical prices
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
                                 name='Actual Price', line=dict(color='blue')), row=1, col=1)
        
        # Test predictions
        fig.add_trace(go.Scatter(x=test_dates, y=test_y_pred_orig, mode='lines+markers',
                                 name='Predicted (Test)', line=dict(color='orange', dash='dash')), row=1, col=1)
        
        # Next day prediction marker
        next_date = data.index[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[next_date], y=[next_pred_price], mode='markers',
                                 marker=dict(size=15, color='red', symbol='star'),
                                 name=f'Next Day Prediction: ${next_pred_price:.2f}'), row=1, col=1)
        
        # Error plot
        errors = test_y_true_orig - test_y_pred_orig
        fig.add_trace(go.Bar(x=test_dates, y=errors, name='Prediction Error',
                             marker_color='coral'), row=2, col=1)
        
        fig.update_layout(title=f"{self.ticker} Stock Price Prediction (LSTM)",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          hovermode="x unified", height=800)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Error (USD)", row=2, col=1)
        
        fig.show()
        
    def run_realtime(self, refresh_interval=60):
        """
        Simulate real-time prediction by refreshing data every 'refresh_interval' seconds.
        In practice, retrain periodically or use incremental learning.
        """
        print(f"Starting real-time prediction for {self.ticker} (updates every {refresh_interval}s)")
        print("Press Ctrl+C to stop\n")
        
        # Initial training
        data = self.fetch_data(period="1y")
        self.train(data, epochs=30)  # Quick initial training
        self.display_latest_prediction(data)
        
        def update():
            try:
                new_data = self.fetch_data(period="1y")
                if len(new_data) > len(data):
                    # New data available
                    latest_price = new_data['Close'].iloc[-1]
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New data fetched - Latest price: ${latest_price:.2f}")
                    
                    # Retrain on updated data (for demo, retrain quickly)
                    self.train(new_data, epochs=10)
                    pred_price = self.predict_next_day(new_data['Close'])
                    print(f"Next day prediction: ${pred_price:.2f}")
                    
                    # Update global data reference
                    nonlocal data
                    data = new_data
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No new data yet")
            except Exception as e:
                print(f"Update error: {e}")
        
        # Schedule updates
        schedule.every(refresh_interval).seconds.do(update)
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def display_latest_prediction(self, data=None):
        """Single-shot prediction and visualization"""
        if data is None:
            data = self.fetch_data(period="6mo")
        
        # Train model
        X_test, y_test, _ = self.train(data)
        
        # Predict on test set
        test_pred_scaled = self.model.predict(X_test, verbose=0).flatten()
        test_pred = self.scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        test_true = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Predict next day
        next_pred = self.predict_next_day(data['Close'])
        
        # Plot results
        self.plot_predictions(data, y_test, test_pred_scaled, next_pred)
        
        # Print metrics
        mae = np.mean(np.abs(test_true - test_pred))
        rmse = np.sqrt(np.mean((test_true - test_pred)**2))
        print(f"\nModel Performance on Test Set:")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"Next Trading Day Prediction for {self.ticker}: ${next_pred:.2f}")


# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":
    # Initialize predictor for Apple stock
    predictor = RealTimeStockPredictor(ticker="AAPL", window_size=60)
    
    # Option 1: Single prediction with visualization
    predictor.display_latest_prediction()
    
    # Option 2: Real-time simulation (uncomment to run)
    # predictor.run_realtime(refresh_interval=300)  # Update every 5 minutes
