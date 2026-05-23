### Project Overview

- Data Source: Yahoo Finance (via yfinance)
- Prediction Model: LSTM (Long Short-Term Memory) neural network
- Visualization: Plotly interactive charts
- Real-Time Simulation: Loop that fetches latest data every 60 seconds and updates predictions

### Features

- Fetches real-time (or latest) stock data for any ticker (e.g., AAPL, TSLA)
- Trains LSTM on historical closing prices
- Predicts next day's closing price based on last window_size days
- Visualizes historical prices, predicted vs actual for test set, and latest prediction
- Optional auto-refresh loop for live monitoring

### Real-Time Simulation

The run_realtime() method:

- Fetches new data every refresh_interval seconds
- Retrains the model on the extended dataset (incremental learning can be added)
- Outputs the latest price and next-day prediction

Note: Free Yahoo Finance data has ~15 min delay. For true real-time (seconds), consider Alpha Vantage or WebSocket APIs.

### Expected Output

- An interactive Plotly chart with:
  - Blue line = historical prices
  - Orange dashed line = predicted prices on test set
  - Red star = next trading day's predicted price
- Console prints MAE, RMSE, and the prediction value.

- This project provides a solid foundation for real-time financial forecasting using deep learning. For production, consider using incremental learning (e.g., model.fit with epochs=1 on new data) to avoid full retraining.


1. Data Fetching: Uses yfinance to get historical closing prices (default 6 months).
2. Preprocessing: Scales prices to [0,1] range and creates sliding windows of 60 days as input features.
3. LSTM Model: Two LSTM layers with dropout for regularization, followed by a dense output layer.
4. Training: Trains on 80% of data, validates on 20%, uses early stopping.
5. Prediction: For a new day, feeds the last 60 days into the model to predict the next closing price.
6. Visualization: Plotly shows actual vs predicted prices, error bars, and marks the next-day prediction.
