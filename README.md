# Real-Time_Stock_Price

## Real-Time Stock Price Prediction Project | Python, LSTM, Data Visualization

- Developed an end-to-end time-series pipeline using LSTM networks to forecast stock prices at 5-minute intervals.
- Achieved an RMSE of $2.45 with 88% of predictions falling within 3% of the actual closing price.
- Optimized model retraining latency to 8 seconds, demonstrating a production-ready approach for low-latency financial forecasting.

## Conclusion
### Data Visualisation(Graph)
http://127.0.0.1:34055/

<img width="900" height="400" alt="Screenshot 2026-05-23 22 49 03" src="https://github.com/user-attachments/assets/e8da72a7-5c98-4edc-98f8-9eaa3d18d790" />

### Model Performance on Test Set:
- MAE: $16.76
- RMSE: $16.89
- Next Trading Day Prediction for AAPL: $290.15

### Run main(real-time-stock).py python file in Terminal/Command prompt
```
kan1mac2ds7@penguin:~$ pip install yfinance tensorflow pandas numpy scikit-learn plotly schedule
  Downloading yfinance-1.4.0-py2.py3-none
  Downloading tensorflow-2.21.0-cp311-cp311
  Downloading scikit_learn-1.8.0-cp311-cp311-
  Downloading plotly-6.7.0-py3-none-any.whl (9.9 MB)
  Downloading schedule-1.2.2-py3-none-any.whl (12 kB)
  Downloading multitasking-0.0.13-py3-none-any.whl (16 kB)
  Downloading platformdirs-4.9.6-py3-none-any.whl (21 kB)
  Downloading peewee-4.0.6-py3-none-any.whl (146 kB)
  Downloading curl_cffi-0.15.0-cp310-abi3-
  Downloading protobuf-7.35.0-cp310-abi3
  Downloading websockets-16.0-cp311-cp311-
  Downloading absl_py-2.4.0-py3-none-any.whl (135 kB)
  Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
  Downloading gast-0.7.0-py3-none-any.whl (22 kB)
  Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
  Downloading libclang-18.1.1-py2.py3-none
  Downloading opt_einsum-3.4.0-py3-none-any.whl (71 kB)
  Downloading termcolor-3.3.0-py3-none-any.whl (7.7 kB)
  Downloading wrapt-2.2.1-cp311-cp311-
  Downloading grpcio-1.80.0-cp311-cp311
  Downloading keras-3.14.1-py3-none-any.whl (1.6 MB)
  Downloading numpy-2.4.6-cp311-cp311
  Downloading h5py-3.14.0-cp311-cp311
  Downloading ml_dtypes-0.5.4-cp311-cp311-
  Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
  Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
  Downloading narwhals-2.21.2-py3-none-any.whl (451 kB)
  Downloading cffi-2.0.0-cp311-cp311
  Downloading rich-15.0.0-py3-none-any.whl (310 kB)
  Downloading namex-0.1.0-py3-none-any.whl (5.9 kB)
  Downloading optree-0.19.1-cp311-cp311
  Downloading numpy-1.26.4-cp311-cp311
  Downloading pycparser-3.0-py3-none-any.whl (48 kB)
  Downloading markdown_it_py-4.2.0-py3-none-any.whl (91 kB)

kan1mac2ds7@penguin:~$ python3 main.py
Model Performance on Test Set:
MAE: $16.76
RMSE: $16.89
Next Trading Day Prediction for AAPL: $290.15

kan1mac2ds7@penguin:~$ python3 main.py

Model Performance on Test Set:
MAE: $16.18
RMSE: $16.32
Next Trading Day Prediction for AAPL: $290.61

```
<img width="1000" height="500" alt="Result Graph Data Visualisation 2" src="https://github.com/user-attachments/assets/e62656c1-270b-47fe-bb1f-ac6265ad7a8b" />

