"""
AQI Forecasting - trains and compares three models:
  1. ARIMA        (classical time-series baseline)
  2. Random Forest (feature-based ML baseline)
  3. LSTM         (deep learning sequence model)

Input: CSV export of your collected readings (from the backend SQLite DB), with at
least columns: timestamp, aqi, pm25, pm10, co_ppm, nh3_ppm, temp, humidity

Export your data first, e.g.:
  sqlite3 -header -csv backend/aqi_data.db "SELECT * FROM readings;" > ml/data/readings.csv

Usage:
  python train_forecast.py --data data/readings.csv --target_horizon 1
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")


def load_and_prepare(path, horizon):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["aqi"])

    # Resample to hourly average if you have frequent readings (adjust as needed)
    df = df.set_index("timestamp").resample("1H").mean().interpolate().reset_index()

    # Feature engineering: lag features + time-of-day
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # Target: AQI `horizon` steps ahead
    df["target"] = df["aqi"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df


def run_arima(df, horizon):
    from statsmodels.tsa.arima.model import ARIMA

    series = df["aqi"]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    model = ARIMA(train, order=(2, 1, 2))
    fitted = model.fit()
    preds = fitted.forecast(steps=len(test))

    mae = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    return {"model": "ARIMA", "mae": mae, "rmse": rmse}


def run_random_forest(df):
    feature_cols = [c for c in df.columns if c.startswith("aqi_lag_")] + \
                   ["pm25", "pm10", "co_ppm", "nh3_ppm", "temp", "humidity", "hour", "dayofweek"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    joblib.dump(model, "rf_aqi_model.joblib")
    joblib.dump(feature_cols, "rf_feature_cols.joblib")

    # Feature importance -- useful for your report (which pollutant drives predictions)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nRandom Forest feature importances:\n", importances)

    return {"model": "RandomForest", "mae": mae, "rmse": rmse}


def run_lstm(df, horizon):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    seq_len = 24  # use last 24 hourly readings to predict next
    values = df["aqi"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - seq_len - horizon):
        X.append(scaled[i:i + seq_len, 0])
        y.append(scaled[i + seq_len + horizon - 1, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, activation="relu", input_shape=(seq_len, 1), return_sequences=True),
        LSTM(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))

    model.save("lstm_aqi_model.keras")
    joblib.dump(scaler, "lstm_scaler.joblib")

    return {"model": "LSTM", "mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV export of readings")
    parser.add_argument("--target_horizon", type=int, default=1, help="Hours ahead to forecast")
    args = parser.parse_args()

    df = load_and_prepare(args.data, args.target_horizon)
    print(f"Loaded {len(df)} hourly rows after cleaning.")

    results = []
    try:
        results.append(run_arima(df, args.target_horizon))
    except Exception as e:
        print("ARIMA failed:", e)

    try:
        results.append(run_random_forest(df))
    except Exception as e:
        print("Random Forest failed:", e)

    try:
        results.append(run_lstm(df, args.target_horizon))
    except Exception as e:
        print("LSTM failed (is tensorflow installed?):", e)

    print("\n=== Model comparison (lower MAE/RMSE = better) ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv("model_comparison.csv", index=False)
    print("\nSaved comparison to model_comparison.csv -- use this table directly in your report.")


if __name__ == "__main__":
    main()
