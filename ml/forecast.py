"""
Loads the trained Random Forest model and predicts the next AQI value from the
latest data pulled out of the backend API. Intended to be run periodically
(e.g. every hour via cron) to push a forecast back into the system.
"""

import requests
import joblib
import pandas as pd

BACKEND_URL = "http://localhost:5000"


def build_features_from_history(hours=48):
    resp = requests.get(f"{BACKEND_URL}/api/history", params={"hours": hours})
    resp.raise_for_status()
    data = resp.json()
    if len(data) < 25:
        raise ValueError("Not enough history yet to build lag features (need 24h+).")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp").resample("1H").mean().interpolate()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    return df.dropna().iloc[[-1]]  # most recent complete row


def predict_next_aqi():
    model = joblib.load("rf_aqi_model.joblib")
    feature_cols = joblib.load("rf_feature_cols.joblib")

    latest_row = build_features_from_history()
    X = latest_row[[c for c in feature_cols if c in latest_row.columns]]
    prediction = model.predict(X)[0]
    return prediction


if __name__ == "__main__":
    aqi_forecast = predict_next_aqi()
    print(f"Predicted AQI for next hour: {aqi_forecast:.1f}")
