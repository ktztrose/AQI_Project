"""
Converts a CPCB-format air quality Excel/CSV file into aqi_dataset.json
for the dashboard to display.

Expected input columns: From Date, To Date, PM2.5, PM10, NH3, CO
(same format as CPCB's CCR portal hourly download)

Usage:
    python convert_dataset.py path/to/your_dataset.xlsx

Re-run this anytime you get an updated dataset -- it regenerates
aqi_dataset.json, and the dashboard picks up the new data automatically
next time it's loaded (no HTML changes needed).
"""

import sys
import json
import pandas as pd
import numpy as np

BP_PM25 = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,380,401,500)]
BP_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,510,401,500)]
BP_CO   = [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)]
BP_NH3  = [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,2400,401,500)]

# Set to True to cap AQI at 500 (common convention) instead of extrapolating past it
CAP_AT_500 = True


def sub_index(conc, breakpoints):
    if pd.isna(conc):
        return np.nan
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
    c_low, c_high, i_low, i_high = breakpoints[-1]
    if conc > c_high:
        val = ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
        return min(val, 500) if CAP_AT_500 else val
    return np.nan


def aqi_category(aqi):
    if pd.isna(aqi): return None
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def dominant_pollutant(row):
    subs = {"PM2.5": row["sub_index_pm25"], "PM10": row["sub_index_pm10"],
            "CO": row["sub_index_co"], "NH3": row["sub_index_nh3"]}
    subs = {k: v for k, v in subs.items() if not pd.isna(v)}
    return max(subs, key=subs.get) if subs else None


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_dataset.py path/to/dataset.xlsx")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["From Date"] = pd.to_datetime(df["From Date"], format="%d-%m-%Y %H:%M", errors="coerce")

    df["sub_index_pm25"] = df["PM2.5"].apply(lambda x: sub_index(x, BP_PM25))
    df["sub_index_pm10"] = df["PM10"].apply(lambda x: sub_index(x, BP_PM10))
    df["sub_index_co"]   = df["CO"].apply(lambda x: sub_index(x, BP_CO))
    df["sub_index_nh3"]  = df["NH3"].apply(lambda x: sub_index(x, BP_NH3))

    df["AQI"] = df[["sub_index_pm25", "sub_index_pm10", "sub_index_co", "sub_index_nh3"]].max(axis=1)
    df["Category"] = df["AQI"].apply(aqi_category)
    df["Dominant"] = df.apply(dominant_pollutant, axis=1)

    df = df.dropna(subset=["From Date", "AQI"]).sort_values("From Date")

    records = []
    for _, row in df.iterrows():
        records.append({
            "timestamp": row["From Date"].strftime("%Y-%m-%dT%H:%M:%S"),
            "pm25": None if pd.isna(row["PM2.5"]) else round(float(row["PM2.5"]), 1),
            "pm10": None if pd.isna(row["PM10"]) else round(float(row["PM10"]), 1),
            "co": None if pd.isna(row["CO"]) else round(float(row["CO"]), 2),
            "nh3": None if pd.isna(row["NH3"]) else round(float(row["NH3"]), 1),
            "aqi": round(float(row["AQI"]), 1),
            "category": row["Category"],
            "dominant": row["Dominant"],
        })

    with open("aqi_dataset.json", "w") as f:
        json.dump(records, f, indent=None)

    print(f"Wrote {len(records)} records to aqi_dataset.json")
    print(f"Date range: {records[0]['timestamp']} to {records[-1]['timestamp']}")


if __name__ == "__main__":
    main()
