"""
Backend for AIoT Air Quality project.
- Subscribes to MQTT topic where ESP32 nodes publish JSON readings
- Stores every reading in SQLite
- Exposes REST API for the dashboard (latest reading, historical range, alerts)

Run: python app.py
Requires an MQTT broker running (e.g. Mosquitto) at MQTT_BROKER below.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta

import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------- CONFIG ----------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "aqi/+/readings"   # + wildcard matches any node id
DB_PATH = "aqi_data.db"

# AQI category thresholds (CPCB-style) used for alerting
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]
ALERT_THRESHOLD = 200  # trigger alert at "Poor" and above
# -----------------------------------------

app = Flask(__name__)
CORS(app)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            timestamp TEXT,
            pm1 REAL, pm25 REAL, pm10 REAL,
            co_ppm REAL, nh3_ppm REAL,
            temp REAL, humidity REAL,
            sub_index_pm25 REAL, sub_index_pm10 REAL,
            sub_index_co REAL, sub_index_nh3 REAL,
            aqi REAL
        )
    """)
    conn.commit()
    conn.close()


def aqi_category(aqi):
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Hazardous" if aqi > 500 else "Unknown"


# ---------------- MQTT ----------------
def on_connect(client, userdata, flags, rc):
    print("MQTT connected with code", rc)
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        conn = get_db()
        conn.execute("""
            INSERT INTO readings
            (node_id, timestamp, pm1, pm25, pm10, co_ppm, nh3_ppm, temp, humidity,
             sub_index_pm25, sub_index_pm10, sub_index_co, sub_index_nh3, aqi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("node_id"),
            datetime.utcnow().isoformat(),
            data.get("pm1"), data.get("pm25"), data.get("pm10"),
            data.get("co_ppm"), data.get("nh3_ppm"),
            data.get("temp"), data.get("humidity"),
            data.get("sub_index_pm25"), data.get("sub_index_pm10"),
            data.get("sub_index_co"), data.get("sub_index_nh3"),
            data.get("aqi"),
        ))
        conn.commit()
        conn.close()
        print(f"Stored reading from {data.get('node_id')}: AQI={data.get('aqi')}")
    except Exception as e:
        print("Error handling MQTT message:", e)


def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


# ---------------- REST API ----------------
@app.route("/api/latest")
def latest():
    """Latest reading, optionally filtered by node_id."""
    node_id = request.args.get("node_id")
    conn = get_db()
    if node_id:
        row = conn.execute(
            "SELECT * FROM readings WHERE node_id=? ORDER BY id DESC LIMIT 1", (node_id,)
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "no data yet"}), 404
    result = dict(row)
    result["category"] = aqi_category(result["aqi"])
    result["alert"] = result["aqi"] >= ALERT_THRESHOLD
    return jsonify(result)


@app.route("/api/history")
def history():
    """Historical readings for the last N hours (default 24)."""
    hours = int(request.args.get("hours", 24))
    node_id = request.args.get("node_id")
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    conn = get_db()
    if node_id:
        rows = conn.execute(
            "SELECT * FROM readings WHERE timestamp >= ? AND node_id=? ORDER BY id ASC",
            (since, node_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM readings WHERE timestamp >= ? ORDER BY id ASC", (since,)
        ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts")
def alerts():
    """Recent readings that crossed the alert threshold."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM readings WHERE aqi >= ? ORDER BY id DESC LIMIT 20", (ALERT_THRESHOLD,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    init_db()
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
