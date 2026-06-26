"""
InfluxDB Writer — MQTT → InfluxDB Subscriber
=============================================
Subscribes to the MQTT topic and writes incoming telemetry
into InfluxDB 2.x for persistent time-series storage.

Usage (requires InfluxDB 2.x + Mosquitto):
    docker compose -f src/cloud/docker-compose.yml up -d
    python src/cloud/influxdb_writer.py

Dependencies:
    pip install paho-mqtt influxdb-client
"""

import json
from datetime import datetime, timezone

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
MQTT_TOPIC    = "wearable/telemetry"

INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "wearable-dev-token-2026"     # set via env var in production
INFLUX_ORG    = "wearable_lab"
INFLUX_BUCKET = "wearable_metrics"


# ─────────────────────────────────────────────────────────────────────────────
# InfluxDB Writer
# ─────────────────────────────────────────────────────────────────────────────
def payload_to_point(payload: dict) -> "Point":
    """Convert a telemetry dict to an InfluxDB Point."""
    point = (
        Point("physiological_metrics")
        .tag("subject_id", payload.get("subject_id", "unknown"))
        .tag("activity",   payload.get("activity",   "unknown"))
        .field("heart_rate_bpm", float(payload["heart_rate_bpm"]))
        .field("rmssd_ms",       float(payload["rmssd_ms"]))
        .field("sdnn_ms",        float(payload.get("sdnn_ms", 0.0)))
        .field("ae_score",       float(payload["ae_score"]))
        .field("anomaly_flag",   int(payload["anomaly_flag"]))
        .field("alert_tier",     int(payload.get("alert_tier", 0)))
        .time(payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
              WritePrecision.SECONDS)
    )
    return point


class InfluxWriter:
    def __init__(self):
        self.client    = None
        self.write_api = None
        self.connected = False

        if INFLUX_AVAILABLE:
            try:
                self.client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.connected = True
                print(f"  ✅ Connected to InfluxDB at {INFLUX_URL}")
            except Exception as e:
                print(f"  ⚠️  InfluxDB unavailable: {e}. Running in dry-run mode.")
        else:
            print("  ⚠️  influxdb-client not installed. Running in dry-run mode.")

    def write(self, payload: dict):
        if self.connected:
            point = payload_to_point(payload)
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        else:
            # Dry-run: just log the payload
            print(f"  [DRY-RUN] Would write: HR={payload['heart_rate_bpm']:.1f} | "
                  f"Activity={payload['activity']} | AE={payload['ae_score']:.5f}")

    def close(self):
        if self.client:
            self.client.close()


# ─────────────────────────────────────────────────────────────────────────────
# MQTT Subscriber
# ─────────────────────────────────────────────────────────────────────────────
influx_writer = InfluxWriter()


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        influx_writer.write(payload)
        flag = "🚨" if payload.get("alert_active") else "  "
        print(f"  {flag} t={payload.get('session_time_s', '?'):>4}s | "
              f"Activity={payload['activity']:<8} | "
              f"HR={payload['heart_rate_bpm']:>6.1f} | "
              f"RMSSD={payload['rmssd_ms']:>5.1f} | "
              f"Written ✅")
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}")
    except Exception as e:
        print(f"  ❌ Write error: {e}")


def run_subscriber():
    if not MQTT_AVAILABLE:
        print("⚠️  paho-mqtt not installed. Cannot subscribe.")
        return

    client = mqtt.Client(client_id="influxdb-subscriber")
    client.on_message = on_message

    print("=" * 60)
    print("  InfluxDB Writer — MQTT Subscriber")
    print(f"  Broker : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Topic  : {MQTT_TOPIC}")
    print(f"  InfluxDB: {INFLUX_URL}/{INFLUX_BUCKET}")
    print("=" * 60)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.subscribe(MQTT_TOPIC, qos=1)
        print(f"  ✅ Subscribed to {MQTT_TOPIC}. Waiting for messages...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n  Subscriber stopped.")
    except ConnectionRefusedError:
        print("  ❌ Cannot connect to MQTT broker. Is Mosquitto running?")
    finally:
        client.disconnect()
        influx_writer.close()


if __name__ == "__main__":
    run_subscriber()
