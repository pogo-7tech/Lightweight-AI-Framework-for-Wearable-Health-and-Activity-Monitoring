"""
MQTT Publisher — Simulated Edge Wearable Device
================================================
Simulates a wearable sensor that:
  1. Generates synthetic PPG + IMU readings
  2. Runs edge inference (simulated HRV extraction + anomaly scoring)
  3. Publishes structured JSON telemetry to an MQTT broker

Usage (requires Mosquitto broker):
    pip install paho-mqtt
    mosquitto &   # Start local broker
    python src/cloud/mqtt_publisher.py

MQTT Topic: wearable/telemetry
Broker:     localhost:1883
QoS:        1 (at least once delivery)
"""

import json
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠️  paho-mqtt not installed. Running in SIMULATION mode (no broker needed).")
    print("    Install with: pip install paho-mqtt")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BROKER_HOST   = "localhost"
BROKER_PORT   = 1883
TOPIC         = "wearable/telemetry"
PUBLISH_HZ    = 1          # 1 telemetry packet per second
SUBJECT_ID    = "sub_demo_001"
ANOMALY_THRESHOLD = 0.05

# Activity schedule (seconds into session → activity)
ACTIVITY_SCHEDULE = [
    (0,   "sitting"),
    (60,  "walking"),
    (140, "running"),
    (220, "walking"),
    (270, "sitting"),
]

# Activity-aware alert thresholds
ALERT_RULES = {
    "sitting": {"hr_high": 100, "hr_low": 50, "rmssd_low": 15},
    "walking": {"hr_high": 130, "hr_low": 50, "rmssd_low": 10},
    "running": {"hr_high": 170, "hr_low": 60, "rmssd_low":  5},
}

TIER_MAP = {"sitting": 1, "walking": 2, "running": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Signal Simulator
# ─────────────────────────────────────────────────────────────────────────────
class WearableSimulator:
    """Simulates physiological signals based on current activity."""

    HR_BASE   = {"sitting": 68,  "walking": 95,  "running": 145}
    RMSSD_BASE = {"sitting": 42, "walking": 28,  "running": 15}

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.t = 0
        self._anomaly_counter = 0

    def get_activity(self) -> str:
        current = "sitting"
        for onset, activity in ACTIVITY_SCHEDULE:
            if self.t >= onset:
                current = activity
        return current

    def _inject_anomaly(self) -> bool:
        """Randomly inject anomaly every ~2 minutes."""
        self._anomaly_counter += 1
        return self._anomaly_counter % 120 < 8   # 8-second anomaly window

    def sample(self) -> dict:
        activity = self.get_activity()
        anomaly  = self._inject_anomaly()

        # Heart rate
        hr = self.HR_BASE[activity] + 5 * np.sin(2 * np.pi * self.t / 60)
        hr += np.random.randn() * 2.5
        if anomaly:
            hr += np.random.uniform(35, 50)   # tachycardia spike

        # RMSSD
        rmssd = self.RMSSD_BASE[activity] + np.random.randn() * 2.5
        if anomaly:
            rmssd = max(2.0, rmssd - 25)

        # SDNN (correlated with RMSSD)
        sdnn = rmssd * 1.15 + np.random.randn() * 3

        # AE reconstruction score
        ae_score = np.random.exponential(0.012)
        if anomaly:
            ae_score = np.random.uniform(0.07, 0.15)

        # Anomaly flag
        anomaly_flag = int(ae_score > ANOMALY_THRESHOLD)

        # Alert tier
        rules       = ALERT_RULES[activity]
        alert_active = (hr > rules["hr_high"] or hr < rules["hr_low"] or rmssd < rules["rmssd_low"])
        alert_tier  = TIER_MAP[activity] if alert_active else 0

        # Build payload
        payload = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "subject_id":     SUBJECT_ID,
            "session_time_s": self.t,
            "activity":       activity,
            "heart_rate_bpm": round(float(hr), 1),
            "rmssd_ms":       round(float(max(rmssd, 1.0)), 2),
            "sdnn_ms":        round(float(max(sdnn, 1.0)), 2),
            "ae_score":       round(float(ae_score), 6),
            "anomaly_flag":   anomaly_flag,
            "alert_tier":     alert_tier,
            "alert_active":   bool(alert_active),
        }

        self.t += 1
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# MQTT Client
# ─────────────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"  ✅ Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
    else:
        print(f"  ❌ Connection failed with code {rc}")


def run_publisher(duration_s: int = 300, simulate_only: bool = False):
    """Run the MQTT publisher for `duration_s` seconds."""
    simulator = WearableSimulator()
    client    = None

    if MQTT_AVAILABLE and not simulate_only:
        client = mqtt.Client(client_id=f"edge-{SUBJECT_ID}")
        client.on_connect = on_connect
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            client.loop_start()
        except ConnectionRefusedError:
            print("  ⚠️  Cannot connect to MQTT broker. Running in SIMULATION mode.")
            client = None

    print("=" * 60)
    print(f"  MQTT Publisher — Edge Wearable Simulator")
    print(f"  Topic   : {TOPIC}")
    print(f"  Subject : {SUBJECT_ID}")
    print(f"  Duration: {duration_s}s at {PUBLISH_HZ} Hz")
    print("=" * 60)

    interval = 1.0 / PUBLISH_HZ
    start    = time.time()

    for _ in range(duration_s):
        payload_dict = simulator.sample()
        payload_json = json.dumps(payload_dict)
        payload_bytes = len(payload_json.encode())

        if client is not None:
            result = client.publish(TOPIC, payload_json, qos=1)
            status = "✅" if result.rc == 0 else "❌"
        else:
            status = "🔵 SIM"

        print(f"  {status} t={payload_dict['session_time_s']:>4}s | "
              f"Activity={payload_dict['activity']:<8} | "
              f"HR={payload_dict['heart_rate_bpm']:>6.1f} bpm | "
              f"RMSSD={payload_dict['rmssd_ms']:>5.1f} ms | "
              f"AE={payload_dict['ae_score']:.5f} | "
              f"Alert={'🚨' if payload_dict['alert_active'] else '✅'} | "
              f"Bytes={payload_bytes}")

        elapsed = time.time() - start
        sleep_for = max(0, interval - (elapsed % interval))
        time.sleep(sleep_for)

        if time.time() - start >= duration_s:
            break

    if client is not None:
        client.loop_stop()
        client.disconnect()

    print(f"\n  ✅ Publisher finished after {duration_s}s.")


if __name__ == "__main__":
    # Run for 10 seconds in demo/simulation mode
    run_publisher(duration_s=10, simulate_only=True)
