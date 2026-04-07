import paho.mqtt.client as mqtt
import time, random, json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))

client = mqtt.Client()
client.connect(BROKER, PORT)

DEVICES = {
    "home/livingroom/light": lambda: {"state": random.choice(["ON", "OFF"]), "brightness": random.randint(20, 100)},
    "home/bedroom/thermostat": lambda: {"temp_c": round(random.uniform(18, 24), 1), "mode": random.choice(["heat", "cool", "auto"])},
    "home/frontdoor/motion": lambda: {"detected": random.choice([True, False])},
}

print("📡 Simulator running — publishing device data every 5s. Press Ctrl+C to stop.")

while True:
    hour = datetime.now().hour
    for topic, payload_fn in DEVICES.items():
        payload = payload_fn()
        payload["timestamp"] = datetime.now().isoformat()
        payload["hour"] = hour
        client.publish(topic, json.dumps(payload))
        print(f"  → {topic}: {payload}")
    time.sleep(5)