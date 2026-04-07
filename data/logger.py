import paho.mqtt.client as mqtt
import csv, json, os
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = os.getenv("LOG_FILE", "data/events.csv")
os.makedirs("data", exist_ok=True)

FIELDS = ["timestamp", "hour", "topic", "key", "value"]

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    topic = msg.topic
    timestamp = payload.get("timestamp", "")
    hour = payload.get("hour", "")
    
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for key, val in payload.items():
            if key not in ("timestamp", "hour"):
                writer.writerow({"timestamp": timestamp, "hour": hour,
                                 "topic": topic, "key": key, "value": val})

client = mqtt.Client()
client.connect(os.getenv("MQTT_BROKER", "localhost"), int(os.getenv("MQTT_PORT", 1883)))
client.subscribe("home/#")  # subscribe to ALL home topics
client.on_message = on_message

print(f"📝 Logger running — saving events to {LOG_FILE}")
client.loop_forever()