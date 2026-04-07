import pandas as pd
import random
from datetime import datetime, timedelta

rows = []
base_time = datetime(2026, 4, 1, 0, 0, 0)

for day in range(7):  # 7 days of data
    for hour in range(24):
        timestamp = base_time + timedelta(days=day, hours=hour)

        # Realistic light pattern — ON in morning and evening
        if 6 <= hour <= 9 or 17 <= hour <= 23:
            light_state = random.choices(["ON", "OFF"], weights=[80, 20])[0]
        else:
            light_state = random.choices(["ON", "OFF"], weights=[10, 90])[0]

        # Realistic temperature pattern — warmer midday
        if 6 <= hour <= 9:
            temp = round(random.uniform(19, 21), 1)
            mode = "heat"
        elif 10 <= hour <= 16:
            temp = round(random.uniform(22, 24), 1)
            mode = "cool"
        elif 17 <= hour <= 22:
            temp = round(random.uniform(20, 22), 1)
            mode = "auto"
        else:
            temp = round(random.uniform(18, 20), 1)
            mode = "heat"

        # Motion — more likely when awake
        motion = random.choices(
            [True, False],
            weights=[70, 30] if 7 <= hour <= 23 else [10, 90]
        )[0]

        rows.extend([
            [timestamp.isoformat(), hour, "home/livingroom/light", "state", light_state],
            [timestamp.isoformat(), hour, "home/livingroom/light", "brightness", random.randint(20, 100)],
            [timestamp.isoformat(), hour, "home/bedroom/thermostat", "temp_c", temp],
            [timestamp.isoformat(), hour, "home/bedroom/thermostat", "mode", mode],
            [timestamp.isoformat(), hour, "home/frontdoor/motion", "detected", motion],
        ])

df = pd.DataFrame(rows, columns=["timestamp", "hour", "topic", "key", "value"])
df.to_csv("data/events.csv", index=False)
print(f"✅ Generated {len(df)} rows of realistic data across 7 days")
print(df.head(10))