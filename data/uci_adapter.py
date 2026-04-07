import pandas as pd
import os

def load_uci_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def convert_to_events(df):
    """
    Converts UCI dataset columns into your existing events.csv format:
    timestamp, hour, topic, key, value
    """
    rows = []

    for _, row in df.iterrows():
        timestamp = row["date"].isoformat()
        hour = row["date"].hour

        # Light state — UCI measures in Wh, >10 means lights are ON
        light_state = "ON" if row["lights"] > 10 else "OFF"
        rows.append([timestamp, hour, "home/livingroom/light", "state", light_state])
        rows.append([timestamp, hour, "home/livingroom/light", "brightness", round(row["lights"], 1)])

        # Thermostat — T1 is kitchen temp (closest to living area)
        rows.append([timestamp, hour, "home/bedroom/thermostat", "temp_c", round(row["T1"], 1)])

        # Humidity
        rows.append([timestamp, hour, "home/livingroom/humidity", "humidity_pct", round(row["RH_1"], 1)])

        # Appliance energy usage
        rows.append([timestamp, hour, "home/appliances/energy", "usage_wh", round(row["Appliances"], 1)])

        # Outside temperature
        rows.append([timestamp, hour, "home/outside/weather", "temp_out_c", round(row["T_out"], 1)])

    events_df = pd.DataFrame(rows, columns=["timestamp", "hour", "topic", "key", "value"])
    return events_df

def save_events(events_df):
    # Back up old data first
    if os.path.exists("data/events.csv"):
        backup_path = "data/events_backup.csv"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename("data/events.csv", backup_path)
        print("📦 Old data backed up to events_backup.csv")

    events_df.to_csv("data/events.csv", index=False)
    print(f"✅ Saved {len(events_df):,} real events to data/events.csv")

def print_summary(df, events_df):
    print("\n" + "=" * 50)
    print("       📊 UCI DATASET SUMMARY")
    print("=" * 50)
    print(f"Date range:        {df['date'].min()} → {df['date'].max()}")
    print(f"Total readings:    {len(df):,} (every 10 minutes)")
    print(f"Total events:      {len(events_df):,}")
    print(f"Unique hours:      {events_df['hour'].nunique()}")
    print(f"Topics created:    {events_df['topic'].nunique()}")
    print(f"\nTopics:")
    for t in events_df['topic'].unique():
        count = len(events_df[events_df['topic'] == t])
        print(f"  {t}: {count:,} events")
    print("=" * 50)

if __name__ == "__main__":
    print("📂 Loading UCI dataset...")
    df = load_uci_data()
    print(f"   {len(df):,} rows loaded")

    print("\n⚙️ Converting to events format...")
    events_df = convert_to_events(df)

    print_summary(df, events_df)
    save_events(events_df)