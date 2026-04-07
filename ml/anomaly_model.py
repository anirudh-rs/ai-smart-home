import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv("data/events.csv")
    df["hour"] = df["hour"].astype(int)
    return df

def prepare_features(df):
    """
    Build a feature matrix from device events.
    Each row = one hour snapshot of all device states.
    """
    # Pivot light states by hour
    light_df = df[
        (df["topic"] == "home/livingroom/light") &
        (df["key"] == "state")
    ].copy()

    # Encode ON/OFF as 1/0
    light_df["light_on"] = (light_df["value"] == "ON").astype(int)

    # Pivot thermostat temp by hour
    temp_df = df[
        (df["topic"] == "home/bedroom/thermostat") &
        (df["key"] == "temp_c")
    ].copy()
    temp_df["temp_c"] = temp_df["value"].astype(float)

    # Pivot motion by hour
    motion_df = df[
        (df["topic"] == "home/frontdoor/motion") &
        (df["key"] == "detected")
    ].copy()
    motion_df["motion"] = (motion_df["value"] == "True").astype(int)

    # Merge all features on hour
    features = light_df[["hour", "light_on"]].copy()
    features = features.merge(
        temp_df[["hour", "temp_c"]], on="hour", how="left"
    )
    features = features.merge(
        motion_df[["hour", "motion"]], on="hour", how="left"
    )

    features = features.fillna(0)
    return features

def train_anomaly_model(features):
    """
    Train Isolation Forest — flags rows that don't fit normal patterns.
    contamination=0.05 means we expect ~5% of data to be anomalous.
    """
    # Sample for speed — 5000 rows is enough for Isolation Forest
    if len(features) > 5000:
        features = features.sample(n=5000, random_state=42)
    
    X = features[["hour", "light_on", "temp_c", "motion"]]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    # -1 = anomaly, 1 = normal
    predictions = model.predict(X)
    scores = model.decision_function(X)

    features = features.copy()
    features["anomaly"] = predictions
    features["anomaly_score"] = scores
    features["is_anomaly"] = features["anomaly"] == -1

    return model, features

def save_anomaly_model(model):
    os.makedirs("ml/models", exist_ok=True)
    pickle.dump(model, open("ml/models/anomaly_model.pkl", "wb"))
    print("💾 Anomaly model saved")

def print_anomaly_report(features):
    anomalies = features[features["is_anomaly"] == True].copy()
    normal = features[features["is_anomaly"] == False].copy()

    print("\n" + "=" * 50)
    print("       🚨 ANOMALY DETECTION REPORT")
    print("=" * 50)
    print(f"Total events analyzed:    {len(features)}")
    print(f"Normal events:            {len(normal)}")
    print(f"Anomalies detected:       {len(anomalies)}")
    print(f"Anomaly rate:             {len(anomalies)/len(features)*100:.1f}%")

    if len(anomalies) > 0:
        print(f"\n🔍 Anomalous Hours Detected:")
        print("-" * 50)
        for _, row in anomalies.iterrows():
            light = "ON" if row["light_on"] == 1 else "OFF"
            motion = "YES" if row["motion"] == 1 else "NO"
            print(f"  Hour {int(row['hour']):02d}:00 → "
                  f"Light: {light} | "
                  f"Temp: {row['temp_c']:.1f}°C | "
                  f"Motion: {motion} | "
                  f"Score: {row['anomaly_score']:.3f}")

    print("\n💡 What this means:")
    print("   Negative anomaly scores = further from normal pattern")
    print("   These hours had unusual combinations of device states")
    print("=" * 50)

if __name__ == "__main__":
    print("🔍 Loading data...")
    df = load_data()
    print(f"   {len(df)} events loaded")

    print("\n⚙️ Preparing features...")
    features = prepare_features(df)
    print(f"   {len(features)} hour snapshots created")

    print("\n🧠 Training Isolation Forest anomaly model...")
    model, results = train_anomaly_model(features)

    print_anomaly_report(results)
    save_anomaly_model(model)