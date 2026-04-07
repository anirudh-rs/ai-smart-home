import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_data():
    df = pd.read_csv("data/events.csv")
    return df

def train_light_model(df):
    # Filter only light state events
    light_df = df[(df["topic"] == "home/livingroom/light") & (df["key"] == "state")].copy()
    
    if len(light_df) < 10:
        print("⚠️ Not enough light data yet — run simulator longer next time")
        return None, None
    
    # Features: hour of day
    X = light_df[["hour"]].astype(int)
    y = light_df["value"]
    
    # Encode ON/OFF to 1/0
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"💡 Light model trained — Accuracy: {accuracy:.0%}")
    
    return model, le

def train_thermostat_model(df):
    # Filter thermostat temp events
    temp_df = df[(df["topic"] == "home/bedroom/thermostat") & (df["key"] == "temp_c")].copy()
    
    if len(temp_df) < 10:
        print("⚠️ Not enough thermostat data yet")
        return None
    
    X = temp_df[["hour"]].astype(int)
    y = temp_df["value"].astype(float)
    
    # Bin temperatures into comfort zones
    y_binned = pd.cut(y, bins=[0, 19, 21, 23, 100], labels=["cold", "cool", "comfortable", "warm"])
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binned)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"🌡️ Thermostat model trained — Accuracy: {accuracy:.0%}")
    
    return model, le

def save_models(light_model, light_le, thermo_model, thermo_le):
    os.makedirs("ml/models", exist_ok=True)
    
    if light_model:
        pickle.dump(light_model, open("ml/models/light_model.pkl", "wb"))
        pickle.dump(light_le, open("ml/models/light_le.pkl", "wb"))
        print("💾 Light model saved")
    
    if thermo_model:
        pickle.dump(thermo_model, open("ml/models/thermo_model.pkl", "wb"))
        pickle.dump(thermo_le, open("ml/models/thermo_le.pkl", "wb"))
        print("💾 Thermostat model saved")

def predict(hour):
    print(f"\n🤖 AI Predictions for hour {hour}:00")
    
    try:
        light_model = pickle.load(open("ml/models/light_model.pkl", "rb"))
        light_le = pickle.load(open("ml/models/light_le.pkl", "rb"))
        light_pred = light_le.inverse_transform(light_model.predict(pd.DataFrame([[hour]], columns=["hour"])))[0]
        print(f"  💡 Living room light should be: {light_pred}")
    except:
        print("  💡 Light model not found — train first")
    
    try:
        thermo_model = pickle.load(open("ml/models/thermo_model.pkl", "rb"))
        thermo_le = pickle.load(open("ml/models/thermo_le.pkl", "rb"))
        thermo_pred = thermo_le.inverse_transform(thermo_model.predict(pd.DataFrame([[hour]], columns=["hour"])))[0]
        print(f"  🌡️ Thermostat comfort zone: {thermo_pred}")
    except:
        print("  🌡️ Thermostat model not found — train first")

if __name__ == "__main__":
    print("📊 Loading data...")
    df = load_data()
    print(f"   {len(df)} events loaded")
    
    print("\n🏋️ Training models...")
    light_model, light_le = train_light_model(df)
    thermo_model, thermo_le = train_thermostat_model(df)
    
    save_models(light_model, light_le, thermo_model, thermo_le)
    
    # Test predictions for a few hours
    print("\n🔮 Sample predictions:")
    for hour in [8, 12, 18, 22]:
        predict(hour)