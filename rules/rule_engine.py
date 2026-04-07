import pickle
import pandas as pd
import json
import os

def generate_rules():
    rules = []

    try:
        light_model = pickle.load(open("ml/models/light_model.pkl", "rb"))
        light_le = pickle.load(open("ml/models/light_le.pkl", "rb"))

        for hour in range(24):
            pred = light_le.inverse_transform(
                light_model.predict(pd.DataFrame([[hour]], columns=["hour"]))
            )[0]
            rules.append({
                "hour": hour,
                "device": "home/livingroom/light",
                "action": "state",
                "value": pred,
                "label": f"At {hour:02d}:00 → Living room light: {pred}"
            })
    except Exception as e:
        print(f"⚠️ Could not load light model: {e}")

    try:
        thermo_model = pickle.load(open("ml/models/thermo_model.pkl", "rb"))
        thermo_le = pickle.load(open("ml/models/thermo_le.pkl", "rb"))

        for hour in range(24):
            pred = thermo_le.inverse_transform(
                thermo_model.predict(pd.DataFrame([[hour]], columns=["hour"]))
            )[0]
            rules.append({
                "hour": hour,
                "device": "home/bedroom/thermostat",
                "action": "comfort_zone",
                "value": pred,
                "label": f"At {hour:02d}:00 → Thermostat comfort zone: {pred}"
            })
    except Exception as e:
        print(f"⚠️ Could not load thermostat model: {e}")

    return rules

def save_rules(rules):
    os.makedirs("rules", exist_ok=True)
    with open("rules/generated_rules.json", "w") as f:
        json.dump(rules, f, indent=2)
    print(f"💾 {len(rules)} rules saved to rules/generated_rules.json")

def print_rules(rules):
    print("\n📋 Generated Automation Rules:")
    print("=" * 45)
    
    light_rules = [r for r in rules if "light" in r["device"]]
    thermo_rules = [r for r in rules if "thermostat" in r["device"]]
    
    print("\n💡 Living Room Light:")
    for r in light_rules:
        print(f"  {r['label']}")
    
    print("\n🌡️ Bedroom Thermostat:")
    for r in thermo_rules:
        print(f"  {r['label']}")

if __name__ == "__main__":
    print("⚙️ Generating rules from trained models...")
    rules = generate_rules()
    print_rules(rules)
    save_rules(rules)