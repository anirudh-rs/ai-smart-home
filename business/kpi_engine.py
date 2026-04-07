import pandas as pd
import pickle
import json
from datetime import datetime

# --- Energy cost assumptions (adjustable) ---
COST_PER_KWH = 0.13          # Average US electricity cost
LIGHT_WATTS = 60              # Watts when light is ON
THERMOSTAT_WASTE_WATTS = 500  # Extra watts when thermostat is in wrong zone
HOURS_IN_YEAR = 8760

def load_data():
    df = pd.read_csv("data/events.csv")
    df["hour"] = df["hour"].astype(int)
    return df

def load_rules():
    with open("rules/generated_rules.json") as f:
        return json.load(f)

# --- KPI 1: Energy Efficiency Score ---
def calc_efficiency_score(df, rules):
    """
    Compares actual device states vs AI recommended states.
    Returns % of time devices were in the optimal state.
    """
    light_rules = {r["hour"]: r["value"] for r in rules if "light" in r["device"]}
    
    light_df = df[
        (df["topic"] == "home/livingroom/light") & 
        (df["key"] == "state")
    ].copy()
    
    if len(light_df) == 0:
        return 0.0
    
    light_df["recommended"] = light_df["hour"].map(light_rules)
    light_df["is_optimal"] = light_df["value"] == light_df["recommended"]
    
    score = light_df["is_optimal"].mean() * 100
    return round(score, 1)

# --- KPI 2: Estimated Energy Savings ---
def calc_energy_savings(df, rules):
    """
    Calculates $ savings if AI rules were followed vs random usage.
    """
    light_rules = {r["hour"]: r["value"] for r in rules if "light" in r["device"]}
    
    light_df = df[
        (df["topic"] == "home/livingroom/light") & 
        (df["key"] == "state")
    ].copy()
    
    if len(light_df) == 0:
        return 0.0, 0.0
    
    # Hours light was ON in actual data
    actual_on_hours = (light_df["value"] == "ON").sum()
    
    # Hours light should be ON per AI rules
    optimal_on_hours = sum(1 for v in light_rules.values() if v == "ON")
    
    # Scale to a full year
    days_in_data = max(light_df["hour"].nunique() / 24, 1)
    scale_factor = 365 / max(days_in_data, 1)
    
    actual_kwh_year = (actual_on_hours * scale_factor * LIGHT_WATTS) / 1000
    optimal_kwh_year = (optimal_on_hours * 365 * LIGHT_WATTS) / 1000
    
    actual_cost = actual_kwh_year * COST_PER_KWH
    optimal_cost = optimal_kwh_year * COST_PER_KWH

    savings = max(actual_cost - optimal_cost, 0)
    return round(actual_cost, 2), round(savings, 2)

# --- KPI 3: Anomaly Count ---
def calc_anomaly_count(df, rules):
    """
    Counts hours where device state deviated from AI recommendation.
    """
    light_rules = {r["hour"]: r["value"] for r in rules if "light" in r["device"]}
    
    light_df = df[
        (df["topic"] == "home/livingroom/light") & 
        (df["key"] == "state")
    ].copy()
    
    if len(light_df) == 0:
        return 0
    
    light_df["recommended"] = light_df["hour"].map(light_rules)
    anomalies = (light_df["value"] != light_df["recommended"]).sum()
    return int(anomalies)

# --- KPI 4: AI Rule Compliance Rate ---
def calc_compliance_rate(df, rules):
    """
    % of events that already match AI recommendations.
    """
    light_rules = {r["hour"]: r["value"] for r in rules if "light" in r["device"]}
    
    light_df = df[
        (df["topic"] == "home/livingroom/light") & 
        (df["key"] == "state")
    ].copy()
    
    if len(light_df) == 0:
        return 0.0
    
    light_df["recommended"] = light_df["hour"].map(light_rules)
    compliance = (light_df["value"] == light_df["recommended"]).mean() * 100
    return round(compliance, 1)

# --- KPI 5: Peak Usage Hours ---
def calc_peak_hours(df):
    """
    Returns the top 3 hours with highest device activity.
    """
    light_df = df[
        (df["topic"] == "home/livingroom/light") & 
        (df["key"] == "state") &
        (df["value"] == "ON")
    ]
    
    peak = light_df.groupby("hour").size().sort_values(ascending=False).head(3)
    return peak.index.tolist()

# --- Master KPI Report ---
def generate_kpi_report():
    df = load_data()
    rules = load_rules()
    
    efficiency = calc_efficiency_score(df, rules)
    actual_cost, savings = calc_energy_savings(df, rules)
    anomalies = calc_anomaly_count(df, rules)
    compliance = calc_compliance_rate(df, rules)
    peak_hours = calc_peak_hours(df)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "efficiency_score": efficiency,
        "actual_annual_cost_usd": actual_cost,
        "estimated_annual_savings_usd": savings,
        "anomaly_count": anomalies,
        "compliance_rate": compliance,
        "peak_usage_hours": peak_hours,
        "recommendation": generate_recommendation(efficiency, savings, anomalies)
    }
    
    return report

def generate_recommendation(efficiency, savings, anomalies):
    """Generates a plain English business recommendation."""
    if efficiency >= 80:
        eff_msg = "Device usage is well optimized."
    elif efficiency >= 60:
        eff_msg = "Device usage is moderately optimized with room for improvement."
    else:
        eff_msg = "Device usage is poorly optimized — significant savings possible."
    
    savings_msg = f"Following AI rules could save an estimated ${savings:.2f} annually."
    
    if anomalies > 20:
        anomaly_msg = f"High anomaly count ({anomalies}) suggests irregular usage patterns worth investigating."
    else:
        anomaly_msg = f"Anomaly count ({anomalies}) is within acceptable range."
    
    return f"{eff_msg} {savings_msg} {anomaly_msg}"

if __name__ == "__main__":
    print("📊 Generating KPI Report...\n")
    report = generate_kpi_report()
    
    print("=" * 50)
    print("       🏠 SMART HOME KPI REPORT")
    print("=" * 50)
    print(f"⚡ Energy Efficiency Score:     {report['efficiency_score']}%")
    print(f"💸 Estimated Annual Cost:       ${report['actual_annual_cost_usd']}")
    print(f"💰 Estimated Annual Savings:    ${report['estimated_annual_savings_usd']}")
    print(f"🚨 Anomalies Detected:          {report['anomaly_count']}")
    print(f"✅ AI Rule Compliance Rate:     {report['compliance_rate']}%")
    print(f"🕐 Peak Usage Hours:            {report['peak_usage_hours']}")
    print(f"\n📝 Recommendation:")
    print(f"   {report['recommendation']}")
    print("=" * 50)