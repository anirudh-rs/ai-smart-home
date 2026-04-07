import streamlit as st
import pandas as pd
import json
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from business.kpi_engine import generate_kpi_report
from ml.anomaly_model import load_data, prepare_features, train_anomaly_model
from ml.forecasting import load_appliance_data, train_forecast_model, generate_forecast, calc_forecast_kpis

st.set_page_config(
    page_title="AI Smart Home — Executive Dashboard",
    page_icon="🏠",
    layout="wide"
)

# --- Header ---
st.title("🏠 AI Smart Home — Executive Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%A, %B %d %Y at %H:%M')}")
st.divider()

# --- Load everything ---
@st.cache_data(show_spinner=False)
def load_rules():
    with open("rules/generated_rules.json") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_kpis():
    return generate_kpi_report()

@st.cache_data(show_spinner=False)
def get_anomalies():
    df = load_data()
    features = prepare_features(df)
    _, results = train_anomaly_model(features)
    return results

@st.cache_data(show_spinner=False)
def get_forecast():
    daily_df = load_appliance_data()
    model, prophet_df = train_forecast_model(daily_df)
    forecast = generate_forecast(model, prophet_df, days=14)
    kpis = calc_forecast_kpis(forecast, days=14)
    return forecast, kpis

rules = load_rules()
kpis = get_kpis()
anomaly_df = get_anomalies()
df = pd.read_csv("data/events.csv")

# ── Section 1: KPI Cards ──────────────────────────────
st.subheader("📊 Business KPIs")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⚡ Efficiency Score", f"{kpis['efficiency_score']}%",
            delta="vs 50% baseline")
col2.metric("💰 Est. Annual Savings", f"${kpis['estimated_annual_savings_usd']}")
col3.metric("💸 Est. Annual Cost", f"${kpis['actual_annual_cost_usd']}")
col4.metric("🚨 Anomalies Detected",
            int(anomaly_df[anomaly_df["is_anomaly"]]["is_anomaly"].sum()))
col5.metric("✅ AI Compliance Rate", f"{kpis['compliance_rate']}%")

st.divider()

# ── Section 2: Recommendation Box ────────────────────
st.subheader("📝 AI Recommendation")
st.info(f"💡 {kpis['recommendation']}")
st.divider()

# ── Section 3: Device Schedule Charts ────────────────
st.subheader("🕐 24-Hour Automation Schedule")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**💡 Living Room Light Energy by Hour**")

    # Use raw UCI brightness data instead of ON/OFF rules
    raw_light = df[
        (df["topic"] == "home/livingroom/light") &
        (df["key"] == "brightness")
    ].copy()
    raw_light["value"] = pd.to_numeric(raw_light["value"], errors="coerce")
    hourly_light = raw_light.groupby("hour")["value"].mean().reset_index()
    hourly_light.columns = ["hour", "avg_wh"]

    fig_light = go.Figure()
    fig_light.add_trace(go.Bar(
        x=hourly_light["hour"],
        y=hourly_light["avg_wh"],
        marker_color=[
            "gold" if v > 10 else "#e0e0e0"
            for v in hourly_light["avg_wh"]
        ],
        text=hourly_light["avg_wh"].round(1),
        textposition="auto"
    ))
    fig_light.update_layout(
        xaxis_title="Hour",
        yaxis_title="Avg Energy (Wh)",
        height=280,
        showlegend=False,
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_light, use_container_width=True)
    st.caption("💡 Showing actual average light energy (Wh) per hour from UCI dataset. "
               "Gold bars indicate active usage (>10Wh). Note: UCI lights are low-wattage LEDs — "
               "the habit model correctly predicts OFF for all hours, so raw energy "
               "consumption is visualized here instead for better insight.")

with col_right:
    st.markdown("**🌡️ Thermostat Comfort Zone**")
    thermo_rules = [r for r in rules if "thermostat" in r["device"]]
    thermo_df = pd.DataFrame(thermo_rules)
    zone_colors = {
        "cold": "#90caf9",
        "cool": "#64b5f6",
        "comfortable": "#a5d6a7",
        "warm": "#ffb74d"
    }
    fig_thermo = go.Figure()
    fig_thermo.add_trace(go.Bar(
        x=thermo_df["hour"],
        y=[1] * len(thermo_df),
        marker_color=[zone_colors.get(v, "gray") for v in thermo_df["value"]],
        text=thermo_df["value"],
        textposition="auto"
    ))
    fig_thermo.update_layout(
        xaxis_title="Hour",
        yaxis=dict(visible=False),
        height=280,
        showlegend=False,
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_thermo, use_container_width=True)

st.divider()

# ── Section 4: Anomaly Analysis ───────────────────────
st.subheader("🚨 Anomaly Analysis")

anomalies_only = anomaly_df[anomaly_df["is_anomaly"] == True].copy()
normal_only = anomaly_df[anomaly_df["is_anomaly"] == False].copy()

col_a, col_b = st.columns(2)

with col_a:
    anomaly_by_hour = anomalies_only.groupby("hour").size().reset_index(name="count")
    fig_anom = px.bar(
        anomaly_by_hour, x="hour", y="count",
        title="Anomalies by Hour of Day",
        color="count",
        color_continuous_scale="Reds"
    )
    fig_anom.update_layout(height=300, margin=dict(t=40, b=40))
    st.plotly_chart(fig_anom, use_container_width=True)

with col_b:
    fig_pie = go.Figure(go.Pie(
        labels=["Normal", "Anomaly"],
        values=[len(normal_only), len(anomalies_only)],
        marker_colors=["#a5d6a7", "#ef9a9a"],
        hole=0.4
    ))
    fig_pie.update_layout(
        title="Normal vs Anomalous Events",
        height=300,
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("🔍 View Anomalous Events Detail"):
    display_df = anomalies_only[["hour", "light_on", "temp_c", "motion", "anomaly_score"]].copy()
    display_df["light"] = display_df["light_on"].map({1: "ON", 0: "OFF"})
    display_df["motion"] = display_df["motion"].map({1: "YES", 0: "NO"})
    display_df = display_df[["hour", "light", "temp_c", "motion", "anomaly_score"]]
    display_df = display_df.sort_values("anomaly_score").head(20)
    st.dataframe(display_df, use_container_width=True)

st.divider()

# ── Section 5: Time Series Forecast ──────────────────
st.subheader("📈 14-Day Energy Consumption Forecast")

with st.spinner("Running forecast model..."):
    forecast, forecast_kpis = get_forecast()

col_f1, col_f2, col_f3, col_f4 = st.columns(4)
col_f1.metric("📅 Forecast Days", forecast_kpis["forecast_days"])
col_f2.metric("⚡ Avg Daily", f"{forecast_kpis['avg_daily_kwh']} kWh")
col_f3.metric("💰 Total Cost", f"${forecast_kpis['total_forecast_cost_usd']}")
col_f4.metric("📆 Annual Projection", f"${forecast_kpis['projected_annual_cost_usd']}")

future_forecast = forecast.tail(14)
historical = forecast.head(len(forecast) - 14)

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=historical["ds"],
    y=historical["yhat"] / 1000,
    name="Historical (fitted)",
    line=dict(color="#64b5f6", width=1.5)
))

fig_forecast.add_trace(go.Scatter(
    x=future_forecast["ds"],
    y=future_forecast["yhat"] / 1000,
    name="Forecast",
    line=dict(color="#ff7043", width=2.5, dash="dash")
))

fig_forecast.add_trace(go.Scatter(
    x=pd.concat([future_forecast["ds"], future_forecast["ds"].iloc[::-1]]),
    y=pd.concat([future_forecast["yhat_upper"] / 1000,
                 future_forecast["yhat_lower"].iloc[::-1] / 1000]),
    fill="toself",
    fillcolor="rgba(255,112,67,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="95% Confidence Interval"
))

fig_forecast.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy (kWh)",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=40, b=40)
)

st.plotly_chart(fig_forecast, use_container_width=True)

with st.expander("📋 View Daily Forecast Details"):
    COST_PER_KWH = 0.13
    forecast_table = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_table["date"] = forecast_table["ds"].dt.strftime("%Y-%m-%d (%a)")
    forecast_table["kwh"] = (forecast_table["yhat"] / 1000).round(2)
    forecast_table["cost_usd"] = ((forecast_table["yhat"] / 1000) * COST_PER_KWH).round(2)
    forecast_table["range_kwh"] = (
        (forecast_table["yhat_lower"] / 1000).round(1).astype(str) +
        " → " +
        (forecast_table["yhat_upper"] / 1000).round(1).astype(str)
    )
    st.dataframe(
        forecast_table[["date", "kwh", "cost_usd", "range_kwh"]],
        use_container_width=True
    )

st.divider()

# ── Section 6: AI Predictor ───────────────────────────
st.subheader("🔮 AI Predictor")

col_pred, col_result = st.columns([1, 2])

with col_pred:
    selected_hour = st.slider("Select Hour", 0, 23, datetime.now().hour)
    predict_btn = st.button("▶ Run Prediction", type="primary")

with col_result:
    if predict_btn:
        try:
            light_model = pickle.load(open("ml/models/light_model.pkl", "rb"))
            light_le = pickle.load(open("ml/models/light_le.pkl", "rb"))
            light_pred = light_le.inverse_transform(
                light_model.predict(
                    pd.DataFrame([[selected_hour]], columns=["hour"])
                )
            )[0]

            thermo_model = pickle.load(open("ml/models/thermo_model.pkl", "rb"))
            thermo_le = pickle.load(open("ml/models/thermo_le.pkl", "rb"))
            thermo_pred = thermo_le.inverse_transform(
                thermo_model.predict(
                    pd.DataFrame([[selected_hour]], columns=["hour"])
                )
            )[0]

            anomaly_model = pickle.load(open("ml/models/anomaly_model.pkl", "rb"))
            hour_features = pd.DataFrame(
                [[selected_hour, 1 if light_pred == "ON" else 0, 21.0, 0]],
                columns=["hour", "light_on", "temp_c", "motion"]
            )
            anomaly_result = anomaly_model.predict(hour_features)[0]
            anomaly_label = (
                "⚠️ Unusual pattern expected at this hour"
                if anomaly_result == -1
                else "✅ Normal pattern expected at this hour"
            )

            st.success(f"💡 Light recommendation: **{light_pred}**")
            st.info(f"🌡️ Thermostat zone: **{thermo_pred}**")
            if anomaly_result == -1:
                st.warning(anomaly_label)
            else:
                st.success(anomaly_label)
            st.caption("ℹ️ Predictions trained on UCI Belgian household data (Jan–May 2016). "
                       "Consistent indoor climate and LED lighting result in stable "
                       "predictions. Models would show more variation with multi-year "
                       "or multi-season data.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.divider()

# ── Section 7: Raw Data ───────────────────────────────
with st.expander("🗃️ View Raw Event Data"):
    st.dataframe(df, use_container_width=True)