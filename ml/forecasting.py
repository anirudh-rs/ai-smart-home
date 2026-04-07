import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_appliance_data():
    """
    Load and aggregate UCI appliance energy data for forecasting.
    We'll forecast daily energy consumption.
    """
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    # Aggregate to daily total energy consumption
    daily = df.groupby(df["date"].dt.date).agg({
        "Appliances": "sum",  # Total daily appliance energy (Wh)
        "lights": "sum",      # Total daily light energy (Wh)
        "T1": "mean",         # Average daily temp
        "RH_1": "mean"        # Average daily humidity
    }).reset_index()
    
    daily.columns = ["date", "appliances_wh", "lights_wh", "avg_temp", "avg_humidity"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["total_energy_wh"] = daily["appliances_wh"] + daily["lights_wh"]
    
    return daily

def train_forecast_model(daily_df):
    """
    Train Prophet model to forecast total energy consumption.
    Prophet expects columns: ds (date) and y (value to forecast)
    """
    # Prepare Prophet format
    prophet_df = daily_df[["date", "total_energy_wh"]].copy()
    prophet_df.columns = ["ds", "y"]
    
    # Add regressors for better accuracy
    prophet_df["avg_temp"] = daily_df["avg_temp"].values
    prophet_df["avg_humidity"] = daily_df["avg_humidity"].values
    
    # Initialize and configure Prophet
    model = Prophet(
        yearly_seasonality=False,  # Not enough data for yearly
        weekly_seasonality=True,   # Day of week patterns
        daily_seasonality=False,   # Already aggregated to daily
        changepoint_prior_scale=0.05,  # Flexibility of trend
        interval_width=0.95        # 95% confidence intervals
    )
    
    # Add external regressors
    model.add_regressor("avg_temp")
    model.add_regressor("avg_humidity")
    
    model.fit(prophet_df)
    
    return model, prophet_df

def generate_forecast(model, prophet_df, days=14):
    """
    Generate forecast for next N days.
    Uses average temp/humidity from last 7 days as future values.
    """
    # Create future dataframe
    future = model.make_future_dataframe(periods=days)
    
    # Fill regressors for future dates using recent averages
    recent_temp = prophet_df["avg_temp"].tail(7).mean()
    recent_humidity = prophet_df["avg_humidity"].tail(7).mean()
    
    future["avg_temp"] = prophet_df["avg_temp"].reindex(
        future.index, fill_value=recent_temp
    ).values
    future["avg_humidity"] = prophet_df["avg_humidity"].reindex(
        future.index, fill_value=recent_humidity
    ).values
    
    # Fill any NaN values in future regressors
    future["avg_temp"] = future["avg_temp"].fillna(recent_temp)
    future["avg_humidity"] = future["avg_humidity"].fillna(recent_humidity)
    
    forecast = model.predict(future)
    return forecast

def calc_forecast_kpis(forecast, days=14):
    """
    Calculate business KPIs from the forecast.
    """
    COST_PER_KWH = 0.13
    
    future_forecast = forecast.tail(days)
    
    avg_daily_wh = future_forecast["yhat"].mean()
    avg_daily_kwh = avg_daily_wh / 1000
    avg_daily_cost = avg_daily_kwh * COST_PER_KWH
    
    total_forecast_wh = future_forecast["yhat"].sum()
    total_forecast_cost = (total_forecast_wh / 1000) * COST_PER_KWH
    
    # Upper and lower bounds
    upper_cost = (future_forecast["yhat_upper"].sum() / 1000) * COST_PER_KWH
    lower_cost = (future_forecast["yhat_lower"].sum() / 1000) * COST_PER_KWH
    
    return {
        "forecast_days": days,
        "avg_daily_kwh": round(avg_daily_kwh, 2),
        "avg_daily_cost_usd": round(avg_daily_cost, 2),
        "total_forecast_kwh": round(total_forecast_wh / 1000, 2),
        "total_forecast_cost_usd": round(total_forecast_cost, 2),
        "cost_lower_bound_usd": round(lower_cost, 2),
        "cost_upper_bound_usd": round(upper_cost, 2),
        "projected_annual_cost_usd": round(avg_daily_cost * 365, 2)
    }

def save_forecast(forecast, model):
    os.makedirs("ml/models", exist_ok=True)
    forecast.to_csv("data/forecast.csv", index=False)
    pickle.dump(model, open("ml/models/forecast_model.pkl", "wb"))
    print("💾 Forecast saved to data/forecast.csv")
    print("💾 Forecast model saved to ml/models/forecast_model.pkl")

def print_forecast_report(kpis, forecast, days=14):
    future_forecast = forecast.tail(days)
    
    print("\n" + "=" * 55)
    print("       📈 ENERGY CONSUMPTION FORECAST REPORT")
    print("=" * 55)
    print(f"Forecast period:          Next {kpis['forecast_days']} days")
    print(f"Avg daily consumption:    {kpis['avg_daily_kwh']} kWh")
    print(f"Avg daily cost:           ${kpis['avg_daily_cost_usd']}")
    print(f"Total forecasted cost:    ${kpis['total_forecast_cost_usd']}")
    print(f"Cost range (95% CI):      ${kpis['cost_lower_bound_usd']} → ${kpis['cost_upper_bound_usd']}")
    print(f"Projected annual cost:    ${kpis['projected_annual_cost_usd']}")
    print(f"\n📅 Daily Forecast:")
    print("-" * 55)
    
    COST_PER_KWH = 0.13
    for _, row in future_forecast.iterrows():
        date = row["ds"].strftime("%Y-%m-%d (%a)")
        kwh = round(row["yhat"] / 1000, 2)
        cost = round(kwh * COST_PER_KWH, 2)
        lower = round(row["yhat_lower"] / 1000, 2)
        upper = round(row["yhat_upper"] / 1000, 2)
        print(f"  {date}: {kwh} kWh (${cost}) [{lower}-{upper} kWh]")
    
    print("=" * 55)

if __name__ == "__main__":
    print("📂 Loading appliance data...")
    daily_df = load_appliance_data()
    print(f"   {len(daily_df)} days of data loaded")
    print(f"   Date range: {daily_df['date'].min().date()} → {daily_df['date'].max().date()}")
    print(f"   Avg daily consumption: {daily_df['total_energy_wh'].mean():.0f} Wh")

    print("\n🧠 Training Prophet forecasting model...")
    model, prophet_df = train_forecast_model(daily_df)
    print("   Model trained!")

    print("\n🔮 Generating 14-day forecast...")
    forecast = generate_forecast(model, prophet_df, days=14)

    kpis = calc_forecast_kpis(forecast, days=14)
    print_forecast_report(kpis, forecast, days=14)

    save_forecast(forecast, model)