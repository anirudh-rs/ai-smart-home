# 🏠 AI Smart Home Analytics Platform

> An end-to-end IoT analytics platform that processes real household sensor data,
> learns device usage habits using machine learning, detects anomalous behaviour,
> forecasts energy consumption, and delivers actionable insights through an
> executive-grade Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?style=flat-square)](https://share.streamlit.io)
[![GitHub](https://img.shields.io/badge/GitHub-anirudh--rs%2Fai--smart--home-181717?style=flat-square&logo=github)](https://github.com/anirudh-rs/ai-smart-home)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Appliances%20Energy-orange?style=flat-square)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

---

## 📌 Overview

This project solves a real-world data problem: smart homes generate thousands of sensor
events every day, but most of that data is never analysed. Without intelligent processing,
energy is wasted through suboptimal device scheduling, anomalies go undetected, and
no actionable insights reach the people who could act on them.

This platform addresses all three problems using a full ML pipeline built on top of
**118,410 real IoT sensor events** from the UCI Appliances Energy Prediction Dataset —
a real Belgian household monitored over 138 days.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset | UCI Appliances Energy Prediction (real household) |
| Total Events Processed | 118,410 |
| Data Range | Jan 11 – May 27, 2016 (138 days) |
| Data Frequency | Every 10 minutes (19,735 raw readings) |
| Light Habit Model Accuracy | **88%** (Random Forest) |
| Energy Efficiency Score | **88.5%** |
| AI Rule Compliance Rate | **88.5%** |
| Anomaly Detection Rate | **5.0%** (Isolation Forest) |
| Anomalies Flagged | 250 from 5,000-row sample |
| Annual Energy Cost Baseline | $6,465 |
| 14-Day Forecast Cost | $26.59 |
| Projected Annual Cost (forecast) | $693 |
| Peak Device Usage Hours | 20:00, 21:00, 22:00 |

---

## 🏗️ Architecture

```
UCI Dataset (19,735 readings @ 10-min intervals)
              ↓
    data/uci_adapter.py
    Maps UCI columns → events format
              ↓
    data/events.csv
    118,410 rows · 5 device topics · 24 hours
              ↓
    ┌─────────────────────────────────────────┐
    │              ML Pipeline                │
    │                                         │
    │  ml/habit_model.py                      │
    │  Random Forest · 88% accuracy           │
    │  Features: hour of day                  │
    │  Targets: light state, thermostat zone  │
    │               ↓                         │
    │  rules/rule_engine.py                   │
    │  48 automation rules · JSON output      │
    │               ↓                         │
    │  ml/anomaly_model.py                    │
    │  Isolation Forest · 5% anomaly rate     │
    │  Features: hour, light, temp, motion    │
    │               ↓                         │
    │  ml/forecasting.py                      │
    │  Prophet · 14-day forecast              │
    │  Weekly seasonality · 95% CI            │
    └─────────────────────────────────────────┘
              ↓
    business/kpi_engine.py
    Efficiency · Savings · Anomaly count · ROI
              ↓
    dashboard/app.py
    Executive Streamlit Dashboard
    7 sections · Live AI predictor
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10 | Core development |
| Environment | Anaconda / conda | Package management |
| ML — Supervised | scikit-learn RandomForestClassifier | Device habit learning |
| ML — Unsupervised | scikit-learn IsolationForest | Anomaly detection |
| ML — Time Series | Prophet (Meta) | Energy forecasting |
| IoT Protocol | MQTT (paho-mqtt + Mosquitto) | Device communication |
| Dashboard | Streamlit + Plotly | Executive visualisation |
| Data | UCI Appliances Energy Dataset | Real household sensor data |
| Deployment | Streamlit Community Cloud | Live public hosting |
| Version Control | Git + GitHub | Source management |

---

## 📁 Project Structure

```
AI Smart Home/
│
├── simulator/
│   └── device_sim.py           # MQTT device data publisher (3 virtual devices)
│
├── data/
│   ├── logger.py               # MQTT subscriber — logs events to CSV
│   ├── uci_adapter.py          # Converts UCI dataset to project events format
│   ├── generate_data.py        # Synthetic data generator (development only)
│   ├── fix_headers.py          # One-time utility to repair headerless CSV
│   ├── events.csv              # Master event log (118,410 rows, real data)
│   ├── forecast.csv            # Prophet model forecast output
│   └── energydata_complete.csv # Raw UCI source dataset
│
├── ml/
│   ├── habit_model.py          # Random Forest classifier (light + thermostat)
│   ├── anomaly_model.py        # Isolation Forest anomaly detector
│   ├── forecasting.py          # Prophet 14-day energy forecaster
│   └── models/
│       ├── light_model.pkl     # Saved light habit model
│       ├── light_le.pkl        # Label encoder for light states
│       ├── thermo_model.pkl    # Saved thermostat model
│       ├── thermo_le.pkl       # Label encoder for comfort zones
│       ├── anomaly_model.pkl   # Saved Isolation Forest model
│       └── forecast_model.pkl  # Saved Prophet model
│
├── rules/
│   ├── rule_engine.py          # Generates 48 automation rules from trained models
│   └── generated_rules.json   # Output: 24hr schedule for light + thermostat
│
├── business/
│   └── kpi_engine.py           # ROI, efficiency score, savings, compliance rate
│
├── dashboard/
│   └── app.py                  # Streamlit executive dashboard (7 sections)
│
├── .env                        # Environment variables (broker, port, log path)
├── .gitignore                  # Excludes backup files and OS artefacts
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) installed
- [Mosquitto MQTT Broker](https://mosquitto.org/download/) installed (Windows)
- Python 3.10
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/anirudh-rs/ai-smart-home.git
cd "ai-smart-home"
```

### 2 — Create and activate conda environment

```bash
conda create -n smarthome python=3.10 -y
conda activate smarthome
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download the UCI dataset

Go to: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

Download and place `energydata_complete.csv` in the `data/` folder.

### 5 — Convert UCI data to project format

```bash
python data/uci_adapter.py
```

Expected output:
```
📂 Loading UCI dataset...
   19,735 rows loaded
✅ Saved 118,410 real events to data/events.csv
```

### 6 — Train all models

```bash
python ml/habit_model.py
python rules/rule_engine.py
python ml/anomaly_model.py
python ml/forecasting.py
```

### 7 — Generate KPI report

```bash
python business/kpi_engine.py
```

### 8 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens automatically at `http://localhost:8501`

---

## 📈 Dashboard Sections

| # | Section | What It Shows |
|---|---|---|
| 1 | 📊 Business KPIs | Efficiency score, annual savings, anomaly count, compliance rate, annual cost |
| 2 | 📝 AI Recommendation | Plain English insight auto-generated from model outputs |
| 3 | 🕐 Automation Schedule | 24-hour light energy chart + thermostat comfort zone schedule |
| 4 | 🚨 Anomaly Analysis | Anomalies by hour bar chart + normal vs anomalous pie chart + detail table |
| 5 | 📈 14-Day Forecast | Prophet forecast chart with historical fit and 95% confidence interval band |
| 6 | 🔮 AI Predictor | Hour slider → real-time light, thermostat, and anomaly prediction |
| 7 | 🗃️ Raw Data | Full events.csv viewer |

---

## 🔄 Data Pipeline Detail

### UCI Dataset Mapping

The `uci_adapter.py` script maps UCI columns to the project's internal event format:

| UCI Column | MQTT Topic | Key | Notes |
|---|---|---|---|
| `date` | — | `timestamp`, `hour` | Parsed to ISO format + hour integer |
| `lights` | `home/livingroom/light` | `state`, `brightness` | >10Wh = ON, ≤10Wh = OFF |
| `T1` | `home/bedroom/thermostat` | `temp_c` | Kitchen area temperature (°C) |
| `RH_1` | `home/livingroom/humidity` | `humidity_pct` | Relative humidity (%) |
| `Appliances` | `home/appliances/energy` | `usage_wh` | Total appliance energy (Wh) |
| `T_out` | `home/outside/weather` | `temp_out_c` | Outdoor temperature (°C) |

### Model Training Summary

**Random Forest (habit_model.py)**
- Features: `hour` (0–23)
- Light target: binary ON/OFF classification
- Thermostat target: 4-class comfort zone (cold / cool / comfortable / warm)
- Train/test split: 80/20, random_state=42
- **Light accuracy on UCI data: 88%**

**Isolation Forest (anomaly_model.py)**
- Features: `hour`, `light_on`, `temp_c`, `motion`
- contamination=0.05 (expects 5% anomalies)
- Trained on 5,000-row random sample (performance optimisation)
- **Anomaly rate: 5.0%**

**Prophet (forecasting.py)**
- Target: daily total energy consumption (Wh)
- External regressors: avg daily temperature, avg daily humidity
- Weekly seasonality enabled
- Forecast horizon: 14 days
- **95% confidence intervals included**

---

## ⚠️ Known Limitations & Design Decisions

### Light Chart: Energy Consumption vs ON/OFF
The UCI dataset uses low-wattage LED lighting that rarely exceeds 10Wh per reading.
The habit model correctly learned this pattern and predicts OFF for all 24 hours.
Rather than displaying a misleading empty binary chart, the light schedule section
shows **actual hourly average energy consumption (Wh)** from the raw data.
Gold bars indicate active usage (>10Wh). This is more honest and more informative.

### Stable Thermostat & Light Predictions
The dataset covers a single Belgian household during winter/spring (Jan–May 2016).
Indoor temperatures sit consistently in the 19–22°C range, mapping to 'comfortable'
across most hours. The model correctly learned these consistent patterns. Models
trained on multi-year or multi-season data would show greater variation.
A context note appears in the dashboard predictor after each prediction is run.

### Anomaly Detection Sampling
The Isolation Forest was trained on a 5,000-row random sample rather than the full
118,410 rows. This is standard practice — Isolation Forest is computationally
expensive at scale, and 5,000 rows is statistically sufficient for learning the
normal distribution of this dataset. The 5.0% anomaly rate from the sample is
representative of the full dataset.

### Energy Cost Assumptions
The KPI engine uses **$0.13/kWh** (US average residential rate). The UCI household
is Belgian — actual Belgian rates are approximately €0.28–0.35/kWh, which would
roughly double the cost figures shown. The rate is configurable:

```python
# business/kpi_engine.py — line 8
COST_PER_KWH = 0.13   # Change to 0.30 for Belgian rate
```

### KPI Savings Calculation
Because the light model predicts OFF for all hours (correct for this LED household),
the 'optimal' light cost is $0, making estimated savings equal to total light cost.
This is mathematically accurate given the model output but should be interpreted
as the energy cost baseline, not a guaranteed savings figure.

---

## 🌐 Live Demo & Repository

| Resource | Link |
|---|---|
| 🚀 Live Dashboard | https://share.streamlit.io (deployed via GitHub) |
| 📂 GitHub Repository | https://github.com/anirudh-rs/ai-smart-home |
| 📊 UCI Dataset | https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |

---

## 📋 requirements.txt

```
paho-mqtt
scikit-learn
pandas
numpy
streamlit
plotly
python-dotenv
schedule
prophet
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Prophet requires additional system dependencies on some platforms.
> On Windows with Anaconda, `pip install prophet` handles this automatically.
> If you encounter Stan/CmdStanPy errors, run: `pip install pystan cmdstanpy --upgrade`

---

## 🔧 Running the IoT Simulator (Optional)

To generate fresh data using the MQTT simulator instead of the UCI dataset:

**Window 1 — Start Mosquitto broker:**
```bash
# Mosquitto runs as a Windows Service automatically after installation
# To start manually: net start mosquitto (run as Administrator)
mosquitto -v
```

**Window 2 — Start device simulator:**
```bash
conda activate smarthome
cd "AI Smart Home"
python simulator/device_sim.py
```

**Window 3 — Start event logger:**
```bash
conda activate smarthome
cd "AI Smart Home"
python data/logger.py
```

Let run for 5–10 minutes to collect meaningful data, then Ctrl+C both windows.
Run `python fix_headers.py` if the CSV was created without headers.

---

## 💼 Resume Bullets (XYZ Format)

> *Developed an end-to-end IoT analytics platform by engineering a data pipeline
> that processed 118,410 real sensor events from the UCI Appliances Energy Dataset
> and training a Random Forest classifier, resulting in 88% habit prediction accuracy
> across 24-hour device automation schedules*

> *Identified abnormal smart home device behaviour by applying Isolation Forest
> unsupervised ML across 138 days of real household data, resulting in detection of
> a 5% anomaly rate and a quantified $6,465 annual energy cost baseline surfaced
> through an executive KPI dashboard*

> *Forecasted 14-day household energy consumption by training a Prophet time series
> model with 95% confidence intervals on real Belgian household IoT data, resulting
> in a $693 annual cost projection delivered via a fully deployed Streamlit dashboard
> accessible at a live public URL*

---

## 🗺️ Roadmap

Planned improvements in priority order:

| Phase | Feature | Skills Demonstrated |
|---|---|---|
| Session 2 | SQLite data pipeline replacing flat CSV | Data engineering, ETL |
| Session 1 | Real-time streaming dashboard updates | Real-time systems |
| Session 3 | What-if ROI scenario simulator | Business storytelling |
| Session 4 | XGBoost / LightGBM model benchmarking + season features | Advanced ML |

---

## 📚 Dataset Credit

**UCI Appliances Energy Prediction Dataset**
- Source: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- Authors: Luis Candanedo, Véronique Feldheim, Dominique Deramaix
- Period: January 11 – May 27, 2016
- Location: Stambruges, Belgium
- Sampling rate: Every 10 minutes (19,735 readings)
- Features: 28 variables including appliance energy, lighting, temperature per room,
  humidity per room, outdoor weather conditions

---

## 👤 About This Project

Built as a DA/DS portfolio project demonstrating:

- **Data Engineering** — real IoT data ingestion, transformation, and pipeline design
- **Supervised ML** — Random Forest classification with real-world accuracy benchmarking
- **Unsupervised ML** — Isolation Forest anomaly detection on time-series sensor data
- **Time Series Forecasting** — Prophet with external regressors and confidence intervals
- **Business Intelligence** — KPI quantification, ROI framing, executive dashboard design
- **Deployment** — GitHub version control + Streamlit Cloud live hosting
- **Data Literacy** — honest handling of model limitations and dataset characteristics

---

*Last updated: April 2026*
