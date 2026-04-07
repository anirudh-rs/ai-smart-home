\# 🏠 AI Smart Home Automation System

\### An IoT Analytics Platform for Energy Optimization \& Habit Learning



\---



\## 📌 Project Overview



This project simulates and analyzes a smart home environment using real IoT sensor 

data, machine learning, and time series forecasting to optimize energy consumption, 

detect anomalous device behavior, and generate automated control rules.



Built as a \*\*business impact data science project\*\* targeting DA/DS roles, it 

demonstrates end-to-end data pipeline development, predictive modeling, and 

executive-level dashboard storytelling.



\---



\## 🎯 Business Problem



Smart homes generate thousands of sensor events daily. Without intelligent analysis:

\- \*\*Energy is wasted\*\* through suboptimal device scheduling

\- \*\*Anomalies go undetected\*\* — lights left on, unusual activity patterns

\- \*\*No actionable insights\*\* are surfaced for cost reduction



This platform addresses all three problems using ML-driven automation.



\---



\## 📊 Key Results



| Metric | Value |

|---|---|

| Dataset | UCI Appliances Energy Dataset (real household data) |

| Events Processed | 118,410 IoT sensor events |

| Data Range | Jan 11 – May 27, 2016 (138 days) |

| Light Model Accuracy | 88% (Random Forest) |

| Energy Efficiency Score | 88.5% |

| Anomaly Rate | 5.0% (Isolation Forest) |

| Anomalies Detected | 250 flagged from 5,000 sampled events |

| Annual Energy Cost Baseline | $6,465 |

| 14-Day Forecast | $26.59 projected cost |

| Projected Annual Cost | $693 (from forecast model) |



\---



\## 🏗️ Architecture



UCI Dataset (real data)

↓

UCI Adapter (data/uci\_adapter.py)

↓

events.csv (118,410 events)

↓

┌──────────────────────────────────┐

│         ML Pipeline              │

│  ┌─────────────────────────┐     │

│  │ Random Forest           │     │

│  │ Habit Model (88% acc.)  │     │

│  └──────────┬──────────────┘     │

│             ↓                    │

│  ┌─────────────────────────┐     │

│  │ Rule Engine             │     │

│  │ 48 automation rules     │     │

│  └──────────┬──────────────┘     │

│             ↓                    │

│  ┌─────────────────────────┐     │

│  │ Isolation Forest        │     │

│  │ Anomaly Detection (5%)  │     │

│  └──────────┬──────────────┘     │

│             ↓                    │

│  ┌─────────────────────────┐     │

│  │ Prophet Forecasting     │     │

│  │ 14-day energy forecast  │     │

│  └──────────┬──────────────┘     │

└────────────────────────────────┘

↓

KPI \& ROI Engine

↓

Streamlit Executive Dashboard



\---



\## 🛠️ Tech Stack



| Layer | Technology |

|---|---|

| Language | Python 3.10 |

| ML Models | scikit-learn (Random Forest, Isolation Forest) |

| Forecasting | Prophet (Meta) |

| IoT Protocol | MQTT (paho-mqtt + Mosquitto broker) |

| Dashboard | Streamlit + Plotly |

| Data | UCI Appliances Energy Prediction Dataset |

| Environment | Anaconda / conda |



\---



\## 📁 Project Structure



AI Smart Home/

│

├── simulator/

│   └── device\_sim.py          # MQTT device simulator

│

├── data/

│   ├── logger.py              # MQTT event logger

│   ├── uci\_adapter.py         # Converts UCI dataset to events format

│   ├── events.csv             # Main event log (118,410 rows)

│   ├── forecast.csv           # Prophet forecast output

│   └── energydata\_complete.csv # Raw UCI dataset

│

├── ml/

│   ├── habit\_model.py         # Random Forest habit classifier

│   ├── anomaly\_model.py       # Isolation Forest anomaly detector

│   ├── forecasting.py         # Prophet time series forecaster

│   └── models/                # Saved model files (.pkl)

│

├── rules/

│   ├── rule\_engine.py         # Generates automation rules from models

│   └── generated\_rules.json   # 48 automation rules output

│

├── business/

│   └── kpi\_engine.py          # ROI, efficiency, savings calculations

│

├── dashboard/

│   └── app.py                 # Streamlit executive dashboard

│

├── .env                       # Environment variables

└── requirements.txt           # Python dependencies



\---



\## 🚀 How To Run



\### Prerequisites

\- Anaconda / conda installed

\- Mosquitto MQTT broker installed

\- Python 3.10



\### Setup



\*\*1. Create and activate environment:\*\*

```bash

conda create -n smarthome python=3.10 -y

conda activate smarthome

```



\*\*2. Install dependencies:\*\*

```bash

pip install paho-mqtt scikit-learn pandas numpy streamlit plotly python-dotenv schedule prophet

```



\*\*3. Download UCI dataset:\*\*

\- Go to: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

\- Download and place `energydata\_complete.csv` in the `data/` folder



\*\*4. Convert UCI data:\*\*

```bash

python data/uci\_adapter.py

```



\*\*5. Train all models:\*\*

```bash

python ml/habit\_model.py

python rules/rule\_engine.py

python ml/anomaly\_model.py

python ml/forecasting.py

```



\*\*6. Generate KPI report:\*\*

```bash

python business/kpi\_engine.py

```



\*\*7. Launch dashboard:\*\*

```bash

streamlit run dashboard/app.py

```



\---



\## 📈 Dashboard Features



| Section | Description |

|---|---|

| 📊 Business KPIs | Efficiency score, savings, anomaly count, compliance rate |

| 📝 AI Recommendation | Plain English insight generated from model outputs |

| 🕐 Automation Schedule | 24-hour device control schedule from habit model |

| 🚨 Anomaly Analysis | Hourly anomaly breakdown + normal vs anomalous pie chart |

| 📈 Forecast | 14-day energy consumption forecast with confidence intervals |

| 🔮 AI Predictor | Real-time hour-based device state prediction |



\---



\## ⚠️ Known Limitations \& Design Decisions



\### 1. Light Chart Visualization

The UCI dataset uses low-wattage LED lighting (mostly <10Wh per reading).

The habit model correctly learned this and predicts OFF for all hours.

\*\*Decision:\*\* Replaced binary ON/OFF chart with hourly average energy

consumption (Wh) for more meaningful insight.



\### 2. Stable Predictions

The UCI dataset covers a single Belgian household during winter/spring

(Jan–May 2016). Indoor temperatures and lighting patterns are consistent

across this period, resulting in stable model predictions.

\*\*Decision:\*\* Added methodology note in dashboard predictor to provide

context for stakeholders.



\### 3. Anomaly Count Scale

With 118,410 real events, the KPI engine reports 2,271 anomalies.

The Isolation Forest was trained on a 5,000-row sample for performance.

\*\*Decision:\*\* Documented this trade-off — sampling is standard practice

for Isolation Forest on large datasets.



\### 4. Energy Cost Calculation

The KPI engine uses $0.13/kWh (US average). The UCI household is Belgian

so actual costs would differ. This is configurable in `business/kpi\_engine.py`.



\---



\## 🔄 Data Pipeline



Raw UCI CSV (19,735 readings @ 10min intervals)

↓ uci\_adapter.py

events.csv (118,410 rows, 5 topics, 24 hours coverage)

↓ habit\_model.py

Random Forest models (light + thermostat)

↓ rule\_engine.py

generated\_rules.json (48 rules)

↓ anomaly\_model.py

Isolation Forest (5% anomaly rate)

↓ forecasting.py

Prophet 14-day forecast ($693/year projection)

↓ kpi\_engine.py

Business KPIs (88.5% efficiency, $6,465 baseline)

↓ dashboard/app.py

Executive Streamlit Dashboard



\---



\## 📚 Dataset Credit



\*\*UCI Appliances Energy Prediction Dataset\*\*

\- Source: UCI Machine Learning Repository

\- URL: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

\- Period: January 11 – May 27, 2016

\- Location: Stambruges, Belgium

\- Readings: Every 10 minutes (19,735 total)



\---



\## 👤 Author Notes



This project was built as part of a DA/DS portfolio to demonstrate:

\- Real-world IoT data processing

\- Multiple ML paradigms (supervised, unsupervised, time series)

\- Business impact quantification

\- Executive dashboard storytelling

