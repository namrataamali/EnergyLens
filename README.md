# ⚡ EnergyWatch AI — Smart Electricity Monitoring & Anomaly Detection

A dark-themed Streamlit analytics dashboard for hourly electricity consumption, Isolation Forest anomaly detection, statistical risk prioritization, and rule-based energy recommendations.

## Quick start

1. Extract this ZIP.
2. Copy your **`energywatch_model.pkl`** and **`energywatch_data.csv`** (downloaded from Google Colab) into the extracted `EnergyWatch_AI` folder, next to `app.py`.
3. Open a terminal in that folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Using `python -m pip` avoids a common Windows problem caused by stale `pip.exe` launchers.

## Features

- **Overview:** power KPIs, time-series anomalies, daily and hourly usage charts.
- **Anomaly Intelligence:** risk distribution, anomaly timing, flagged records.
- **Recommendations:** usage rules and priority filtering.
- **Data Explorer:** inspect readings and download analyzed CSV.
- **CSV upload:** supports your pre-scored CSV or new data with the seven original electrical features. New files require `energywatch_model.pkl`.

## Required CSV columns

`Datetime` (or `Date` and `Time`), `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.

The model was trained on **hourly mean** values of these seven features in this exact order. The app aggregates minute-level uploads into hourly averages. For reliable results, uploaded measurements should be comparable to the training data.

## Model results from the notebook

| Metric | Result |
|---|---:|
| Algorithm | Isolation Forest |
| Hourly observations | 34,168 |
| Normal observations | 32,459 |
| Flagged anomalies | 1,709 |
| Anomaly rate | 5.00% |
| Silhouette score | 0.5853 |

Accuracy, precision, recall and F1-score are **not available without ground-truth anomaly labels**. The silhouette score describes separation of predicted groups, not true anomaly-detection accuracy. Risk levels are based on anomaly scores, not confirmed electrical faults.

## Dataset

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — real measurements from a household in France, 2006–2010. The model's contamination parameter was set to 0.05.

**Note:** `Global_active_power` is average power in kW, not energy in kWh. Recommendations use illustrative thresholds and are not electrical safety diagnoses.
