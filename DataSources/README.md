# DataSources — Download Inventory

This folder contains all raw data for the AAI-510 EV Charging Demand Prediction project.
Each subfolder corresponds to one of the three data sources in the fusion strategy.

---

## Status

| Source | Folder | Status | Files | Size |
|--------|--------|--------|-------|------|
| UrbanEV (Shenzhen) | `urbanev/` | **Downloaded** | 14 files | ~118 MB |
| Kaggle Global EV Stations | `kaggle_global/` | **Downloaded** | 1 file | ~692 KB |
| ACN-Data (Caltech) | `acn_data/` | **Requires manual setup** | See instructions | ~varies |

---

## 1. UrbanEV — `urbanev/`

**Source:** [GitHub — IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV)
**License:** CC0 (public domain)
**Geography:** Shenzhen, China | **Coverage:** Sep 2022 – Feb 2023 | **Granularity:** Hourly

| File | Description | Shape | ML Use |
|------|-------------|-------|--------|
| `occupancy.csv` | Hourly occupancy rate (%) per zone | 4,344 rows × 276 cols | **Primary target variable** |
| `volume.csv` | Hourly energy consumed (kWh) per zone | Same shape | Secondary target |
| `volume-11kW.csv` | Volume using Tesla Model Y 11kW standard | Same shape | Alternative energy estimate |
| `duration.csv` | Hourly avg charging duration (hrs) per zone | Same shape | Feature + target |
| `e_price.csv` | Electricity price (Yuan/kWh) by time | Hourly | Demand elasticity feature |
| `s_price.csv` | Service fee (Yuan/kWh) by time | Hourly | Demand elasticity feature |
| `weather_airport.csv` | Weather at Bao'an Airport station | Hourly | Weather features (W1–W6) |
| `weather_central.csv` | Weather at Futian city center station | Hourly | Weather features (W1–W6) |
| `weather_header.txt` | Column definitions for weather files | — | Reference |
| `poi.csv` | Points of Interest: food, business, lifestyle | Per zone | Location context features |
| `inf.csv` | Station coordinates, capacity, area (filtered) | 1,362 stations | Infrastructure features |
| `inf_raw.csv` | All station data before filtering | 1,682 stations | Reference |
| `adj.csv` | Zone adjacency matrix (275 × 275) | Sparse | Graph-based models |
| `distance.csv` | Zone-to-zone distance matrix | 275 × 275 | Spatial lag features |

**Weather column reference (from `weather_header.txt`):**

| Column | Meaning |
|--------|---------|
| `T` | Air temperature (°C) at 2m height |
| `P0` | Atmospheric pressure at station level (mmHg) |
| `P` | Atmospheric pressure at mean sea level (mmHg) |
| `U` | Relative humidity (%) at 2m |
| `nRAIN` | Rain intensity: 0=none, 1=light, 2=moderate, 3=heavy |
| `Td` | Dewpoint temperature (°C) at 2m |

---

## 2. Kaggle Global EV Stations — `kaggle_global/`

**Source:** [Kaggle — vivekattri/global-ev-charging-stations-dataset](https://www.kaggle.com/datasets/vivekattri/global-ev-charging-stations-dataset)
**License:** Apache 2.0
**Downloaded via:** Kaggle CLI (`kaggle datasets download vivekattri/global-ev-charging-stations-dataset --unzip`)

| File | Rows | Description |
|------|------|-------------|
| `detailed_ev_charging_stations.csv` | 5,000 | Global charging station records |

**Columns:**

| Column | Type | ML Use |
|--------|------|--------|
| `Station ID` | string | Row identifier |
| `Latitude` / `Longitude` | float | Geospatial features / clustering |
| `Address` | string | Location parsing |
| `Charger Type` | string (AC/DC) | Infrastructure type feature |
| `Cost (USD/kWh)` | float | Pricing context |
| `Availability` | string | Operating hours feature |
| `Distance to City (km)` | float | Urban proximity feature |
| `Usage Stats (avg users/day)` | int | **Demand signal — cross-validate with UrbanEV** |
| `Station Operator` | string | Operator clustering |
| `Charging Capacity (kW)` | float | Power class feature |
| `Connector Types` | string | Multi-label encoding |
| `Installation Year` | int | Infrastructure age feature |
| `Renewable Energy Source` | bool | Sustainability flag |
| `Reviews (Rating)` | float (1–5) | Station quality signal |
| `Parking Spots` | int | Capacity context |
| `Maintenance Frequency` | string | Reliability signal |

> **Note:** This dataset is a structured sample (5,000 records). The `Usage Stats (avg users/day)` column provides a usable demand proxy for infrastructure-level clustering (Week 4) and cross-dataset demand validation.

---

## 3. ACN-Data (Caltech) — `acn_data/`

**Source:** [ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
**License:** Free for research use (requires account registration)
**Status:** Manual download required — see `acn_data/ACN_DATA_SETUP.md`

**Setup instructions:** Open `acn_data/ACN_DATA_SETUP.md` and follow the 3-step guide.

**Key fields after download:**

| Field | Description |
|-------|-------------|
| `connectionTime` | Arrival timestamp → hour_of_day, day_of_week features |
| `disconnectTime` | Departure timestamp → dwell time |
| `kWhDelivered` | Actual energy delivered |
| `requestedEnergy` | Energy requested by driver |
| `userID` | Anonymized ID for user clustering (Week 4) |
| `stationID` | Station-level analysis |

---

## Folder Structure

```
DataSources/
├── README.md                       ← This file
├── urbanev/
│   ├── occupancy.csv               ← PRIMARY TARGET (hourly × zone)
│   ├── volume.csv
│   ├── volume-11kW.csv
│   ├── duration.csv
│   ├── e_price.csv
│   ├── s_price.csv
│   ├── weather_airport.csv
│   ├── weather_central.csv
│   ├── weather_header.txt
│   ├── poi.csv
│   ├── inf.csv
│   ├── inf_raw.csv
│   ├── adj.csv
│   └── distance.csv
├── kaggle_global/
│   └── detailed_ev_charging_stations.csv
└── acn_data/
    ├── ACN_DATA_SETUP.md           ← Setup instructions (start here)
    ├── caltech_sessions.json       ← (after manual download)
    └── caltech_sessions.csv        ← (after JSON-to-CSV conversion)
```

---

## Quick Data Facts

| Metric | Value |
|--------|-------|
| UrbanEV zones | 275 traffic zones |
| UrbanEV stations | 1,362 charging stations |
| UrbanEV charging piles | 17,532 piles |
| UrbanEV time range | Sep 1, 2022 – Feb 28, 2023 |
| UrbanEV hourly rows | 4,344 timestamps |
| Kaggle station records | 5,000 global stations |
| ACN-Data sessions (Caltech) | ~50,000 sessions (2018–2021) |

---

## Re-download Commands

```bash
# UrbanEV (all 14 files from GitHub)
GH_RAW="https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/main/data"
for f in adj.csv duration.csv e_price.csv inf.csv inf_raw.csv occupancy.csv \
          s_price.csv volume.csv "volume-11kW.csv" weather_airport.csv \
          weather_central.csv weather_header.txt distance.csv poi.csv; do
  curl -sL "$GH_RAW/$f" -o "urbanev/$f"
done

# Kaggle Global EV Stations
kaggle datasets download vivekattri/global-ev-charging-stations-dataset \
  --unzip --path kaggle_global/
```

---

*Last updated: June 14, 2026*
