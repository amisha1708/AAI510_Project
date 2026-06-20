# ACN-Data Setup Instructions

ACN-Data requires a free account registration at Caltech before download.
It cannot be downloaded automatically. Follow the steps below.

---

## Step 1 — Register for an Account

1. Go to [https://ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
2. Click **"ACCOUNT"** in the top navigation
3. Register with your email address (free, no institutional affiliation required)
4. Confirm your email and log in

---

## Step 2 — Download via Web Interface (Easiest)

1. After logging in, go to the **Web Interface** tab on the dataset page
2. Select site: **Caltech** (largest dataset, ~50K sessions)
3. Set date range: **2018-01-01 to 2020-12-31** (recommended for richness)
4. Click **Download** — saves as a JSON file
5. Optionally also download **JPL** site for comparison
6. Save both files to this folder: `DataSources/acn_data/`

---

## Step 3 — Convert JSON to CSV (Optional but Recommended)

Run this Python snippet to flatten the JSON into a CSV for easier use:

```python
import json
import pandas as pd

with open("caltech_sessions.json", "r") as f:
    data = json.load(f)

sessions = pd.DataFrame(data["_items"])

# Parse key fields
sessions["connectionTime"]    = pd.to_datetime(sessions["connectionTime"])
sessions["disconnectTime"]    = pd.to_datetime(sessions["disconnectTime"])
sessions["doneChargingTime"]  = pd.to_datetime(sessions["doneChargingTime"])
sessions["dwell_time_hrs"]    = (
    (sessions["disconnectTime"] - sessions["connectionTime"])
    .dt.total_seconds() / 3600
)
sessions["hour_of_day"]  = sessions["connectionTime"].dt.hour
sessions["day_of_week"]  = sessions["connectionTime"].dt.dayofweek
sessions["month"]        = sessions["connectionTime"].dt.month

sessions.to_csv("caltech_sessions.csv", index=False)
print(f"Saved {len(sessions)} sessions")
```

---

## Key Fields in ACN-Data

| Field | Description | Used For |
|-------|-------------|----------|
| `connectionTime` | When the EV plugged in | Arrival distribution features |
| `disconnectTime` | When the EV unplugged | Dwell time calculation |
| `doneChargingTime` | When charging completed | Charging vs. idle split |
| `kWhDelivered` | Energy delivered (kWh) | Energy request features |
| `requestedEnergy` | Energy requested by user (kWh) | Behavioral demand signal |
| `userID` | Anonymized user identifier | User behavior clustering (Week 4) |
| `stationID` | Charging station identifier | Station-level analysis |
| `siteID` | Site (Caltech vs. JPL) | Multi-site comparison |

---

## Alternative: Python API Client

```bash
pip install acnportal
```

```python
from acnportal import acnsim
from acnportal.acnsim import network
from acndata.acndata import DataClient

# Requires API key from your account settings
client = DataClient(api_token="YOUR_API_TOKEN")
docs = client.get_sessions("caltech", 
                            start="2018-01-01", 
                            end="2020-12-31",
                            num_sessions=50000)
```

Get your API token from: Account Settings → API Token on the dataset site.

---

## Expected Files After Setup

```
acn_data/
├── ACN_DATA_SETUP.md          ← This file
├── caltech_sessions.json      ← Raw download from Web Interface
├── caltech_sessions.csv       ← Converted flat CSV
└── jpl_sessions.csv           ← Optional: JPL site data
```

---

## Size Reference

| Site | Approx Sessions | Date Range |
|------|-----------------|------------|
| Caltech | ~50,000 | 2018–2021 |
| JPL | ~10,000 | 2019–2021 |

*ACN-Data is maintained by Caltech / PowerFlex Systems and is free for research use.*
