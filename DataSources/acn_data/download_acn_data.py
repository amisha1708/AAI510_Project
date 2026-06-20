#!/usr/bin/env python3
"""
ACN-Data Download Script
Downloads charging session data from the Caltech Adaptive Charging Network REST API.
Token is injected via ACN_API_TOKEN environment variable (managed by cortex secrets).

Usage:
    ACN_API_TOKEN=<token> python3 download_acn_data.py

Output:
    caltech_sessions.csv  — flat CSV of all sessions
    jpl_sessions.csv      — flat CSV of JPL site sessions (optional)
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
# Token read from env var — inject via: ACN_API_TOKEN="acn_api_token" python3 ...
# or hardcode below for local testing (rotate token after use)
API_TOKEN   = os.environ.get("ACN_API_TOKEN", "oUZXkNDb0Q-cD39tnmtfKzaXoxlqr2ae5yQdEyvkAaA")
BASE_URL    = "https://ev.caltech.edu/api/v1"
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))

SITES = {
    "caltech": {
        "start": "2018-01-01",
        "end":   "2020-12-31",
        "output": "caltech_sessions.csv",
    },
    "jpl": {
        "start": "2019-01-01",
        "end":   "2021-06-30",
        "output": "jpl_sessions.csv",
    },
}

MAX_RESULTS = 100    # ACN API page size (100 is the safe default)
RETRY_WAIT  = 5      # seconds to wait on rate-limit (429)
MAX_RETRIES = 3
# ─────────────────────────────────────────────────────────────────────────────


def http_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to RFC 2822 format required by ACN Eve API."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def get(url: str, params: dict = None) -> dict:
    """GET request with Basic Auth and retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, auth=(API_TOKEN, ""), params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"  Rate limited — waiting {RETRY_WAIT}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_WAIT)
            elif resp.status_code == 401:
                print("[ERROR] Invalid or expired API token.")
                sys.exit(1)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts")
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt}/{MAX_RETRIES}")
            if attempt == MAX_RETRIES:
                raise
    return {}


def flatten_session(s: dict) -> dict:
    """Extract and flatten key fields from a raw session record."""
    return {
        "session_id":          s.get("_id"),
        "site_id":             s.get("siteID"),
        "station_id":          s.get("stationID"),
        "user_id":             s.get("userID"),
        "connection_time":     s.get("connectionTime"),
        "disconnect_time":     s.get("disconnectTime"),
        "done_charging_time":  s.get("doneChargingTime"),
        "kwh_requested":       s.get("kWhRequested"),
        "kwh_delivered":       s.get("kWhDelivered"),
        "arrival_soc":         s.get("arrivalSOC"),
        "departure_soc":       s.get("departureSOC"),
        "user_input_deadline": s.get("userInputDeadline"),
        "user_input_energy":   s.get("userInputEnergy"),
    }


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived time and behavioral features for Week 2 feature engineering."""
    df["connection_time"]   = pd.to_datetime(df["connection_time"],   utc=True)
    df["disconnect_time"]   = pd.to_datetime(df["disconnect_time"],   utc=True, errors="coerce")
    df["done_charging_time"]= pd.to_datetime(df["done_charging_time"],utc=True, errors="coerce")

    df["hour_of_day"]   = df["connection_time"].dt.hour
    df["day_of_week"]   = df["connection_time"].dt.dayofweek     # 0=Monday
    df["month"]         = df["connection_time"].dt.month
    df["year"]          = df["connection_time"].dt.year
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)

    # Season: 1=Winter 2=Spring 3=Summer 4=Fall
    df["season"] = df["month"].map({
        12:1, 1:1, 2:1,
        3:2, 4:2, 5:2,
        6:3, 7:3, 8:3,
        9:4, 10:4, 11:4
    })

    df["dwell_time_hrs"] = (
        (df["disconnect_time"] - df["connection_time"])
        .dt.total_seconds() / 3600
    ).clip(lower=0)

    df["charge_time_hrs"] = (
        (df["done_charging_time"] - df["connection_time"])
        .dt.total_seconds() / 3600
    ).clip(lower=0)

    df["idle_time_hrs"] = (df["dwell_time_hrs"] - df["charge_time_hrs"]).clip(lower=0)
    df["is_fast_session"] = (df["dwell_time_hrs"] < 0.5).astype(int)

    return df


def download_site(site_id: str, start: str, end: str, output_file: str):
    """Download all sessions for a site using ACN Eve-API pagination."""
    print(f"\n{'='*60}")
    print(f"Downloading site: {site_id.upper()}")
    print(f"Date range: {start} → {end}")
    print(f"{'='*60}")

    # ACN API uses Eve-style where clause with RFC 2822 dates
    where_clause = (
        f'connectionTime >= "{http_date(start)}" '
        f'and connectionTime <= "{http_date(end)}"'
    )
    first_url = f"{BASE_URL}/sessions/{site_id}"
    first_params = {
        "where":       where_clause,
        "sort":        "connectionTime",
        "max_results": MAX_RESULTS,
    }

    all_sessions = []
    url    = first_url
    params = first_params
    page   = 1

    while True:
        print(f"  Page {page:>4} — {len(all_sessions):>6} records so far...", end=" ", flush=True)
        data = get(url, params)

        if not data:
            print("empty response, stopping.")
            break

        items = data.get("_items", [])
        if not items:
            print("done.")
            break

        all_sessions.extend([flatten_session(s) for s in items])
        total = data.get("_meta", {}).get("total", "?")
        print(f"+{len(items)} records (total available: {total})")

        # Follow _links.next for next page — ACN Eve pagination
        next_href = data.get("_links", {}).get("next", {}).get("href")
        if not next_href:
            print("  No more pages.")
            break

        # Next page URL is relative — prepend base
        url    = f"{BASE_URL}/{next_href.lstrip('/')}"
        params = None   # params are already encoded in the next href
        page  += 1
        time.sleep(0.3)

    if not all_sessions:
        print(f"[WARNING] No sessions downloaded for {site_id}.")
        return

    df = pd.DataFrame(all_sessions)
    df = enrich_dataframe(df)

    output_path = os.path.join(OUTPUT_DIR, output_file)
    df.to_csv(output_path, index=False)

    print(f"\n  Saved {len(df):,} sessions → {output_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range in data: {df['connection_time'].min()} → {df['connection_time'].max()}")

    # Print quick summary
    print(f"\n  Quick stats:")
    print(f"    Avg kWh delivered:   {df['kwh_delivered'].mean():.2f}")
    print(f"    Avg dwell time (hrs):{df['dwell_time_hrs'].mean():.2f}")
    print(f"    Fast sessions (<30m):{df['is_fast_session'].sum():,} ({df['is_fast_session'].mean()*100:.1f}%)")
    print(f"    Unique users:        {df['user_id'].nunique():,}")
    print(f"    Unique stations:     {df['station_id'].nunique():,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ACN-Data sessions")
    parser.add_argument("--sites",  nargs="+", default=["caltech"],
                        choices=["caltech", "jpl"],
                        help="Sites to download (default: caltech)")
    parser.add_argument("--start",  default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end",    default=None, help="Override end date (YYYY-MM-DD)")
    args = parser.parse_args()

    for site in args.sites:
        cfg = SITES[site]
        download_site(
            site_id     = site,
            start       = args.start or cfg["start"],
            end         = args.end   or cfg["end"],
            output_file = cfg["output"],
        )

    print("\nAll downloads complete.")
