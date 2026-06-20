#!/usr/bin/env python3
"""
ACN-Data API Connection Test
Run: python3 test_connection.py
Enter your token when prompted — input is hidden (no echo).
"""

import getpass
import requests

BASE_URL = "https://ev.caltech.edu/api/v1/sessions"
SITE     = "caltech"
TEST_URL = f"{BASE_URL}/{SITE}"
PARAMS   = {
    "where":       'connectionTime >= "2019-01-01" and connectionTime <= "2019-01-03"',
    "max_results": 3,
}

def test(label, **kwargs):
    try:
        resp = requests.get(TEST_URL, params=PARAMS, timeout=15, **kwargs)
        status = resp.status_code
        if status == 200:
            data  = resp.json()
            total = data.get("_meta", {}).get("total", "?")
            items = data.get("_items", [])
            print(f"  [{label}] HTTP {status} — CONNECTED")
            print(f"  Total sessions in range : {total}")
            print(f"  Fields in first session : {list(items[0].keys()) if items else 'none'}")
            return True
        else:
            print(f"  [{label}] HTTP {status} — failed")
            return False
    except Exception as e:
        print(f"  [{label}] ERROR — {e}")
        return False

def main():
    print("ACN-Data API Connection Test")
    print("=" * 50)
    token = getpass.getpass("Paste your API token (hidden): ").strip()

    if not token:
        print("No token entered. Exiting.")
        return

    print(f"\nToken length : {len(token)} characters")
    print("Testing auth methods...\n")

    methods = [
        ("Basic auth  token:''     ", dict(auth=(token, ""))),
        ("Basic auth  BPALAN:token ", dict(auth=("BPALAN", token))),
        ("Basic auth  token:BPALAN ", dict(auth=(token, "BPALAN"))),
        ("Bearer header             ", dict(headers={"Authorization": f"Bearer {token}"})),
        ("Token header              ", dict(headers={"Authorization": f"Token {token}"})),
        ("X-API-Key header          ", dict(headers={"X-API-Key": token})),
    ]

    success = False
    for label, kwargs in methods:
        if test(label, **kwargs):
            success = True
            print(f"\nWorking method: {label.strip()}")
            break
        
    if not success:
        print("\nAll methods returned 401.")
        print("Possible causes:")
        print("  1. Token was pasted with extra whitespace — try again carefully")
        print("  2. Token was not activated on the ACN website yet")
        print("  3. Token was already rotated/revoked")
        print("  4. Account registration is still pending approval")

if __name__ == "__main__":
    main()
