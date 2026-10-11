"""
fetch_open_meteo_weather.py
Fetches authentic 2024 historical weather data for all 8 Indian retail cities
directly from Open-Meteo Historical Archive API (Free, open-source, ECMWF ERA5 data).
Saves output to datasets/historical_weather_2024.csv.
"""

import os
import time
import requests
import pandas as pd

CITIES = {
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
}

START_DATE = "2024-01-01"
END_DATE = "2024-12-30"


def fetch_city_weather(city_name, coords):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={coords['lat']}&longitude={coords['lon']}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily=temperature_2m_mean,precipitation_sum"
        f"&timezone=Asia/Kolkata"
    )
    print(f"[Open-Meteo] Requesting 2024 archive for {city_name} (lat: {coords['lat']}, lon: {coords['lon']})...")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    rains = daily.get("precipitation_sum", [])

    records = []
    for d, t, r in zip(dates, temps, rains):
        records.append({
            "Date": d,
            "City": city_name,
            "Avg_Temperature_C": round(t, 2) if t is not None else None,
            "Precipitation_mm": round(r, 2) if r is not None else 0.0,
        })
    return pd.DataFrame(records)


def fetch_all_weather(output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    print("=" * 60)
    print("FETCHING REAL 2024 HISTORICAL WEATHER FROM OPEN-METEO")
    print("=" * 60)

    for city, coords in CITIES.items():
        try:
            df_city = fetch_city_weather(city, coords)
            all_dfs.append(df_city)
            print(f" -> Successfully fetched {len(df_city)} days for {city}.")
            time.sleep(0.5)  # Friendly delay
        except Exception as e:
            print(f" [ERROR] Failed to fetch for {city}: {e}")
            raise e

    combined = pd.concat(all_dfs, ignore_index=True)
    out_file = os.path.join(output_dir, "historical_weather_2024.csv")
    combined.to_csv(out_file, index=False)
    print("=" * 60)
    print(f"SUCCESS: Saved {len(combined)} authentic weather records to -> {out_file}")
    print("=" * 60)
    return combined


if __name__ == "__main__":
    fetch_all_weather()

