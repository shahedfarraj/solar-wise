"""
Single-pass, full-country dataset build.

1. Generate a grid of points spanning Jordan's bounding box.
2. Classify each point against Jordan's real border via Nominatim reverse
   geocoding (checks the actual polygon, not a nearest-known-city guess) -
   an earlier attempt using the offline reverse_geocoder library dropped
   real Jordanian territory (the whole eastern desert past ~37.5E, and the
   far north past ~32.7N) because its sparse place database in empty desert
   has its nearest known city across the Iraq/Saudi border even when the
   point itself is inside Jordan. Nominatim checks the real polygon instead.
3. Fetch NASA POWER daily weather data (2020-2025) at every kept point, so
   the trained model - and the app - has real, pre-fetched coverage
   everywhere in the country instead of 13 fixed cities.
4. Combine into one final CSV, with a small amount of realistic messiness
   injected (duplicate rows, missing-value markers, a few outliers) so the
   dataset has real preprocessing work to do.
"""

import csv
import json
import random
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "final_dataset.csv"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "CLOUD_AMT",
    "WS10M",
    "PRECTOTCORR",
]

START = "20200101"
END = "20251231"
COMMUNITY = "RE"

# Jordan's bounding box - grid spacing chosen to be finer than NASA POWER's
# native irradiance resolution (~0.5 deg) so ground-weather variables
# (temperature, humidity, wind, precipitation), which vary more locally,
# are captured with real spatial detail across the whole country.
LAT_MIN, LAT_MAX = 29.2, 33.4
LON_MIN, LON_MAX = 34.9, 39.3
STEP = 0.25


def build_grid() -> list[tuple[float, float]]:
    points = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            points.append((round(lat, 3), round(lon, 3)))
            lon += STEP
        lat += STEP
    return points


def reverse_geocode(lat: float, lon: float, retries: int = 3) -> dict:
    params = urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1})
    req = Request(f"{NOMINATIM_URL}?{params}", headers={
        "User-Agent": "solarwise-jordan-project",
        "Accept-Language": "en",
    })
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError) as e:
            print(f"  reverse-geocode attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def filter_to_jordan(points: list[tuple[float, float]]) -> list[dict]:
    kept = []
    for i, (lat, lon) in enumerate(points, 1):
        result = reverse_geocode(lat, lon)
        time.sleep(1)  # Nominatim rate limit
        address = result.get("address", {})
        # country_code is a stable ISO code regardless of Nominatim's response
        # language (it returned Arabic address fields by default here, so
        # matching on address['country'] == 'Jordan' silently matched nothing).
        if address.get("country_code") != "jo":
            continue
        place = (address.get("town") or address.get("village") or address.get("city")
                 or address.get("county") or address.get("state") or "Jordan")
        # Nominatim labels most desert areas "X Sub-District" - "District" is
        # the more commonly understood term for a general audience.
        place = place.replace("Sub-District", "District")
        kept.append({"lat": lat, "lon": lon, "nearest_place": place})
        if i % 20 == 0:
            print(f"  ...classified {i}/{len(points)} candidate points")
    return kept


def fetch_weather(lat: float, lon: float, retries: int = 3) -> dict:
    url = (
        f"{NASA_URL}?parameters={','.join(PARAMETERS)}"
        f"&community={COMMUNITY}&longitude={lon}&latitude={lat}"
        f"&start={START}&end={END}&format=JSON"
    )
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError) as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def main():
    grid = build_grid()
    print(f"Candidate grid: {len(grid)} points over Jordan's bounding box")

    jordan_points = filter_to_jordan(grid)
    print(f"Kept {len(jordan_points)} points actually inside Jordan (offline border check)")

    rows = []
    for i, pt in enumerate(jordan_points, 1):
        lat, lon = pt["lat"], pt["lon"]
        print(f"[{i}/{len(jordan_points)}] fetching ({lat}, {lon}) near {pt['nearest_place']}...")
        data = fetch_weather(lat, lon)
        time.sleep(1)  # be polite to the API

        geometry = data.get("geometry", {})
        coords = geometry.get("coordinates", [lon, lat, None])
        elevation = coords[2]
        param_data = data.get("properties", {}).get("parameter", {})
        if not param_data:
            print(f"  skipping ({lat},{lon}): no parameter data")
            continue
        dates = sorted(next(iter(param_data.values())).keys())

        for date in dates:
            row = {
                "lat": lat, "lon": lon, "elevation": elevation,
                "nearest_place": pt["nearest_place"], "date": date,
            }
            for p in PARAMETERS:
                row[p] = param_data.get(p, {}).get(date)
            rows.append(row)

    print(f"\nCollected {len(rows)} rows. Injecting realistic messiness...")
    rng = random.Random(42)

    # Duplicate ~1% of rows
    n_dupes = max(1, len(rows) // 100)
    rows.extend(rng.sample(rows, n_dupes))

    # Mark ~0.5% of weather values as missing (-999, NASA POWER's own convention)
    n_missing = max(1, len(rows) // 200)
    for _ in range(n_missing):
        row = rng.choice(rows)
        col = rng.choice(PARAMETERS)
        row[col] = -999

    # Plant a handful of physically implausible outliers
    n_outliers = 5
    for _ in range(n_outliers):
        row = rng.choice(rows)
        row["T2M"] = rng.choice([150.0, -80.0])

    rng.shuffle(rows)

    fieldnames = ["lat", "lon", "elevation", "nearest_place", "date"] + PARAMETERS
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Final dataset: {OUT_PATH} ({len(rows)} rows, {len(jordan_points)} locations)")


if __name__ == "__main__":
    main()
