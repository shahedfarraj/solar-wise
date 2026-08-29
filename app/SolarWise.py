import json
import time
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

import numpy as np
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

st.set_page_config(page_title="SolarWise Jordan", page_icon="☀️")

# Load the trained model and 2025 weather features for every covered location
model = joblib.load("solarwise_model.pkl")
location_features = joblib.load("location_2025_features.pkl")

FEATURE_COLS = ['lat', 'lon', 'elevation', 'clear_sky_irradiance(kwh/m2)', 'temperature(c)', 'humidity(percentage)',
                'cloud_cover(percentage)', 'wind_speed(m/s)', 'precipitation(mm)', 'day_of_year_sin', 'day_of_year_cos']

LOCATIONS = location_features[['location_id', 'nearest_place', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)

# A grid point every ~28km covers all of Jordan (border-filtered) - a real
# input farther than this from every covered point isn't actually in Jordan.
MAX_COVERAGE_KM = 40

# Standard reference system - same panel assumptions as the training notebook
# (plan.md Section 4). Not user-adjustable: every user sees the same
# reference system so results are comparable and the app stays a one-question
# tool ("is solar worth it"), not a system-sizing calculator.
SYSTEM_SIZE_KWP = 5.0
PANEL_WATTAGE_W = 450
PANEL_COUNT = round(SYSTEM_SIZE_KWP * 1000 / PANEL_WATTAGE_W)
PANEL_EFFICIENCY_PCT = 21
PANEL_AREA_M2 = 2.1
PANEL_LENGTH_M = 2.1
PANEL_WIDTH_M = 1.0
TOTAL_AREA_M2 = PANEL_COUNT * PANEL_LENGTH_M * PANEL_WIDTH_M
SQUARE_LAYOUT_SIDE_M = TOTAL_AREA_M2 ** 0.5

# JEPCO residential tiered tariff (JOD per kWh)
TARIFF_TIERS = [(300, 0.050), (600, 0.100), (float('inf'), 0.200)]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def bill_jod(monthly_kwh):
    """Progressive tiered bill for a given monthly usage."""
    remaining = monthly_kwh
    prev_cap = 0
    total = 0.0
    for cap, price in TARIFF_TIERS:
        tier_kwh = max(0, min(remaining, cap - prev_cap))
        total += tier_kwh * price
        remaining -= tier_kwh
        prev_cap = cap
        if remaining <= 0:
            break
    return total


def kwh_from_bill(bill_amount_jod):
    """Invert the tiered tariff: how many kWh does this bill amount correspond to?"""
    t1_cap_jod = 300 * 0.050
    t2_cap_jod = t1_cap_jod + 300 * 0.100
    if bill_amount_jod <= t1_cap_jod:
        return bill_amount_jod / 0.050
    elif bill_amount_jod <= t2_cap_jod:
        return 300 + (bill_amount_jod - t1_cap_jod) / 0.100
    else:
        return 600 + (bill_amount_jod - t2_cap_jod) / 0.200


def _nominatim_search(query):
    params = urlencode({"q": query, "format": "json", "limit": 5, "countrycodes": "jo"})
    req = Request(f"{NOMINATIM_URL}?{params}", headers={"User-Agent": "solarwise-jordan-app", "Accept-Language": "en"})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (HTTPError, URLError):
        return []


def geocode_address(address):
    """Turn a typed address into (lat, lon) via OpenStreetMap Nominatim.

    Nominatim's top-ranked hit isn't always the actual place (e.g. a river or
    a governorate boundary can outrank the real city/town) - only trust
    results explicitly classed as a populated place, falling back to the
    plain top hit if nothing better is found.
    """
    results = _nominatim_search(f"{address}, Jordan")
    place_hits = [r for r in results if r.get("class") == "place"]
    if place_hits:
        best = max(place_hits, key=lambda r: float(r.get("importance", 0)))
        return float(best["lat"]), float(best["lon"])
    if results:
        return float(results[0]["lat"]), float(results[0]["lon"])
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_location(lat, lon):
    """Snap a user's coordinates to the closest pre-fetched grid location."""
    distances = haversine_km(lat, lon, LOCATIONS['lat'].values, LOCATIONS['lon'].values)
    idx = distances.argmin()
    return LOCATIONS.iloc[idx], distances[idx]


def monthly_generation(location_id):
    """Predict daily kWh/kWp for the location's 2025 weather, aggregate to monthly kWh at the standard system size."""
    lf = location_features[location_features['location_id'] == location_id].copy()
    lf['predicted_kwh'] = model.predict(lf[FEATURE_COLS]) * SYSTEM_SIZE_KWP
    return lf.groupby('month')['predicted_kwh'].sum()


def location_picker():
    """Returns (lat, lon) from whichever input method the user used, or None."""
    method = st.radio("How do you want to give us your location?",
                       ["Type an address", "Pick on the map", "Detect my location"], horizontal=True)

    if method == "Type an address":
        address = st.text_input("Address or area (e.g. 'Jabal Amman, Amman')")
        if address:
            coords = geocode_address(address)
            if coords is None:
                st.error("Couldn't find that address. Try a nearby city or landmark name instead.")
                return None
            st.caption(f"Found: {coords[0]:.4f}, {coords[1]:.4f}")
            return coords
        return None

    elif method == "Pick on the map":
        m = folium.Map(location=[31.9, 36.0], zoom_start=7)
        m.add_child(folium.LatLngPopup())
        map_state = st_folium(m, height=400, width=700)
        if map_state and map_state.get("last_clicked"):
            lat = map_state["last_clicked"]["lat"]
            lon = map_state["last_clicked"]["lng"]
            st.caption(f"Selected: {lat:.4f}, {lon:.4f}")
            return lat, lon
        st.info("Click anywhere on the map to select your location.")
        return None

    else:  # Detect my location
        # This component must be called on every rerun (not just inside a
        # button click) so it can receive the browser's answer once the
        # permission prompt resolves - gating it behind a one-shot button
        # click means the component never gets mounted again to hear back.
        st.caption("Your browser will ask for permission to share your location.")
        loc = get_geolocation(component_key="detect_location")
        if loc and "coords" in loc:
            lat = loc["coords"]["latitude"]
            lon = loc["coords"]["longitude"]
            st.caption(f"Detected: {lat:.4f}, {lon:.4f}")
            return lat, lon
        st.warning("Waiting for location access - allow it in your browser's prompt, or pick another method above.")
        return None


def main():
    st.title("SolarWise Jordan ☀️")
    st.markdown("""
    **Is solar wise for me?**

    Tell us your location in Jordan and what you pay for electricity, and we'll estimate how much
    a standard solar system would generate there, compare it to your usage, and tell you if it's worth it.
    """)

    with st.expander("What system are we calculating for?"):
        st.markdown(f"""
        We use the same standard solar setup for everyone, so results are fair to compare:
        - A **{SYSTEM_SIZE_KWP:.0f} kW system** ({PANEL_COUNT} panels) - the size most homes in Jordan install
        - Each panel is about **{PANEL_LENGTH_M:.1f}m x {PANEL_WIDTH_M:.1f}m** - roughly the size of a door
        - {PANEL_COUNT} panels x {PANEL_LENGTH_M:.1f}m x {PANEL_WIDTH_M:.1f}m = **{TOTAL_AREA_M2:.1f} m² total area** (roughly {SQUARE_LAYOUT_SIDE_M:.1f}m x {SQUARE_LAYOUT_SIDE_M:.1f}m if arranged in a square layout)
        - Panels turn about **{PANEL_EFFICIENCY_PCT}%** of sunlight into electricity
        - Panels work a little less well on very hot days - we already factor that in
        """)

    lat_lon = location_picker()

    bill_amount = st.number_input(
        "How much is your average monthly electricity bill? (JOD)",
        min_value=1.0, max_value=500.0, value=25.0, step=1.0,
        help="Most people remember what they pay, not their kWh usage - we convert it for you using JEPCO's tariff.",
    )
    monthly_usage = kwh_from_bill(bill_amount)
    st.caption(f"That's about {monthly_usage:.0f} kWh/month at JEPCO's tiered rate.")

    if lat_lon is None:
        st.stop()

    lat, lon = lat_lon
    if st.button("Is solar wise for me?"):
        location, distance_km = nearest_location(lat, lon)

        if distance_km > MAX_COVERAGE_KM:
            st.error("Sorry, our business doesn't cover your area.")
            return

        st.caption(f"Using weather data for {location['nearest_place']} ({distance_km:.1f} km away)")

        gen_by_month = monthly_generation(location['location_id'])
        avg_gen = gen_by_month.mean()

        net_usage = max(0, monthly_usage - avg_gen)
        bill_without = bill_jod(monthly_usage)
        bill_with = bill_jod(net_usage)
        savings = bill_without - bill_with
        coverage = min(100, avg_gen / monthly_usage * 100)

        if coverage >= 100:
            st.success("**Worth it!** Your system would cover your full usage.")
        elif coverage >= 50:
            st.warning(f"**Partially worth it** — covers about {coverage:.0f}% of your usage.")
        else:
            st.error(f"**Probably not enough** — covers only about {coverage:.0f}% of your usage. Consider a bigger system.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted generation", f"{avg_gen:.0f} kWh/month")
        m2.metric("Coverage of your usage", f"{coverage:.0f}%")
        m3.metric("Monthly savings", f"{savings:.2f} JOD")

        st.caption(f"Bill without solar: {bill_without:.2f} JOD → with solar: {bill_with:.2f} JOD (JEPCO tiered tariff)")

        chart_df = gen_by_month.reset_index().rename(columns={'predicted_kwh': 'Generated'})
        chart_df['Used'] = monthly_usage

        fig = go.Figure()
        fig.add_bar(x=chart_df['month'], y=chart_df['Generated'], name='Generated', marker_color='#1f77b4')
        fig.add_bar(x=chart_df['month'], y=chart_df['Used'], name='Used', marker_color='#d62728')
        fig.update_layout(
            barmode='group',
            title=f'Predicted generation vs. your usage - {SYSTEM_SIZE_KWP:.0f} kWp system',
            xaxis_title='Month', yaxis_title='kWh',
        )
        st.plotly_chart(fig, use_container_width=True)


main()
