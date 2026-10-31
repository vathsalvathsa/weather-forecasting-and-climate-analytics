"""Central configuration: paths, API endpoints, variables, and the city sample.

Design note (global scope):
We do NOT train one model per city for every city on Earth. We train on a
climatically diverse SAMPLE of cities, with latitude/longitude/elevation as
features, and test generalisation on cities the model never saw (leave-cities-out).
Any other city/state is handled at run time through geocoding.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# 1991-2020 is the WMO standard climatological normal period, so starting in
# 1991 gives a defensible baseline. ERA5 is published with a few days' delay.
START_DATE = "1991-01-01"
ARCHIVE_LAG_DAYS = 7

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Variables we are confident exist; the probe script checks the optional ones.
CORE_DAILY = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max", "weather_code",
]
OPTIONAL_DAILY = [
    "relative_humidity_2m_mean", "surface_pressure_mean", "cloud_cover_mean",
    "dew_point_2m_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]
ACCEPTED_VARS_FILE = DATA_RAW / "_accepted_variables.json"
CITY_META_FILE = DATA_RAW / "_cities_meta.csv"

# Approximate city-centre coordinates; the API snaps to its own grid and
# returns elevation.
CITIES = [
    # Asia
    {"name": "Delhi", "country": "India", "lat": 28.61, "lon": 77.21, "region": "Asia"},
    {"name": "Mumbai", "country": "India", "lat": 19.08, "lon": 72.88, "region": "Asia"},
    {"name": "Chennai", "country": "India", "lat": 13.08, "lon": 80.27, "region": "Asia"},
    {"name": "Kolkata", "country": "India", "lat": 22.57, "lon": 88.36, "region": "Asia"},
    {"name": "Bengaluru", "country": "India", "lat": 12.97, "lon": 77.59, "region": "Asia"},
    {"name": "Shimla", "country": "India", "lat": 31.10, "lon": 77.17, "region": "Asia"},
    {"name": "Leh", "country": "India", "lat": 34.16, "lon": 77.58, "region": "Asia"},
    {"name": "Dhaka", "country": "Bangladesh", "lat": 23.81, "lon": 90.41, "region": "Asia"},
    {"name": "Karachi", "country": "Pakistan", "lat": 24.86, "lon": 67.01, "region": "Asia"},
    {"name": "Singapore", "country": "Singapore", "lat": 1.35, "lon": 103.82, "region": "Asia"},
    {"name": "Bangkok", "country": "Thailand", "lat": 13.76, "lon": 100.50, "region": "Asia"},
    {"name": "Jakarta", "country": "Indonesia", "lat": -6.21, "lon": 106.85, "region": "Asia"},
    {"name": "Tokyo", "country": "Japan", "lat": 35.68, "lon": 139.69, "region": "Asia"},
    {"name": "Beijing", "country": "China", "lat": 39.90, "lon": 116.41, "region": "Asia"},
    {"name": "Seoul", "country": "South Korea", "lat": 37.57, "lon": 126.98, "region": "Asia"},
    {"name": "Dubai", "country": "UAE", "lat": 25.20, "lon": 55.27, "region": "Asia"},
    {"name": "Riyadh", "country": "Saudi Arabia", "lat": 24.71, "lon": 46.68, "region": "Asia"},
    {"name": "Ulaanbaatar", "country": "Mongolia", "lat": 47.89, "lon": 106.91, "region": "Asia"},
    # Europe
    {"name": "London", "country": "UK", "lat": 51.51, "lon": -0.13, "region": "Europe"},
    {"name": "Paris", "country": "France", "lat": 48.86, "lon": 2.35, "region": "Europe"},
    {"name": "Berlin", "country": "Germany", "lat": 52.52, "lon": 13.40, "region": "Europe"},
    {"name": "Madrid", "country": "Spain", "lat": 40.42, "lon": -3.70, "region": "Europe"},
    {"name": "Rome", "country": "Italy", "lat": 41.90, "lon": 12.50, "region": "Europe"},
    {"name": "Moscow", "country": "Russia", "lat": 55.76, "lon": 37.62, "region": "Europe"},
    {"name": "Reykjavik", "country": "Iceland", "lat": 64.15, "lon": -21.94, "region": "Europe"},
    {"name": "Istanbul", "country": "Turkey", "lat": 41.01, "lon": 28.98, "region": "Europe"},
    # Africa
    {"name": "Cairo", "country": "Egypt", "lat": 30.04, "lon": 31.24, "region": "Africa"},
    {"name": "Lagos", "country": "Nigeria", "lat": 6.52, "lon": 3.38, "region": "Africa"},
    {"name": "Nairobi", "country": "Kenya", "lat": -1.29, "lon": 36.82, "region": "Africa"},
    {"name": "Johannesburg", "country": "South Africa", "lat": -26.20, "lon": 28.05, "region": "Africa"},
    {"name": "Cape Town", "country": "South Africa", "lat": -33.92, "lon": 18.42, "region": "Africa"},
    # North America
    {"name": "New York", "country": "USA", "lat": 40.71, "lon": -74.01, "region": "North America"},
    {"name": "Los Angeles", "country": "USA", "lat": 34.05, "lon": -118.24, "region": "North America"},
    {"name": "Chicago", "country": "USA", "lat": 41.88, "lon": -87.63, "region": "North America"},
    {"name": "Houston", "country": "USA", "lat": 29.76, "lon": -95.37, "region": "North America"},
    {"name": "Phoenix", "country": "USA", "lat": 33.45, "lon": -112.07, "region": "North America"},
    {"name": "Miami", "country": "USA", "lat": 25.76, "lon": -80.19, "region": "North America"},
    {"name": "Anchorage", "country": "USA", "lat": 61.22, "lon": -149.90, "region": "North America"},
    {"name": "Toronto", "country": "Canada", "lat": 43.65, "lon": -79.38, "region": "North America"},
    {"name": "Mexico City", "country": "Mexico", "lat": 19.43, "lon": -99.13, "region": "North America"},
    # South America
    {"name": "Sao Paulo", "country": "Brazil", "lat": -23.55, "lon": -46.63, "region": "South America"},
    {"name": "Manaus", "country": "Brazil", "lat": -3.12, "lon": -60.02, "region": "South America"},
    {"name": "Buenos Aires", "country": "Argentina", "lat": -34.60, "lon": -58.38, "region": "South America"},
    {"name": "Lima", "country": "Peru", "lat": -12.05, "lon": -77.04, "region": "South America"},
    {"name": "Bogota", "country": "Colombia", "lat": 4.71, "lon": -74.07, "region": "South America"},
    {"name": "Santiago", "country": "Chile", "lat": -33.45, "lon": -70.67, "region": "South America"},
    # Oceania
    {"name": "Sydney", "country": "Australia", "lat": -33.87, "lon": 151.21, "region": "Oceania"},
    {"name": "Perth", "country": "Australia", "lat": -31.95, "lon": 115.86, "region": "Oceania"},
    {"name": "Darwin", "country": "Australia", "lat": -12.46, "lon": 130.84, "region": "Oceania"},
    {"name": "Auckland", "country": "New Zealand", "lat": -36.85, "lon": 174.76, "region": "Oceania"},
]

# 10 climatically contrasting cities for the first download.
PILOT = ["Delhi", "London", "Singapore", "Sydney", "New York",
         "Cairo", "Sao Paulo", "Ulaanbaatar", "Nairobi", "Mumbai"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")
