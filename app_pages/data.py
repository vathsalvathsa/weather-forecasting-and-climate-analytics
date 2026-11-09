"""Cached API calls shared by the pages."""
import streamlit as st

from src import config
from src.live.weather_api import geocode, get_air_quality, get_forecast
from src.live.world_api import get_world_snapshot


@st.cache_data(ttl=3600, show_spinner=False)
def cached_geocode(query):
    return geocode(query)


@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast(lat, lon):
    return get_forecast(lat, lon)


@st.cache_data(ttl=600, show_spinner=False)
def cached_air(lat, lon):
    return get_air_quality(lat, lon)


@st.cache_data(ttl=600, show_spinner="Loading the world map...")
def cached_world():
    return get_world_snapshot(config.CITIES)
