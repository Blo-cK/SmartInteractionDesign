from __future__ import annotations
from typing import Optional

import httpx

from ContextProvider.model.context_models import WeatherContext


OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"


async def fetch_current_weather(lat: float, lon: float) -> Optional[WeatherContext]:
    """
    Fetch current weather for a given location from Open-Meteo.
    Docs: https://open-meteo.com/en/docs
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,wind_speed_10m",
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        data = response.json()
        current = data.get("current") or {}

        return WeatherContext(
            temperatureC=_num_or_none(current.get("temperature_2m")),
            windKph=_num_or_none(current.get("wind_speed_10m")),
            precipitationMm=_num_or_none(current.get("precipitation")),
            summary=None,
        )


def _num_or_none(value) -> Optional[float]:
    """Convert value to float if possible, otherwise return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
