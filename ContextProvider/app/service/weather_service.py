from __future__ import annotations
from typing import Optional, List
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import httpx
from ..model.context_models import WeatherContext, WeatherForecastPoint, WeatherTomorrow

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


async def fetch_hourly_forecast(
    lat: float,
    lon: float,
    hours: int,
    timezone: str = "Europe/Berlin",
) -> List[WeatherForecastPoint]:
    """
    Fetch an hourly weather forecast for the next `hours` hours
    for a given location from Open-Meteo.
    Returns a list of WeatherForecastPoint, sorted by time ascending.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": timezone,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params)
            response.raise_for_status()
        except httpx.HTTPError:
            return []

        data = response.json()
        hourly = data.get("hourly") or {}

        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        precs = hourly.get("precipitation") or []
        winds = hourly.get("wind_speed_10m") or []

        length = min(len(times), len(temps), len(precs), len(winds))
        times = times[:length]
        temps = temps[:length]
        precs = precs[:length]
        winds = winds[:length]

        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        horizon = now + timedelta(hours=hours)

        forecast_points: List[WeatherForecastPoint] = []

        for t_str, t_val, p_val, w_val in zip(times, temps, precs, winds):
            try:
                t_dt = datetime.fromisoformat(t_str)
            except ValueError:
                continue

            # Attach timezone if naive
            if t_dt.tzinfo is None:
                t_dt = t_dt.replace(tzinfo=tz)

            if t_dt < now or t_dt > horizon:
                continue

            forecast_points.append(
                WeatherForecastPoint(
                    timestamp=t_dt.isoformat(),
                    date=t_dt.date().isoformat(),
                    time=t_dt.strftime("%H:%M"),
                    weekday=t_dt.strftime("%A"),
                    temperatureC=_num_or_none(t_val),
                    windKph=_num_or_none(w_val),
                    precipitationMm=_num_or_none(p_val),
                )
            )

        forecast_points.sort(key=lambda fp: fp.timestamp)
        return forecast_points


async def fetch_tomorrow_weather(
    lat: float,
    lon: float,
    timezone: str = "Europe/Berlin",
) -> Optional[WeatherTomorrow]:
    """
    Fetch a daily weather summary for tomorrow from Open-Meteo.
    Uses daily max/min temperature, precipitation sum and max wind speed.
    """
    tz = ZoneInfo(timezone)
    today = datetime.now(tz).date()
    tomorrow: date = today + timedelta(days=1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": timezone,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        data = response.json()
        daily = data.get("daily") or {}

        times = daily.get("time") or []
        t_max = daily.get("temperature_2m_max") or []
        t_min = daily.get("temperature_2m_min") or []
        prec = daily.get("precipitation_sum") or []
        wind_max = daily.get("wind_speed_10m_max") or []

        length = min(len(times), len(t_max), len(t_min), len(prec), len(wind_max))
        times = times[:length]
        t_max = t_max[:length]
        t_min = t_min[:length]
        prec = prec[:length]
        wind_max = wind_max[:length]

        target_date_str = tomorrow.isoformat()

        for d_str, d_tmax, d_tmin, d_prec, d_wmax in zip(times, t_max, t_min, prec, wind_max):
            if d_str != target_date_str:
                continue

            return WeatherTomorrow(
                date=d_str,
                temperatureMaxC=_num_or_none(d_tmax),
                temperatureMinC=_num_or_none(d_tmin),
                precipitationMm=_num_or_none(d_prec),
                windKphMax=_num_or_none(d_wmax),
            )

        # No matching tomorrow entry found
        return None


def _num_or_none(value) -> Optional[float]:
    """Convert value to float if possible, otherwise return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
