from __future__ import annotations
from typing import Optional
from datetime import datetime, date
from zoneinfo import ZoneInfo
import httpx
from ContextProvider.app.model.context_models import DaylightContext

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"

async def fetch_daylight_context(
    lat: float,
    lon: float,
    timezone: str = "Europe/Berlin",
) -> Optional[DaylightContext]:
    """
    Retrieve sunrise and sunset times for the current day and derive
    a small daylight summary for the given location.
    """
    tz = ZoneInfo(timezone)
    today: date = datetime.now(tz).date()
    today_str = today.isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "sunrise,sunset",
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
        sunrises = daily.get("sunrise") or []
        sunsets = daily.get("sunset") or []

        length = min(len(times), len(sunrises), len(sunsets))
        times = times[:length]
        sunrises = sunrises[:length]
        sunsets = sunsets[:length]

        sunrise_str: Optional[str] = None
        sunset_str: Optional[str] = None

        # Use sunrise/sunset entry for today if available
        for t, s_r, s_s in zip(times, sunrises, sunsets):
            if t == today_str:
                sunrise_str = s_r
                sunset_str = s_s
                break

        if not sunrise_str or not sunset_str:
            return None

        try:
            sunrise_dt = datetime.fromisoformat(sunrise_str)
            sunset_dt = datetime.fromisoformat(sunset_str)
        except ValueError:
            return None

        # Attach timezone information if missing
        if sunrise_dt.tzinfo is None:
            sunrise_dt = sunrise_dt.replace(tzinfo=tz)
        if sunset_dt.tzinfo is None:
            sunset_dt = sunset_dt.replace(tzinfo=tz)

        now = datetime.now(tz)

        is_daylight = sunrise_dt <= now <= sunset_dt

        minutes_until_sunrise: Optional[int] = None
        minutes_until_sunset: Optional[int] = None

        if now < sunrise_dt:
            minutes_until_sunrise = int((sunrise_dt - now).total_seconds() // 60)
        if now < sunset_dt:
            minutes_until_sunset = int((sunset_dt - now).total_seconds() // 60)

        return DaylightContext(
            sunrise=sunrise_dt.isoformat(),
            sunset=sunset_dt.isoformat(),
            isDaylight=is_daylight,
            minutesUntilSunrise=minutes_until_sunrise,
            minutesUntilSunset=minutes_until_sunset,
        )
