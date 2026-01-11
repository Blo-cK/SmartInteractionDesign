from __future__ import annotations

import hashlib
import json
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from ..model.context_models import (
    LocationHint,
    LocationResolved,
    DateTimeContext,
    EnvironmentContext,
    LocaleContext,
    ContextEnvelope,
    Holiday,
    WeatherContext,
    UpcomingHoliday,
    WeatherForecastPoint,
    WeatherTomorrow,
)
from .holiday_service import fetch_holidays
from .weather_service import (
    fetch_current_weather,
    fetch_hourly_forecast,
    fetch_tomorrow_weather,
)
from .location_detector_service import detect_server_location

TIMEZONE = "Europe/Berlin"

# Will be filled once by IP-based geolocation and then reused
SERVER_LOCATION: Optional[LocationResolved] = None


def _part_of_day(hour: int) -> str:
    """Derive a coarse-grained part of day from hour (0-23)."""
    if hour < 5:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _parse_locale(accept_language: Optional[str]) -> LocaleContext:
    """
    Parse Accept-Language header into a simple LocaleContext.
    Example: "de-DE,de;q=0.9,en-US;q=0.8" -> language="de", locale="de-DE"
    """
    header = accept_language or "en-US"
    primary = header.split(",")[0].strip()
    language = primary.split("-")[0]
    return LocaleContext(language=language, locale=primary)


def _resolve_location(hint: Optional[LocationHint]) -> LocationResolved:
    """
    Resolve location using an optional LocationHint.

    - If a hint with lat/lon is provided: use that.
    - Otherwise: fall back to a default location (Karlsruhe).
      The default may later be overridden via auto-detected server location.
    """
    if hint and hint.lat is not None and hint.lon is not None:
        return LocationResolved(
            lat=hint.lat,
            lon=hint.lon,
            city=hint.city or "Karlsruhe",
            countryCode=hint.countryCode or "DE",
            region=hint.region,
        )

    # Default location (can be changed if needed)
    return LocationResolved(
        lat=49.0069,
        lon=8.4037,
        city="Karlsruhe",
        countryCode="DE",
        region=None,
    )


def _stable_hash(obj: object) -> str:
    """
    Compute a stable hash (sha256, first 16 hex chars) over a JSON-serialised object.
    This is used to detect changes for delta-style updates.
    """
    json_str = json.dumps(obj, sort_keys=True, default=str)
    h = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return h[:16]


def _compute_upcoming_holidays(
    all_holidays: list[Holiday],
    today: date,
    window_days: int = 14,
) -> list[UpcomingHoliday]:
    """
    From a list of holidays, compute those that are within the next 'window_days' days.
    """
    horizon = today + timedelta(days=window_days)
    upcoming: list[UpcomingHoliday] = []

    for h in all_holidays:
        try:
            holiday_date = date.fromisoformat(h.date)
        except ValueError:
            continue

        if today <= holiday_date <= horizon:
            days_until = (holiday_date - today).days
            upcoming.append(
                UpcomingHoliday(
                    date=h.date,
                    localName=h.localName,
                    countryCode=h.countryCode,
                    daysUntil=days_until,
                )
            )

    upcoming.sort(key=lambda uh: uh.date)
    return upcoming


async def build_snapshot(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> ContextEnvelope:
    """
    Build a full environment context snapshot wrapped in a ContextEnvelope.

    - Resolves location (client hint, auto-detected server location, or default)
    - Derives date/time context (timezone, weekday, partOfDay)
    - Fetches holidays (country-wide + region-specific)
    - Computes upcoming holidays (next 14 days)
    - Fetches current weather, hourly forecast (next 8 hours) and tomorrow's summary
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()

    # Determine effective location hint:
    effective_hint: Optional[LocationHint] = location_hint

    global SERVER_LOCATION
    if effective_hint is None:
        if SERVER_LOCATION is None:
            try:
                auto_loc = await detect_server_location()
                if auto_loc is not None:
                    SERVER_LOCATION = auto_loc
                    print(f"[ContextProvider] Auto-detected server location: {SERVER_LOCATION}")
            except Exception:
                SERVER_LOCATION = None

        if SERVER_LOCATION is not None:
            effective_hint = LocationHint(
                lat=SERVER_LOCATION.lat,
                lon=SERVER_LOCATION.lon,
                city=SERVER_LOCATION.city,
                countryCode=SERVER_LOCATION.countryCode,
                region=SERVER_LOCATION.region,
            )

    location = _resolve_location(effective_hint)
    locale = _parse_locale(accept_language)

    dt = DateTimeContext(
        iso=now.isoformat(),
        timezone=TIMEZONE,
        weekday=now.strftime("%A"),
        partOfDay=_part_of_day(now.hour),
    )

    # Holidays (mit optionaler Filterung nach Bundesland/Region)
    holidays: list[Holiday] = []
    upcoming_holidays: list[UpcomingHoliday] = []
    if location.countryCode:
        try:
            all_holidays = await fetch_holidays(location.countryCode, now.year)

            if location.region:
                base_region = location.region  # can be "DE-BW", "BW", or "Baden-Württemberg"

                candidate_codes: list[str] = []
                if "-" in base_region:
                    candidate_codes.append(base_region)
                else:
                    candidate_codes.append(f"{location.countryCode}-{base_region}")

                def matches_region(h: Holiday) -> bool:
                    if h.regions is None:
                        return True
                    return any(code in (h.regions or []) for code in candidate_codes)

                holidays = [h for h in all_holidays if matches_region(h)]
            else:
                holidays = all_holidays

            upcoming_holidays = _compute_upcoming_holidays(holidays, today, window_days=14)

        except Exception:
            holidays = []
            upcoming_holidays = []

    # Weather: current, forecast next 8 hours, tomorrow summary
    weather_current: Optional[WeatherContext] = None
    weather_forecast: list[WeatherForecastPoint] = []
    weather_tomorrow: Optional[WeatherTomorrow] = None

    try:
        weather_current = await fetch_current_weather(location.lat, location.lon)
    except Exception:
        weather_current = None

    try:
        weather_forecast = await fetch_hourly_forecast(
            location.lat,
            location.lon,
            hours=8,
            timezone=TIMEZONE,
        )
    except Exception:
        weather_forecast = []

    try:
        weather_tomorrow = await fetch_tomorrow_weather(
            location.lat,
            location.lon,
            timezone=TIMEZONE,
        )
    except Exception:
        weather_tomorrow = None

    env = EnvironmentContext(
        location=location,
        dateTime=dt,
        holidays=holidays,
        upcoming_holidays=upcoming_holidays,
        weather_current=weather_current,
        weather_forecast=weather_forecast,
        weather_tomorrow=weather_tomorrow,
        locale=locale,
    )

    produced_at = now.isoformat()
    content_hash = _stable_hash(env.model_dump())

    envelope = ContextEnvelope(
        type="context-snapshot",
        version="1.0",
        producedAt=produced_at,
        hash=content_hash,
        data=env,
    )

    return envelope


async def build_delta(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
    since_hash: Optional[str] = None,
) -> ContextEnvelope:
    """
    Build a delta envelope compared to the given since_hash.

    - If since_hash is missing OR equal to the current hash:
      -> return an empty data object (no changes).
    - Otherwise:
      -> return the full EnvironmentContext as data.
    """
    snapshot = await build_snapshot(
        accept_language=accept_language,
        location_hint=location_hint,
    )

    if not since_hash or since_hash == snapshot.hash:
        return ContextEnvelope(
            type="context-delta",
            version=snapshot.version,
            producedAt=snapshot.producedAt,
            hash=snapshot.hash,
            data={},
        )

    return ContextEnvelope(
        type="context-delta",
        version=snapshot.version,
        producedAt=snapshot.producedAt,
        hash=snapshot.hash,
        data=snapshot.data,
    )
