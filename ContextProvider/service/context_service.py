from __future__ import annotations
import hashlib
import json
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
from ContextProvider.model.context_models import (
    LocationHint,
    LocationResolved,
    DateTimeContext,
    EnvironmentContext,
    LocaleContext,
    ContextEnvelope,
    Holiday,
    WeatherContext,
)
from ContextProvider.service.holiday_service import fetch_holidays
from ContextProvider.service.weather_service import fetch_current_weather

TIMEZONE = "Europe/Berlin"

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
    For now we fall back to Karlsruhe if nothing is provided.
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


async def build_snapshot(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> ContextEnvelope:
    """
    Build a full environment context snapshot wrapped in a ContextEnvelope.

    - Resolves location (hint or default)
    - Derives date/time context (timezone, weekday, partOfDay)
    - Fetches holidays (if countryCode is known)
    - Fetches weather (if lat/lon are known)
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)

    location = _resolve_location(location_hint)
    locale = _parse_locale(accept_language)

    dt = DateTimeContext(
        iso=now.isoformat(),
        timezone=TIMEZONE,
        weekday=now.strftime("%A"),
        partOfDay=_part_of_day(now.hour),
    )

    # Holidays
    holidays: list[Holiday] = []
    if location.countryCode:
        holidays = await fetch_holidays(location.countryCode, now.year)

    # Weather
    weather: Optional[WeatherContext] = await fetch_current_weather(location.lat, location.lon)

    env = EnvironmentContext(
        location=location,
        dateTime=dt,
        holidays=holidays,
        weather=weather,
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

    # No hash provided or already up to date -> empty delta
    if not since_hash or since_hash == snapshot.hash:
        return ContextEnvelope(
            type="context-delta",
            version=snapshot.version,
            producedAt=snapshot.producedAt,
            hash=snapshot.hash,
            data={},  # empty payload indicates "no change"
        )

    # Hash differs -> send full context as delta
    return ContextEnvelope(
        type="context-delta",
        version=snapshot.version,
        producedAt=snapshot.producedAt,
        hash=snapshot.hash,
        data=snapshot.data,
    )