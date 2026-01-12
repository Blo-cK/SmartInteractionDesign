from __future__ import annotations

import hashlib
import json
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from ContextProvider.app.model.context_models import (
    LocationHint,
    LocationResolved,
    DateTimeContext,
    DayMeta,
    EnvironmentContext,
    LocaleContext,
    ContextEnvelope,
    Holiday,
    WeatherContext,
    WeatherForecastPoint,
    WeatherTomorrow,
    DaylightContext,
    PlaceContext,
    ComfortContext,
    EventsContext,
    HolidayDistance,
)
from ContextProvider.app.service.holiday_service import fetch_holidays
from ContextProvider.app.service.weather_service import (
    fetch_current_weather,
    fetch_hourly_forecast,
    fetch_tomorrow_weather,
)
from ContextProvider.app.service.location_detector_service import detect_server_location
from ContextProvider.app.service.daylight_service import fetch_daylight_context
from ContextProvider.app.service.place_service import fetch_place_context
from ContextProvider.app.service.comfort_service import compute_comfort_context
from ContextProvider.app.service.events_service import fetch_local_events_context

TIMEZONE = "Europe/Berlin"

# Filled once by IP-based geolocation and then reused
SERVER_LOCATION: Optional[LocationResolved] = None


def _part_of_day(hour: int) -> str:
    """Map hour of day (0–23) to a coarse part-of-day label."""
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
      This default can be overridden by the detected server location.
    """
    if hint and hint.lat is not None and hint.lon is not None:
        return LocationResolved(
            lat=hint.lat,
            lon=hint.lon,
            city=hint.city or "Karlsruhe",
            countryCode=hint.countryCode or "DE",
            region=hint.region,
        )

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


def _compute_holiday_distances(
    all_holidays: list[Holiday],
    today: date,
) -> tuple[Optional[HolidayDistance], Optional[HolidayDistance], Optional[HolidayDistance]]:
    """
    Compute:
    - lastHoliday: most recent holiday on or before today (daysSince >= 0)
    - nextHoliday: next holiday on or after today (daysUntil >= 0)
    - nearestHoliday: holiday with the smallest absolute distance to today
      (uses either daysSince or daysUntil depending on past/future).
    """
    if not all_holidays:
        return None, None, None

    parsed: list[tuple[date, Holiday]] = []
    for h in all_holidays:
        if not h.date:
            continue
        try:
            d = date.fromisoformat(h.date)
        except ValueError:
            continue
        parsed.append((d, h))

    if not parsed:
        return None, None, None

    past_or_today = [(d, h) for (d, h) in parsed if d <= today]
    future_or_today = [(d, h) for (d, h) in parsed if d >= today]

    last_holiday: Optional[HolidayDistance] = None
    next_holiday: Optional[HolidayDistance] = None
    nearest_holiday: Optional[HolidayDistance] = None

    # Last holiday (<= today) → daysSince
    if past_or_today:
        last_date, last_h = max(past_or_today, key=lambda x: x[0])
        days_ago = (today - last_date).days
        last_holiday = HolidayDistance(
            date=last_h.date,
            localName=last_h.localName,
            countryCode=last_h.countryCode,
            daysSince=days_ago,
            daysUntil=None,
        )

    # Next holiday (>= today) → daysUntil
    if future_or_today:
        next_date, next_h = min(future_or_today, key=lambda x: x[0])
        days_until = (next_date - today).days
        next_holiday = HolidayDistance(
            date=next_h.date,
            localName=next_h.localName,
            countryCode=next_h.countryCode,
            daysSince=None,
            daysUntil=days_until,
        )

    # Nearest holiday in absolute days
    nearest_date, nearest_h = min(
        parsed,
        key=lambda x: abs((x[0] - today).days),
    )
    diff = (nearest_date - today).days

    if diff >= 0:
        nearest_holiday = HolidayDistance(
            date=nearest_h.date,
            localName=nearest_h.localName,
            countryCode=nearest_h.countryCode,
            daysSince=None,
            daysUntil=diff,
        )
    else:
        nearest_holiday = HolidayDistance(
            date=nearest_h.date,
            localName=nearest_h.localName,
            countryCode=nearest_h.countryCode,
            daysSince=abs(diff),
            daysUntil=None,
        )

    return last_holiday, next_holiday, nearest_holiday


def _compute_day_meta(
    today: date,
    weekday_index: int,
    holidays: list[Holiday],
) -> DayMeta:
    """
    Compute isWeekend, isPublicHolidayToday and a simple isBridgeDay heuristic.
    """
    is_weekend = weekday_index >= 5  # 5=Saturday, 6=Sunday
    today_str = today.isoformat()
    is_public_holiday_today = any(h.date == today_str for h in holidays)

    is_bridge_day = False
    if not is_public_holiday_today and not is_weekend:
        yesterday_str = (today - timedelta(days=1)).isoformat()
        tomorrow_str = (today + timedelta(days=1)).isoformat()
        has_adjacent_holiday = any(
            h.date in (yesterday_str, tomorrow_str) for h in holidays
        )
        if has_adjacent_holiday and weekday_index in (0, 4):  # Monday or Friday
            is_bridge_day = True

    return DayMeta(
        isWeekend=is_weekend,
        isPublicHolidayToday=is_public_holiday_today,
        isBridgeDay=is_bridge_day,
    )


async def build_snapshot(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> ContextEnvelope:
    """
    Build a full environment context snapshot wrapped in a ContextEnvelope.
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()

    # Effective location: client hint, detected server location, or default
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

    # Holidays and derived fields
    holidays: list[Holiday] = []
    last_holiday: Optional[HolidayDistance] = None
    next_holiday: Optional[HolidayDistance] = None
    nearest_holiday: Optional[HolidayDistance] = None

    if location.countryCode:
        try:
            all_holidays = await fetch_holidays(location.countryCode, now.year)

            # Optional region filter (e.g. "DE-BW")
            if location.region:
                base_region = location.region

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

            last_holiday, next_holiday, nearest_holiday = _compute_holiday_distances(
                holidays, today
            )
        except Exception:
            holidays = []
            last_holiday = None
            next_holiday = None
            nearest_holiday = None

    day_meta = _compute_day_meta(today, now.weekday(), holidays)

    # Weather: current, forecast next hours, tomorrow summary
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

    # Daylight
    daylight: Optional[DaylightContext] = None
    try:
        daylight = await fetch_daylight_context(
            location.lat,
            location.lon,
            timezone=TIMEZONE,
        )
    except Exception:
        daylight = None

    # Place context
    place_ctx: Optional[PlaceContext] = None
    try:
        place_ctx = await fetch_place_context(location.lat, location.lon)
    except Exception:
        place_ctx = None

    # Comfort context
    comfort: Optional[ComfortContext] = None
    try:
        comfort = compute_comfort_context(
            weather_current,
            weather_forecast,
            weather_tomorrow,
        )
    except Exception:
        comfort = None

    # Events context
    try:
        events_ctx = await fetch_local_events_context(location, now)
    except Exception:
        events_ctx = EventsContext(localToday=[])

    env = EnvironmentContext(
        location=location,
        dateTime=dt,
        dayMeta=day_meta,
        holidays=holidays,
        lastHoliday=last_holiday,
        nextHoliday=next_holiday,
        nearestHoliday=nearest_holiday,
        weather_current=weather_current,
        weather_forecast=weather_forecast,
        weather_tomorrow=weather_tomorrow,
        daylight=daylight,
        placeContext=place_ctx,
        comfort=comfort,
        events=events_ctx,
        locale=locale,
    )

    produced_at = now.isoformat()
    content_hash = _stable_hash(env.model_dump())

    return ContextEnvelope(
        type="context-snapshot",
        version="1.0",
        producedAt=produced_at,
        hash=content_hash,
        data=env,
    )


async def build_delta(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
    since_hash: Optional[str] = None,
) -> ContextEnvelope:
    """
    Build a delta envelope compared to the given since_hash.
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
