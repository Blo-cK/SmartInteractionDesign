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
    UpcomingHoliday,
    WeatherForecastPoint,
    WeatherTomorrow,
    DaylightContext,
    PlaceContext,
    ComfortContext,
    EventsContext,
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

# Populated once via IP-based geolocation and reused for later requests.
SERVER_LOCATION: Optional[LocationResolved] = None

def _part_of_day(hour: int) -> str:
    """Return a coarse part-of-day label for the given hour (0–23)."""
    if hour < 5:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"

def _parse_locale(accept_language: Optional[str]) -> LocaleContext:
    """
    Turn an Accept-Language header into a simple LocaleContext.

    Example:
      "de-DE,de;q=0.9,en-US;q=0.8" -> language="de", locale="de-DE"
    """
    header = accept_language or "en-US"
    primary = header.split(",")[0].strip()
    language = primary.split("-")[0]
    return LocaleContext(language=language, locale=primary)

def _resolve_location(hint: Optional[LocationHint]) -> LocationResolved:
    """
    Decide which location to use.

    - If a hint with latitude/longitude is present, use that.
    - Otherwise, fall back to a default location (Karlsruhe).
    """
    if hint and hint.lat is not None and hint.lon is not None:
        return LocationResolved(
            lat=hint.lat,
            lon=hint.lon,
            city=hint.city or "Karlsruhe",
            countryCode=hint.countryCode or "DE",
            region=hint.region,
        )

    # Default location if nothing else is known
    return LocationResolved(
        lat=49.0069,
        lon=8.4037,
        city="Karlsruhe",
        countryCode="DE",
        region=None,
    )

def _stable_hash(obj: object) -> str:
    """
    Compute a deterministic hash (sha256, first 16 hex chars) for a JSON-serialised object.

    Used to detect whether a context snapshot has changed.
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
    Pick holidays that fall into the next 'window_days' days and
    compute the remaining days until each of them.
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

def _compute_day_meta(
    today: date,
    weekday_index: int,
    holidays: list[Holiday],
) -> DayMeta:
    """
    Derive basic day flags such as weekend, public holiday, and a simple
    notion of a bridge day (workday between holiday and weekend).
    """
    is_weekend = weekday_index >= 5  # 5 = Saturday, 6 = Sunday
    today_str = today.isoformat()
    is_public_holiday_today = any(h.date == today_str for h in holidays)

    is_bridge_day = False
    if not is_public_holiday_today and not is_weekend:
        yesterday_str = (today - timedelta(days=1)).isoformat()
        tomorrow_str = (today + timedelta(days=1)).isoformat()
        has_adjacent_holiday = any(
            h.date in (yesterday_str, tomorrow_str) for h in holidays
        )
        # Monday or Friday next to a holiday is treated as a bridge day
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
    Build a complete environment snapshot and wrap it in a ContextEnvelope.
    A snapshot contains:
      - location (from client hint, detected server location, or default)
      - date/time and basic day flags
      - holidays and upcoming holidays
      - weather information (current, short-term forecast, tomorrow)
      - daylight information (sunrise/sunset)
      - place classification (e.g. residential, university campus)
      - comfort indicators (cold/heat/ice risk, outdoor suggestions)
      - local events container
      - locale inferred from the Accept-Language header
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()

    # Resolve which hint to use for location
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

    # Basic time description for the current moment
    dt = DateTimeContext(
        iso=now.isoformat(),
        timezone=TIMEZONE,
        weekday=now.strftime("%A"),
        partOfDay=_part_of_day(now.hour),
    )

    # Holiday data (optionally filtered by region if available)
    holidays: list[Holiday] = []
    upcoming_holidays: list[UpcomingHoliday] = []
    if location.countryCode:
        try:
            all_holidays = await fetch_holidays(location.countryCode, now.year)

            if location.region:
                base_region = location.region  # e.g. "DE-BW", "BW", or "Baden-Württemberg"

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

    # Derived flags for the current day (weekend/holiday/bridge day)
    day_meta = _compute_day_meta(today, now.weekday(), holidays)

    # Weather: current conditions, short-term forecast, and tomorrow's outlook
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

    # Daylight information (sunrise/sunset)
    daylight: Optional[DaylightContext] = None
    try:
        daylight = await fetch_daylight_context(
            location.lat,
            location.lon,
            timezone=TIMEZONE,
        )
    except Exception:
        daylight = None

    # High-level place classification
    place_ctx: Optional[PlaceContext] = None
    try:
        place_ctx = await fetch_place_context(location.lat, location.lon)
    except Exception:
        place_ctx = None

    # Comfort indicators derived from weather
    comfort: Optional[ComfortContext] = None
    try:
        comfort = compute_comfort_context(
            weather_current,
            weather_forecast,
            weather_tomorrow,
        )
    except Exception:
        comfort = None

    # Local events or activity suggestions
    events_ctx: EventsContext
    try:
        events_ctx = await fetch_local_events_context(location, now)
    except Exception:
        events_ctx = EventsContext(localToday=[])

    env = EnvironmentContext(
        location=location,
        dateTime=dt,
        dayMeta=day_meta,
        holidays=holidays,
        upcoming_holidays=upcoming_holidays,
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
    Build a delta response relative to the given hash.

    If the hash is missing or matches the current snapshot, the data field
    is empty to indicate that nothing has changed.
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

    # Hash differs -> return full context as the delta
    return ContextEnvelope(
        type="context-delta",
        version=snapshot.version,
        producedAt=snapshot.producedAt,
        hash=snapshot.hash,
        data=snapshot.data,
    )
