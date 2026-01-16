from __future__ import annotations

import hashlib
import json
from datetime import datetime, date, timedelta
from typing import Optional, List
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
    HolidaySummary,
    HolidayWithDelta,
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

# Filled once via IP-based lookup and then reused
SERVER_LOCATION: Optional[LocationResolved] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _part_of_day(hour: int) -> str:
    """Return a coarse time-of-day label for an hour in [0, 23]."""
    if hour < 5:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _parse_locale(accept_language: Optional[str]) -> LocaleContext:
    """
    Parse an Accept-Language header into a LocaleContext.

    Example: "de-DE,de;q=0.9,en-US;q=0.8" -> language="de", locale="de-DE"
    """
    header = accept_language or "en-US"
    primary = header.split(",")[0].strip()
    language = primary.split("-")[0]
    return LocaleContext(language=language, locale=primary)


def _resolve_location_from_hint(hint: Optional[LocationHint]) -> LocationResolved:
    """
    Resolve location from a client-supplied LocationHint.
    Falls back to Karlsruhe if no coordinates are provided.
    """
    if hint and hint.lat is not None and hint.lon is not None:
        return LocationResolved(
            lat=hint.lat,
            lon=hint.lon,
            city=hint.city or "Karlsruhe",
            countryCode=hint.countryCode or "DE",
            region=hint.region,
        )

    # Default fallback if no hint is available and server location cannot be detected
    return LocationResolved(
        lat=49.0069,
        lon=8.4037,
        city="Karlsruhe",
        countryCode="DE",
        region=None,
    )


def _stable_hash(obj: object) -> str:
    """
    Compute a stable hash (sha256, first 16 hex chars) over a JSON-serialized object.
    Used to detect changes for delta-style updates.
    """
    json_str = json.dumps(obj, sort_keys=True, default=str)
    h = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return h[:16]


def _filter_holidays_for_region(
    all_holidays: List[Holiday],
    country_code: Optional[str],
    region: Optional[str],
) -> List[Holiday]:
    """
    Filter the Nager.Date holidays list down to the relevant region, if possible.

    If a holiday has no regions defined, it is treated as nationwide.
    If regions are present, only holidays whose region matches the given region code
    are kept.
    """
    if not country_code:
        return all_holidays

    if not region:
        return all_holidays

    base_region = region  # e.g. "DE-BW", "BW" or "Baden-Württemberg"
    candidate_codes: List[str] = []

    if "-" in base_region:
        # Already in form "DE-BW"
        candidate_codes.append(base_region)
    else:
        # Short code "BW" -> "DE-BW"
        candidate_codes.append(f"{country_code}-{base_region}")

    def matches_region(h: Holiday) -> bool:
        if h.regions is None:
            return True
        return any(code in (h.regions or []) for code in candidate_codes)

    return [h for h in all_holidays if matches_region(h)]


def _compute_day_meta(
    today: date,
    weekday_index: int,
    holidays: List[Holiday],
) -> DayMeta:
    """Compute weekend / holiday / bridge-day flags for the current date."""
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
        # Simple heuristic: Monday or Friday next to at least one holiday
        if has_adjacent_holiday and weekday_index in (0, 4):
            is_bridge_day = True

    return DayMeta(
        isWeekend=is_weekend,
        isPublicHolidayToday=is_public_holiday_today,
        isBridgeDay=is_bridge_day,
    )


def _to_holiday_with_delta(h: Holiday, delta_days: int) -> HolidayWithDelta:
    """
    Convert a Holiday plus a signed delta in days into a HolidayWithDelta.

    - For past holidays (delta_days < 0): daysAgo > 0, daysUntil = None
    - For future holidays (delta_days > 0): daysUntil > 0, daysAgo = None
    - For today (delta_days == 0): both daysAgo and daysUntil are 0
    """
    if delta_days < 0:
        return HolidayWithDelta(
            date=h.date,
            localName=h.localName,
            countryCode=h.countryCode,
            daysAgo=-delta_days,
            daysUntil=None,
        )
    elif delta_days > 0:
        return HolidayWithDelta(
            date=h.date,
            localName=h.localName,
            countryCode=h.countryCode,
            daysAgo=None,
            daysUntil=delta_days,
        )
    else:
        return HolidayWithDelta(
            date=h.date,
            localName=h.localName,
            countryCode=h.countryCode,
            daysAgo=0,
            daysUntil=0,
        )


def _compute_holiday_summary(
    holidays: List[Holiday],
    today: date,
) -> HolidaySummary:
    """
    Compute a compact summary:
    - lastHoliday: most recent holiday in the past
    - nextHoliday: next upcoming holiday in the future
    - nearestHoliday: closest holiday in time relative to today
    """
    if not holidays:
        return HolidaySummary(
            lastHoliday=None,
            nextHoliday=None,
            nearestHoliday=None,
        )

    last_h: Optional[HolidayWithDelta] = None
    next_h: Optional[HolidayWithDelta] = None
    nearest_h: Optional[HolidayWithDelta] = None

    min_future_delta: Optional[int] = None
    max_past_delta: Optional[int] = None
    min_abs_delta: Optional[int] = None

    for h in holidays:
        try:
            h_date = date.fromisoformat(h.date)
        except ValueError:
            continue

        delta_days = (h_date - today).days

        # Track last holiday (closest in the past)
        if delta_days < 0:
            if max_past_delta is None or delta_days > max_past_delta:
                max_past_delta = delta_days
                last_h = _to_holiday_with_delta(h, delta_days)

        # Track next holiday (closest in the future)
        elif delta_days > 0:
            if min_future_delta is None or delta_days < min_future_delta:
                min_future_delta = delta_days
                next_h = _to_holiday_with_delta(h, delta_days)

        # Today is a holiday
        else:
            # For today, we treat it as both last and next with distance 0.
            last_h = _to_holiday_with_delta(h, 0)
            next_h = _to_holiday_with_delta(h, 0)

        # Track nearest holiday in absolute distance
        abs_delta = abs(delta_days)
        if min_abs_delta is None or abs_delta < min_abs_delta:
            min_abs_delta = abs_delta
            nearest_h = _to_holiday_with_delta(h, delta_days)

    return HolidaySummary(
        lastHoliday=last_h,
        nextHoliday=next_h,
        nearestHoliday=nearest_h,
    )


async def _determine_effective_location(
    location_hint: Optional[LocationHint],
) -> LocationResolved:
    """
    Determine the effective location in this order:
    1) client-supplied LocationHint (lat/lon)
    2) cached server location (resolved via IP)
    3) static fallback (Karlsruhe)
    """
    global SERVER_LOCATION

    # 1) Client hint with coordinates wins
    if location_hint and location_hint.lat is not None and location_hint.lon is not None:
        return _resolve_location_from_hint(location_hint)

    # 2) Try auto-detection once
    if SERVER_LOCATION is None:
        try:
            auto_loc = await detect_server_location()
            if auto_loc is not None:
                SERVER_LOCATION = auto_loc
                print(f"[ContextProvider] Auto-detected server location: {SERVER_LOCATION}")
        except Exception:
            SERVER_LOCATION = None

    if SERVER_LOCATION is not None:
        return LocationResolved(
            lat=SERVER_LOCATION.lat,
            lon=SERVER_LOCATION.lon,
            city=SERVER_LOCATION.city,
            countryCode=SERVER_LOCATION.countryCode,
            region=SERVER_LOCATION.region,
        )

    # 3) Static fallback
    return _resolve_location_from_hint(None)


# ---------------------------------------------------------------------------
# Dynamic context (for HTTP + dynamic Kafka topic)
# ---------------------------------------------------------------------------

async def build_dynamic_context(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> ContextEnvelope:
    """
    Build a full environment context snapshot:

    - location (hint / server / fallback)
    - locale
    - holidays + holiday summary
    - day meta (weekend / holiday / bridge day)
    - time of day
    - weather now, short-term forecast, tomorrow summary
    - daylight (sunrise/sunset)
    - place categorisation
    - comfort estimation
    - events container (local events)
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()

    location = await _determine_effective_location(location_hint)
    locale = _parse_locale(accept_language)

    # Core date/time
    dt = DateTimeContext(
        iso=now.isoformat(),
        timezone=TIMEZONE,
        weekday=now.strftime("%A"),
        partOfDay=_part_of_day(now.hour),
    )

    # Holidays + summary
    holidays: List[Holiday] = []
    holiday_summary: Optional[HolidaySummary] = None
    if location.countryCode:
        try:
            all_holidays = await fetch_holidays(location.countryCode, now.year)
            holidays = _filter_holidays_for_region(
                all_holidays,
                country_code=location.countryCode,
                region=location.region,
            )
            holiday_summary = _compute_holiday_summary(holidays, today)
        except Exception:
            holidays = []
            holiday_summary = None

    day_meta = _compute_day_meta(today, now.weekday(), holidays)

    # Weather
    weather_current: Optional[WeatherContext] = None
    weather_forecast: List[WeatherForecastPoint] = []
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

    # Events
    try:
        events_ctx = await fetch_local_events_context(location, now)
    except Exception:
        events_ctx = EventsContext(localToday=[])

    env = EnvironmentContext(
        location=location,
        dateTime=dt,
        dayMeta=day_meta,
        holidays=holidays,
        holidaySummary=holiday_summary,
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


# ---------------------------------------------------------------------------
# Static context (for static Kafka topic)
# ---------------------------------------------------------------------------

async def build_static_context(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> dict:
    """
    Build a reduced, mostly static context payload.

    Intended for the "static" Kafka topic:
    - location
    - locale
    - holidays (filtered for region)
    - holidaySummary (last / next / nearest)
    - dayMeta (weekend / holiday / bridge day)
    - placeContext
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    today = now.date()

    location = await _determine_effective_location(location_hint)
    locale = _parse_locale(accept_language)

    holidays: List[Holiday] = []
    holiday_summary: Optional[HolidaySummary] = None
    if location.countryCode:
        try:
            all_holidays = await fetch_holidays(location.countryCode, now.year)
            holidays = _filter_holidays_for_region(
                all_holidays,
                country_code=location.countryCode,
                region=location.region,
            )
            holiday_summary = _compute_holiday_summary(holidays, today)
        except Exception:
            holidays = []
            holiday_summary = None

    day_meta = _compute_day_meta(today, now.weekday(), holidays)

    place_ctx: Optional[PlaceContext] = None
    try:
        place_ctx = await fetch_place_context(location.lat, location.lon)
    except Exception:
        place_ctx = None

    # Static payload is a simple dict, not a ContextEnvelope
    return {
        "location": location.model_dump(),
        "locale": locale.model_dump(),
        "holidays": [h.model_dump() for h in holidays],
        "holidaySummary": holiday_summary.model_dump() if holiday_summary else None,
        "dayMeta": day_meta.model_dump(),
        "placeContext": place_ctx.model_dump() if place_ctx else None,
    }


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers for HTTP API
# ---------------------------------------------------------------------------

async def build_snapshot(
    accept_language: Optional[str],
    location_hint: Optional[LocationHint] = None,
) -> ContextEnvelope:
    """
    Wrapper used by the HTTP /context endpoints.
    Delegates to the dynamic context builder.
    """
    return await build_dynamic_context(
        accept_language=accept_language,
        location_hint=location_hint,
    )


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
