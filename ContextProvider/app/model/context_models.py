from typing import Optional, List, Literal
from pydantic import BaseModel, Field

# ---------- Basic Types ----------

CountryCode = str  # ISO-3166-1 alpha-2 (e.g. "DE")
RegionCode = str   # ISO-3166-2 (e.g. "DE-BW")


# ---------- Location ----------

class LocationHint(BaseModel):
    """Optional location input provided by the client."""
    city: Optional[str] = None
    countryCode: Optional[CountryCode] = None
    region: Optional[RegionCode] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class LocationResolved(BaseModel):
    """Resolved location used internally and returned to clients."""
    lat: float
    lon: float
    city: Optional[str] = None
    countryCode: Optional[CountryCode] = None
    region: Optional[RegionCode] = None


# ---------- Date & Time ----------

PartOfDay = Literal["morning", "afternoon", "evening", "night"]


class DateTimeContext(BaseModel):
    """Time information used for contextual conversation."""
    iso: str
    timezone: str
    weekday: str
    partOfDay: PartOfDay


class DayMeta(BaseModel):
    """Meta information about 'today', derived from date + holidays."""
    isWeekend: bool
    isPublicHolidayToday: bool
    isBridgeDay: bool


# ---------- Holidays ----------

class Holiday(BaseModel):
    """Public holiday information (optionally region-scoped)."""
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    regions: Optional[List[RegionCode]] = None


class HolidayWithDelta(BaseModel):
    """Single holiday with distance (in days) to 'today'."""
    date: str                   # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    # If the holiday lies in the past: daysAgo > 0, daysUntil = None.
    daysAgo: Optional[int] = None
    # If the holiday lies in the future: daysUntil > 0, daysAgo = None.
    daysUntil: Optional[int] = None


class HolidaySummary(BaseModel):
    """Summary around the current date: last / next / nearest holiday."""
    lastHoliday: Optional[HolidayWithDelta] = None
    nextHoliday: Optional[HolidayWithDelta] = None
    nearestHoliday: Optional[HolidayWithDelta] = None


# ---------- Weather ----------

class WeatherContext(BaseModel):
    """Current weather at the resolved location."""
    provider: str = "open-meteo"
    temperatureC: Optional[float] = None
    windKph: Optional[float] = None
    precipitationMm: Optional[float] = None
    summary: Optional[str] = None


class WeatherForecastPoint(BaseModel):
    """Weather forecast for a specific hour in the near future."""
    timestamp: str                # ISO 8601 datetime with timezone
    date: str                     # YYYY-MM-DD (local date)
    time: str                     # HH:MM (local time)
    weekday: str                  # Weekday name, e.g. "Sunday"
    temperatureC: Optional[float] = None
    windKph: Optional[float] = None
    precipitationMm: Optional[float] = None


class WeatherTomorrow(BaseModel):
    """Daily weather summary for tomorrow."""
    date: str                     # YYYY-MM-DD
    temperatureMaxC: Optional[float] = None
    temperatureMinC: Optional[float] = None
    precipitationMm: Optional[float] = None
    windKphMax: Optional[float] = None


# ---------- Daylight ----------

class DaylightContext(BaseModel):
    """Information about sunrise/sunset and daylight."""
    sunrise: Optional[str] = None        # ISO 8601
    sunset: Optional[str] = None         # ISO 8601
    isDaylight: Optional[bool] = None
    minutesUntilSunrise: Optional[int] = None
    minutesUntilSunset: Optional[int] = None


# ---------- Place / Umgebung ----------

PlaceType = Literal[
    "unknown",
    "home_like",
    "office_area",
    "university_campus",
    "city_center",
    "residential",
    "rural",
    "mall",
    "transport_hub",
]


class PlaceContext(BaseModel):
    """Coarse-grained classification of the surrounding place."""
    placeType: PlaceType = "unknown"
    rawCategory: Optional[str] = None
    rawType: Optional[str] = None
    nearbyCategories: List[str] = Field(default_factory=list)


# ---------- Comfort ----------

RiskLevel = Literal["none", "low", "medium", "high"]


class ComfortContext(BaseModel):
    """Heuristic comfort and risk estimation derived from weather."""
    coldRisk: RiskLevel = "none"
    heatRisk: RiskLevel = "none"
    iceRisk: RiskLevel = "none"
    outdoorRecommended: bool = True
    shortWalkOk: bool = True
    longWalkOk: bool = True


# ---------- Events ----------

class LocalEvent(BaseModel):
    """Representation of a local event that could be relevant for small talk."""
    title: str
    category: Optional[str] = None
    startTime: Optional[str] = None      # ISO datetime
    endTime: Optional[str] = None        # ISO datetime
    locationName: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None


class EventsContext(BaseModel):
    """Container for local events, e.g., today/nearby."""
    localToday: List[LocalEvent] = Field(default_factory=list)


# ---------- Locale ----------

class LocaleContext(BaseModel):
    """Language and locale derived from the Accept-Language header."""
    language: str
    locale: str


# ---------- Environment Context ----------

class EnvironmentContext(BaseModel):
    """
    Full environment context returned to the client.

    This model is used both for dynamic and static snapshots. Static
    snapshots simply omit or leave None the dynamic parts if desired.
    """
    location: LocationResolved
    dateTime: DateTimeContext
    dayMeta: DayMeta

    holidays: List[Holiday] = Field(default_factory=list)
    holidaySummary: Optional[HolidaySummary] = None

    weather_current: Optional[WeatherContext] = None
    weather_forecast: List[WeatherForecastPoint] = Field(default_factory=list)
    weather_tomorrow: Optional[WeatherTomorrow] = None

    daylight: Optional[DaylightContext] = None
    placeContext: Optional[PlaceContext] = None
    comfort: Optional[ComfortContext] = None
    events: EventsContext = Field(default_factory=EventsContext)

    locale: LocaleContext


# ---------- Envelope ----------

class ContextEnvelope(BaseModel):
    """Wrapper for snapshots and delta-style responses."""
    type: Literal["context-snapshot", "context-delta"]
    version: str
    producedAt: str
    hash: str
    data: EnvironmentContext | dict


# ---------- Request Body ----------

class ContextInput(BaseModel):
    """POST /context input model."""
    locationHint: Optional[LocationHint] = None
