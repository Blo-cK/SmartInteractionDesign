from typing import Optional, List, Literal
from pydantic import BaseModel, Field

# ---------- Basic Types ----------
CountryCode = str  # ISO-3166-1 alpha-2 (e.g. "DE")
RegionCode = str   # ISO-3166-2 (e.g. "DE-BW")

# ---------- Location ----------
class LocationHint(BaseModel):
    """Location information provided by the caller, if available."""
    city: Optional[str] = None
    countryCode: Optional[CountryCode] = None
    region: Optional[RegionCode] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class LocationResolved(BaseModel):
    """Location used internally and returned as part of the context."""
    lat: float
    lon: float
    city: Optional[str] = None
    countryCode: Optional[CountryCode] = None
    region: Optional[RegionCode] = None

# ---------- Date & Time ----------
PartOfDay = Literal["morning", "afternoon", "evening", "night"]

class DateTimeContext(BaseModel):
    """Time-related information used for context-aware behaviour."""
    iso: str
    timezone: str
    weekday: str
    partOfDay: PartOfDay

class DayMeta(BaseModel):
    """Basic flags derived from the current date and holiday information."""
    isWeekend: bool
    isPublicHolidayToday: bool
    isBridgeDay: bool

# ---------- Holidays ----------
class Holiday(BaseModel):
    """Information about a public holiday."""
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    regions: Optional[List[RegionCode]] = None

class UpcomingHoliday(BaseModel):
    """Public holiday within a defined lookahead window."""
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    daysUntil: int      # days from today until this holiday

# ---------- Weather ----------
class WeatherContext(BaseModel):
    """Snapshot of the current weather at the resolved location."""
    provider: str = "open-meteo"
    temperatureC: Optional[float] = None
    windKph: Optional[float] = None
    precipitationMm: Optional[float] = None
    summary: Optional[str] = None

class WeatherForecastPoint(BaseModel):
    """Hourly weather forecast entry for the near future."""
    timestamp: str                # ISO 8601 datetime with timezone
    date: str                     # YYYY-MM-DD (local date)
    time: str                     # HH:MM (local time)
    weekday: str                  # Weekday name, e.g. "Sunday"
    temperatureC: Optional[float] = None
    windKph: Optional[float] = None
    precipitationMm: Optional[float] = None

class WeatherTomorrow(BaseModel):
    """Aggregated weather outlook for the next day."""
    date: str                     # YYYY-MM-DD
    temperatureMaxC: Optional[float] = None
    temperatureMinC: Optional[float] = None
    precipitationMm: Optional[float] = None
    windKphMax: Optional[float] = None

# ---------- Daylight ----------
class DaylightContext(BaseModel):
    """Sunrise/sunset and basic daylight information."""
    sunrise: Optional[str] = None        # ISO 8601
    sunset: Optional[str] = None         # ISO 8601
    isDaylight: Optional[bool] = None
    minutesUntilSunrise: Optional[int] = None
    minutesUntilSunset: Optional[int] = None

# ---------- Place / Surroundings ----------
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
    """High-level classification of the surrounding area."""
    placeType: PlaceType = "unknown"
    rawCategory: Optional[str] = None
    rawType: Optional[str] = None
    nearbyCategories: List[str] = Field(default_factory=list)

# ---------- Comfort / Well-being ----------
RiskLevel = Literal["none", "low", "medium", "high"]

class ComfortContext(BaseModel):
    """Approximate comfort and risk indicators derived from weather data."""
    coldRisk: RiskLevel = "none"
    heatRisk: RiskLevel = "none"
    iceRisk: RiskLevel = "none"
    outdoorRecommended: bool = True
    shortWalkOk: bool = True
    longWalkOk: bool = True

# ---------- Events ----------
class LocalEvent(BaseModel):
    """Local event or activity relevant for the user's surroundings."""
    title: str
    category: Optional[str] = None
    startTime: Optional[str] = None      # ISO datetime
    endTime: Optional[str] = None        # ISO datetime
    locationName: Optional[str] = None
    url: Optional[str] = None

class EventsContext(BaseModel):
    """Collection of local events for the current area."""
    localToday: List[LocalEvent] = Field(default_factory=list)

# ---------- Locale ----------
class LocaleContext(BaseModel):
    """Language and locale inferred from the client's preferences."""
    language: str
    locale: str

# ---------- Environment Context ----------
class EnvironmentContext(BaseModel):
    """Combined environment information exposed by the service."""
    location: LocationResolved
    dateTime: DateTimeContext
    dayMeta: DayMeta

    holidays: List[Holiday] = Field(default_factory=list)
    upcoming_holidays: List[UpcomingHoliday] = Field(default_factory=list)

    # Weather
    weather_current: Optional[WeatherContext] = None
    weather_forecast: List[WeatherForecastPoint] = Field(default_factory=list)
    weather_tomorrow: Optional[WeatherTomorrow] = None

    # Daylight / place / comfort / events
    daylight: Optional[DaylightContext] = None
    placeContext: Optional[PlaceContext] = None
    comfort: Optional[ComfortContext] = None
    events: EventsContext = Field(default_factory=EventsContext)

    # Language / locale
    locale: LocaleContext

# ---------- Envelope ----------
class ContextEnvelope(BaseModel):
    """Wrapper for full snapshots and delta responses."""
    type: Literal["context-snapshot", "context-delta"]
    version: str
    producedAt: str
    hash: str
    data: EnvironmentContext | dict

# ---------- Request Body ----------
class ContextInput(BaseModel):
    """Request body for context endpoints that accept a location hint."""
    locationHint: Optional[LocationHint] = None
