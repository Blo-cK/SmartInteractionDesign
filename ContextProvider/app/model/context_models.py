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
    """Meta information about 'today', derived from date and holidays."""
    isWeekend: bool
    isPublicHolidayToday: bool
    isBridgeDay: bool


# ---------- Holidays ----------

class Holiday(BaseModel):
    """Public holiday information."""
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    regions: Optional[List[RegionCode]] = None


class HolidayDistance(BaseModel):
    """
    Holiday with an explicit distance in days.

    Exactly one of the fields is usually set:
    - daysSince: holiday is in the past (0 = today, >0 = days ago)
    - daysUntil: holiday is in the future (0 = today, >0 = days until)
    """
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    daysSince: Optional[int] = None
    daysUntil: Optional[int] = None


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
    """Information about sunrise, sunset and daylight situation."""
    sunrise: Optional[str] = None        # ISO 8601
    sunset: Optional[str] = None         # ISO 8601
    isDaylight: Optional[bool] = None
    minutesUntilSunrise: Optional[int] = None
    minutesUntilSunset: Optional[int] = None


# ---------- Place / Area ----------

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


# ---------- Comfort Context ----------

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
    """Representation of a local event that could be relevant for smalltalk."""
    title: str
    category: Optional[str] = None
    startTime: Optional[str] = None      # ISO datetime
    endTime: Optional[str] = None        # ISO datetime
    locationName: Optional[str] = None
    url: Optional[str] = None


class EventsContext(BaseModel):
    """Container for local events, e.g., for 'today' and nearby."""
    localToday: List[LocalEvent] = Field(default_factory=list)


# ---------- Locale ----------

class LocaleContext(BaseModel):
    """Language and locale derived from the Accept-Language header."""
    language: str
    locale: str


# ---------- Environment Context ----------

class EnvironmentContext(BaseModel):
    """Full environment context returned to the client."""
    location: LocationResolved
    dateTime: DateTimeContext
    dayMeta: DayMeta

    holidays: List[Holiday] = Field(default_factory=list)

    # Precomputed holiday anchors
    lastHoliday: Optional[HolidayDistance] = None
    nextHoliday: Optional[HolidayDistance] = None
    nearestHoliday: Optional[HolidayDistance] = None

    # Weather
    weather_current: Optional[WeatherContext] = None
    weather_forecast: List[WeatherForecastPoint] = Field(default_factory=list)
    weather_tomorrow: Optional[WeatherTomorrow] = None

    # Daylight / place / comfort / events
    daylight: Optional[DaylightContext] = None
    placeContext: Optional[PlaceContext] = None
    comfort: Optional[ComfortContext] = None
    events: EventsContext = Field(default_factory=EventsContext)

    # Language / localisation
    locale: LocaleContext


# ---------- Envelope ----------

class ContextEnvelope(BaseModel):
    """Generic wrapper for snapshots and delta updates."""
    type: Literal["context-snapshot", "context-delta"]
    version: str
    producedAt: str
    hash: str
    data: EnvironmentContext | dict


# ---------- Request Body ----------

class ContextInput(BaseModel):
    """POST /context input model."""
    locationHint: Optional[LocationHint] = None
