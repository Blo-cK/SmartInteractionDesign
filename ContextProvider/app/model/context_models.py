from pydantic import BaseModel
from typing import Optional, List, Literal


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


# ---------- Holidays ----------

class Holiday(BaseModel):
    """Public holiday information."""
    date: str           # YYYY-MM-DD
    localName: str
    countryCode: CountryCode
    regions: Optional[List[RegionCode]] = None


# ---------- Weather ----------

class WeatherContext(BaseModel):
    """Current weather at the resolved location."""
    provider: str = "open-meteo"
    temperatureC: Optional[float] = None
    windKph: Optional[float] = None
    precipitationMm: Optional[float] = None
    summary: Optional[str] = None


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
    holidays: List[Holiday] = []
    weather: Optional[WeatherContext] = None
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