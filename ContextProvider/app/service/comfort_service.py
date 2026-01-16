from __future__ import annotations
from typing import Optional, List
from ContextProvider.app.model.context_models import (
    ComfortContext,
    WeatherContext,
    WeatherForecastPoint,
    WeatherTomorrow,
    RiskLevel,
)

def _max_precipitation(forecast: List[WeatherForecastPoint]) -> float:
    """Return the highest precipitation value found in the forecast."""
    values = [fp.precipitationMm or 0.0 for fp in forecast]
    return max(values) if values else 0.0

def _choose_baseline_temp(
    weather_current: Optional[WeatherContext],
    weather_forecast: List[WeatherForecastPoint],
) -> Optional[float]:
    """
    Choose a representative temperature:
    - prefer the current reading if available
    - otherwise fall back to the average of the forecast temperatures
    """
    if weather_current and weather_current.temperatureC is not None:
        return weather_current.temperatureC

    temps = [fp.temperatureC for fp in weather_forecast if fp.temperatureC is not None]
    if temps:
        return sum(temps) / len(temps)

    return None

def compute_comfort_context(
    weather_current: Optional[WeatherContext],
    weather_forecast: List[WeatherForecastPoint],
    weather_tomorrow: Optional[WeatherTomorrow],
) -> ComfortContext:
    """
    Build a coarse comfort/risk assessment from current and short-term weather data.
    """
    baseline_temp = _choose_baseline_temp(weather_current, weather_forecast)
    precip_max = _max_precipitation(weather_forecast)

    cold_risk: RiskLevel = "none"
    heat_risk: RiskLevel = "none"
    ice_risk: RiskLevel = "none"

    # Cold risk
    if baseline_temp is not None:
        if baseline_temp <= 0:
            cold_risk = "high"
        elif baseline_temp <= 5:
            cold_risk = "medium"
        elif baseline_temp <= 10:
            cold_risk = "low"

    # Heat risk
    if baseline_temp is not None:
        if baseline_temp >= 30:
            heat_risk = "high"
        elif baseline_temp >= 25:
            heat_risk = "medium"
        elif baseline_temp >= 20:
            heat_risk = "low"

    # Ice risk
    if baseline_temp is not None:
        if baseline_temp <= 1 and precip_max > 0.1:
            ice_risk = "high"
        elif baseline_temp <= 3 and precip_max > 0.0:
            ice_risk = "medium"
        else:
            ice_risk = "none"

    # Outdoor recommendations
    outdoor_recommended = True
    short_walk_ok = True
    long_walk_ok = True

    if cold_risk == "high" or heat_risk == "high" or ice_risk == "high":
        outdoor_recommended = False
        long_walk_ok = False

    if precip_max >= 5.0:
        outdoor_recommended = False
        long_walk_ok = False
    elif precip_max >= 2.0:
        long_walk_ok = False

    if cold_risk == "medium" and precip_max > 0.0:
        long_walk_ok = False

    return ComfortContext(
        coldRisk=cold_risk,
        heatRisk=heat_risk,
        iceRisk=ice_risk,
        outdoorRecommended=outdoor_recommended,
        shortWalkOk=short_walk_ok,
        longWalkOk=long_walk_ok,
    )
