from __future__ import annotations
from typing import Optional
import httpx
from ..model.context_models import LocationResolved, CountryCode, RegionCode

IP_API_URL = "https://api.ipify.org"
GEO_API_URL = "https://ipapi.co"  # can be swapped for another provider if needed


async def detect_server_location() -> Optional[LocationResolved]:
    """
    Try to detect the physical location of the host running this service
    using IP-based geolocation.
    Returns:
        LocationResolved or None if detection fails.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1) Determine outward-facing IP of this host
            ip_resp = await client.get(IP_API_URL, params={"format": "json"})
            ip_resp.raise_for_status()
            ip_data = ip_resp.json()
            ip = ip_data.get("ip")
            if not ip:
                return None

            # 2) Use a geolocation API to resolve IP -> location
            geo_resp = await client.get(f"{GEO_API_URL}/{ip}/json/")
            geo_resp.raise_for_status()
            data = geo_resp.json()

    except httpx.HTTPError:
        return None

    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        return None

    city = data.get("city")
    country_code: Optional[CountryCode] = data.get("country_code")
    # ipapi: "region_code" is for example "BW", "BY" ...
    region_code_raw = data.get("region_code")

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None

    region: Optional[RegionCode] = None
    if country_code and region_code_raw:
        region = f"{country_code}-{region_code_raw}"

    return LocationResolved(
        lat=lat_f,
        lon=lon_f,
        city=city,
        countryCode=country_code,
        region=region,
    )
