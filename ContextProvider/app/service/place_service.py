from __future__ import annotations
from typing import List
import httpx
from ContextProvider.app.model.context_models import PlaceContext, PlaceType

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

async def fetch_place_context(lat: float, lon: float) -> PlaceContext:
    """
    Use OpenStreetMap Nominatim reverse geocoding to derive a coarse place type.
    This is heuristic but good enough for conversational hints.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "extratags": 1,
        "zoom": 14,
    }

    headers = {
        # Nominatim requires a User-Agent. You can customize this string.
        "User-Agent": "context_provider/1.0 (smart_interaction_lab)",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
            response = await client.get(NOMINATIM_URL, params=params)
            response.raise_for_status()
    except httpx.HTTPError:
        # Fall back to a very generic context
        return PlaceContext(placeType="unknown", nearbyCategories=[])

    data = response.json()

    category = (data.get("category") or "").lower()
    type_ = (data.get("type") or "").lower()
    addresstype = (data.get("addresstype") or "").lower()

    place_type: PlaceType = "unknown"

    # Very simple heuristics – can be refined later
    if "university" in type_ or "college" in type_:
        place_type = "university_campus"
    elif category in ("residential",) or addresstype in ("residential", "suburb", "village"):
        place_type = "residential"
    elif category in ("retail", "commercial") or addresstype in ("town", "city"):
        place_type = "city_center"
    elif category in ("office", "industrial", "commercial"):
        place_type = "office_area"
    elif category in ("farmland", "meadow", "forest") or addresstype in ("hamlet", "farmland"):
        place_type = "rural"
    elif "mall" in type_ or "shopping" in type_:
        place_type = "mall"
    elif "station" in type_ or "airport" in type_ or "halt" in type_:
        place_type = "transport_hub"

    # Build a simple nearbyCategories list from what we know
    nearby: List[str] = []
    for v in {category, type_, addresstype}:
        if v:
            nearby.append(v)

    return PlaceContext(
        placeType=place_type,
        rawCategory=category or None,
        rawType=type_ or None,
        nearbyCategories=nearby,
    )
