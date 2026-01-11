from typing import Optional

import asyncio
from fastapi import FastAPI, Header, Query

from .model.context_models import (
    LocationHint,
    ContextInput,
    ContextEnvelope,
)
from .service.context_service import build_snapshot, build_delta
from .push.push_loop import push_loop

app = FastAPI(
    title="context_provider",
    description="Python Environment Context Service (Request/Response)",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """Simple health-check endpoint used by monitoring or local checks."""
    return {"ok": True}


@app.get(
    "/context",
    response_model=ContextEnvelope,
)
async def get_context(
    accept_language: Optional[str] = Header(
        default=None, alias="Accept-Language", description="HTTP Accept-Language header"
    ),
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    city: Optional[str] = Query(default=None),
    countryCode: Optional[str] = Query(default=None),
    region: Optional[str] = Query(default=None),
) -> ContextEnvelope:
    """
    Return a full environment context snapshot.

    Location can optionally be provided via query parameters.
    If no location is given, a default / auto-detected location is used.
    """
    location_hint: Optional[LocationHint] = None
    if lat is not None and lon is not None:
        location_hint = LocationHint(
            lat=lat,
            lon=lon,
            city=city,
            countryCode=countryCode,
            region=region,
        )

    envelope = await build_snapshot(
        accept_language=accept_language,
        location_hint=location_hint,
    )
    return envelope


@app.post(
    "/context",
    response_model=ContextEnvelope,
)
async def post_context(
    body: ContextInput,
    accept_language: Optional[str] = Header(
        default=None, alias="Accept-Language", description="HTTP Accept-Language header"
    ),
) -> ContextEnvelope:
    """
    Same as GET /context but accepts a JSON body with a LocationHint.

    Useful when the caller already has a structured location object instead of query parameters.
    """
    envelope = await build_snapshot(
        accept_language=accept_language,
        location_hint=body.locationHint,
    )
    return envelope


@app.get(
    "/context/delta",
    response_model=ContextEnvelope,
)
async def get_context_delta(
    accept_language: Optional[str] = Header(
        default=None, alias="Accept-Language", description="HTTP Accept-Language header"
    ),
    sinceHash: Optional[str] = Query(default=None, description="Hash of the last known context snapshot"),
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    city: Optional[str] = Query(default=None),
    countryCode: Optional[str] = Query(default=None),
    region: Optional[str] = Query(default=None),
) -> ContextEnvelope:
    """
    Return a delta envelope compared to the given sinceHash.

    - If sinceHash is missing or equal to the current hash:
      -> data will be an empty object (no changes).
    - If sinceHash differs:
      -> data will contain the full EnvironmentContext.
    """
    location_hint: Optional[LocationHint] = None
    if lat is not None and lon is not None:
        location_hint = LocationHint(
            lat=lat,
            lon=lon,
            city=city,
            countryCode=countryCode,
            region=region,
        )

    envelope = await build_delta(
        accept_language=accept_language,
        location_hint=location_hint,
        since_hash=sinceHash,
    )
    return envelope


@app.on_event("startup")
async def startup_event() -> None:
    """
    FastAPI startup hook that starts the background Kafka push loop
    for environment context snapshots.
    """
    asyncio.create_task(push_loop())
