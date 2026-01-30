import asyncio
import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional

from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata
from ContextProvider.app.service.context_service import build_snapshot
from ContextProvider.app.model.context_models import EnvironmentContext

# Topic layout: output.<source_id>.<service_id>
SOURCE_ID = "contextprovider"
SERVICE_ID_DYNAMIC = "context_dynamic"
SERVICE_ID_STATIC = "context_static"


def _hash_payload(payload: dict) -> str:
    """Create a short stable hash over a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def extract_static_context(env: EnvironmentContext) -> dict:
    """
    Extract the slowly changing parts of the environment context.

    This includes location, holidays, holiday summary, basic day meta
    and coarse place information. And also local events.
    """
    return {
        "location": env.location.model_dump(),
        "dayMeta": env.dayMeta.model_dump(),
        "holidays": [h.model_dump() for h in env.holidays],
        "holidaySummary": env.holidaySummary.model_dump()
        if getattr(env, "holidaySummary", None) is not None
        else None,
        "placeContext": env.placeContext.model_dump()
        if env.placeContext is not None
        else None,
        "locale": env.locale.model_dump(),
        "events": env.events.model_dump() if hasattr(env, "events") and env.events else None,
    }


def extract_dynamic_context(env: EnvironmentContext) -> dict:
    """
    Extract the fast-changing parts of the environment context.

    This focuses on time, weather, daylight and comfort estimation.
    """
    return {
        "dateTime": env.dateTime.model_dump(),
        "weather_current": env.weather_current.model_dump()
        if env.weather_current is not None
        else None,
        "weather_forecast": [fp.model_dump() for fp in env.weather_forecast],
        "weather_tomorrow": env.weather_tomorrow.model_dump()
        if env.weather_tomorrow is not None
        else None,
        "daylight": env.daylight.model_dump()
        if env.daylight is not None
        else None,
        "comfort": env.comfort.model_dump()
        if env.comfort is not None
        else None,
    }


async def push_dynamic_loop(interval_seconds: int = 900) -> None:
    """
    Periodically push dynamic context (time, weather, daylight, comfort)
    to the output layer.

    - Runs once immediately on startup, then every `interval_seconds`.
    - Uses its own hash so that updates are only sent when something changes.
    """
    producer = OutputLayerProducer()
    last_hash: Optional[str] = None

    while True:
        try:
            envelope = await build_snapshot(
                accept_language="de-DE",
                location_hint=None,
            )

            # envelope.data is an EnvironmentContext in our service
            if isinstance(envelope.data, EnvironmentContext):
                env = envelope.data
            else:
                env = EnvironmentContext(**envelope.data)

            dynamic_payload = extract_dynamic_context(env)
            payload_hash = _hash_payload(dynamic_payload)

            if payload_hash != last_hash:
                now_iso = datetime.now(timezone.utc).isoformat()
                completed_at = int(time.time())

                metadata = OutputLayerMetadata(
                    source_id=SOURCE_ID,
                    service_id=SERVICE_ID_DYNAMIC,
                    time_stamp=envelope.producedAt or now_iso,
                    completed_at=completed_at,
                    result=dynamic_payload,
                )

                await producer.sendDataWithMetadata(
                    metadata=metadata,
                    result=dynamic_payload,
                    service_id=SERVICE_ID_DYNAMIC,
                )

                print(
                    f"[ContextProvider] Pushed dynamic context "
                    f"(service_id={SERVICE_ID_DYNAMIC}, hash={payload_hash})"
                )
                last_hash = payload_hash

        except Exception as e:
            # Do not kill the loop on errors; just log and retry later.
            print(f"[ContextProvider] Error in push_dynamic_loop: {e}")

        # First run happens before this sleep, so push is immediate on startup
        await asyncio.sleep(interval_seconds)


async def push_static_loop(interval_seconds: int = 86400) -> None:
    """
    Periodically push static context (location, holidays, place, locale)
    to the output layer.

    - Runs once immediately on startup, then every `interval_seconds`.
    - Uses a separate hash so that updates are only sent when something
      in the static payload actually changed.
    """
    producer = OutputLayerProducer()
    last_hash: Optional[str] = None

    while True:
        try:
            envelope = await build_snapshot(
                accept_language="de-DE",
                location_hint=None,
            )

            if isinstance(envelope.data, EnvironmentContext):
                env = envelope.data
            else:
                env = EnvironmentContext(**envelope.data)

            static_payload = extract_static_context(env)
            static_hash = _hash_payload(static_payload)

            if static_hash != last_hash:
                now_iso = datetime.now(timezone.utc).isoformat()
                completed_at = int(time.time())

                metadata = OutputLayerMetadata(
                    source_id=SOURCE_ID,
                    service_id=SERVICE_ID_STATIC,
                    time_stamp=envelope.producedAt or now_iso,
                    completed_at=completed_at,
                    result=static_payload,
                )

                await producer.sendDataWithMetadata(
                    metadata=metadata,
                    result=static_payload,
                    service_id=SERVICE_ID_STATIC,
                )

                print(
                    f"[ContextProvider] Pushed static context "
                    f"(service_id={SERVICE_ID_STATIC}, hash={static_hash})"
                )
                last_hash = static_hash

        except Exception as e:
            print(f"[ContextProvider] Error in push_static_loop: {e}")

        # First run happens before this sleep, so push is immediate on startup
        await asyncio.sleep(interval_seconds)


async def push_loop() -> None:
    """
    Convenience wrapper that runs both the dynamic and static loops in parallel.

    This is what is called from the FastAPI startup hook.
    """
    await asyncio.gather(
        push_dynamic_loop(),
        push_static_loop(),
    )
