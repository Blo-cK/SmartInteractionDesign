import asyncio
from typing import Optional

from push_client import get_push_config, send_context_to_webhook
from ContextProvider.app.service.context_service import build_snapshot
from ContextProvider.app.model.context_models import LocationHint


async def push_loop() -> None:
    """
    Background loop that periodically builds a context snapshot
    and pushes it to an upstream message service.

    The loop:
    - reads configuration from environment (PUSH_* variables)
    - builds a snapshot via build_snapshot(...)
    - compares hashes to only push on changes
    - sleeps for the configured interval
    """
    cfg = get_push_config()

    if not cfg["enabled"]:
        # Push is disabled, exit silently
        return

    webhook_url = cfg["webhook_url"]
    interval = cfg["interval_seconds"]

    if not webhook_url:
        # No target configured -> nothing to do
        return

    last_hash: Optional[str] = None

    # Optional: fixed location hint; you can change this to None or something dynamic if needed.
    location_hint: Optional[LocationHint] = None

    while True:
        try:
            # Build a fresh snapshot. You can change the default language if needed.
            envelope = await build_snapshot(
                accept_language="de-DE",
                location_hint=location_hint,
            )

            # Only push if something has changed since the last push
            if envelope.hash != last_hash:
                payload = {
                    "source": "context_provider",
                    "payload": envelope.model_dump(),
                }
                await send_context_to_webhook(webhook_url, payload)
                last_hash = envelope.hash
        except Exception:
            # In a real application you would log the exception.
            # For now we just swallow it to keep the loop running.
            pass

        await asyncio.sleep(interval)
