import os
import json
from typing import TypedDict

import httpx


class PushConfig(TypedDict):
    """Configuration for server-side push behaviour."""
    enabled: bool
    webhook_url: str
    interval_seconds: int


def get_push_config() -> PushConfig:
    """
    Read push configuration from environment variables.

    - PUSH_ENABLED=true/false
    - PUSH_WEBHOOK_URL=<URL of the message service or webhook>
    - PUSH_INTERVAL_SECONDS=600
    """
    enabled_str = os.getenv("PUSH_ENABLED", "false").lower()
    enabled = enabled_str in ("1", "true", "yes", "on")

    webhook_url = os.getenv("PUSH_WEBHOOK_URL", "").strip()
    interval_seconds_str = os.getenv("PUSH_INTERVAL_SECONDS", "600")

    try:
        interval_seconds = int(interval_seconds_str)
    except ValueError:
        interval_seconds = 600

    return {
        "enabled": enabled,
        "webhook_url": webhook_url,
        "interval_seconds": interval_seconds,
    }


async def send_context_to_webhook(webhook_url: str, payload: dict) -> None:
    """
    Send the context payload as JSON to a generic webhook.
    This can be a Slack webhook, your own message service, etc.
    """
    if not webhook_url:
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # For a simple JSON-consuming endpoint, this is fine.
            # For Slack you might need to wrap it, e.g. {"text": "```json\n...```"}
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            # In a real system you would log this or add retry logic.
            return
