import os
from typing import TypedDict
import httpx

class PushConfig(TypedDict):
    """Runtime settings for periodic push behaviour."""
    enabled: bool
    webhook_url: str
    interval_seconds: int

def get_push_config() -> PushConfig:
    """
    Load push settings from environment variables.

    Expected variables:
      - PUSH_ENABLED=true/false
      - PUSH_WEBHOOK_URL=<target URL>
      - PUSH_INTERVAL_SECONDS=<interval in seconds, default 600>
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
    Send the given payload as JSON to the configured webhook endpoint.
    """
    if not webhook_url:
        return

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            # Swallow errors here; callers can decide how much they care.
            return
