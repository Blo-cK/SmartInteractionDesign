import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata
from ..service.context_service import build_snapshot

# Topic name pattern used by the shared output layer:
#   output.<source_id>.<service_id>  (lowercase)
SOURCE_ID = "contextprovider"
SERVICE_ID = "contextprovider"

async def push_loop(interval_seconds: int = 60) -> None:
    """
    Periodically builds a context snapshot and publishes it to Kafka.
    The loop keeps track of the last hash and only pushes a new message
    when the context changes, to reduce needless traffic.
    """
    producer = OutputLayerProducer()
    last_hash: Optional[str] = None

    while True:
        try:
            envelope = await build_snapshot(
                accept_language="de-DE",  # can be made configurable if needed
                location_hint=None,
            )

            # Only publish if the context content has changed
            if envelope.hash != last_hash:
                now_iso = datetime.now(timezone.utc).isoformat()
                completed_at = int(time.time())

                # Complete snapshot is sent as the result payload
                result_dict = envelope.model_dump()

                metadata = OutputLayerMetadata(
                    source_id=SOURCE_ID,
                    service_id=SERVICE_ID,
                    time_stamp=envelope.producedAt or now_iso,
                    completed_at=completed_at,
                    result=result_dict,
                )

                await producer.sendDataWithMetadata(
                    metadata=metadata,
                    result=result_dict,
                    service_id=SERVICE_ID,
                )

                print(f"[ContextProvider] Pushed new context with hash {envelope.hash}")
                last_hash = envelope.hash

        except Exception as e:
            # Keep the loop running even if sending fails once
            print(f"[ContextProvider] Error in push_loop: {e}")

        await asyncio.sleep(interval_seconds)
