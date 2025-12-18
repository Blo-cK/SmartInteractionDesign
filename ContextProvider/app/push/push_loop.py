import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata
from ContextProvider.app.service.context_service import build_snapshot


# IDs used to build the Kafka topic:
# topic = output.<source_id>.<service_id>  (all lowercased)
# → "output.contextprovider.contextprovider"
# Make sure this topic exists or is allowed to be auto-created.
SOURCE_ID = "contextprovider"
SERVICE_ID = "contextprovider"


async def push_loop(interval_seconds: int = 60) -> None:
    """
    Background loop that periodically builds a context snapshot
    and pushes it into the shared Kafka output layer.

    - Uses OutputLayerMetadata, which is what the architecture expects.
    - Sends only when the context hash changes (to avoid spamming Kafka).
    """
    producer = OutputLayerProducer()
    last_hash: Optional[str] = None

    while True:
        try:
            # 1) Build a fresh context snapshot (your existing logic)
            envelope = await build_snapshot(
                accept_language="de-DE",  # you can make this configurable later
                location_hint=None,
            )

            # 2) Only push if context has changed
            if envelope.hash != last_hash:
                now_iso = datetime.now(timezone.utc).isoformat()
                completed_at = int(time.time())

                # Full envelope (snapshot) as the result payload
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
            # Do not crash the loop on errors; just log and retry later.
            print(f"[ContextProvider] Error in push_loop: {e}")

        await asyncio.sleep(interval_seconds)
