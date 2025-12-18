import asyncio
import time
from datetime import datetime, timezone

from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata


async def main():
    producer = OutputLayerProducer()

    # 1) Eindeutige IDs wählen
    #    source_id = "Quelle" im Sinne der Architektur
    #    service_id = Name deines Services
    #    => Topic: output.<source_id>.<service_id>
    source_id = "contextprovider"   # oder ein Name, den euer Team vorgibt
    service_id = "contextprovider"  # oder z.B. "environment"

    # 2) Zeitstempel bauen
    time_stamp = datetime.now(timezone.utc).isoformat()
    completed_at = int(time.time())

    # 3) Ergebnis-Payload (hier noch Dummy)
    result = {
        "message": "hello from ContextProvider via sendDataWithMetadata"
    }

    # 4) Metadata-Objekt gemäß OutputLayerMetadata
    metadata = OutputLayerMetadata(
        source_id=source_id,
        service_id=service_id,
        time_stamp=time_stamp,
        completed_at=completed_at,
        result=result,
    )

    # 5) Senden
    await producer.sendDataWithMetadata(
        metadata=metadata,
        result=result,
        service_id=service_id,
    )

    # 6) Sauber trennen
    await producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
