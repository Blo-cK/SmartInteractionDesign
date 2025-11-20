import asyncio
from datetime import datetime

from library.output_layer import OutputLayerMetadata, OutputLayerProducer, OutputLayerReceiver




async def onMetadataReceived(metadata):
    print("Got metadata:", metadata)

async def receive_loop():
    
    receiver = OutputLayerReceiver(group_id="test-group")
    try:
        await receiver.receiveData("camera1", "object_detection", onMetadataReceived)
    finally:
        await receiver.disconnect()

async def send_once():
    producer = OutputLayerProducer()
    metadata = OutputLayerMetadata(
        source_name="sensor1",
        frame_id="34",
        service="matteo_test_service",
        timestamp_producer=datetime.utcnow().isoformat(),
        result={"status": "ok", "objects": ["car", "person"]}
    )
    await producer.sendData(metadata)
    await producer.disconnect()

async def main():
    while True:
        choice = input("Drücke [s] zum Senden, [r] zum Empfangen, [q] zum Beenden: ").strip().lower()
        if choice == "s":
            await send_once()
        elif choice == "r":
            print("Starte Receiver (Strg+C zum Abbrechen)...")
            try:
                await receive_loop()
            except KeyboardInterrupt:
                print("Receiver gestoppt.")
        elif choice == "q":
            print("Beende...")
            break
        else:
            print("Ungültige Eingabe.")

if __name__ == "__main__":
    asyncio.run(main())
