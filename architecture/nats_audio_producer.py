import asyncio
import uuid

from library.audio_grabber import AudioGrabber
from library.input_layer import InputLayerProducer


async def main():
    broker = "152.53.32.66:4222"
    topic = "audio.stream1"
    myid = str(uuid.uuid4())
    # Create producer
    producer = InputLayerProducer(broker=broker, topic=topic,source_name=myid)

    # Create audio grabber
    grabber = AudioGrabber(
        sample_rate=16000,
        channels=1,
        chunk_ms=100  # 100ms chunk = fps 10
    )

    await producer.connect()
    print("Audio producer connected — streaming...")

    # Determine fps from chunk size so it matches perfectly

    try:
        while True:
            await producer.send_audio_chunk(
                audio_grabber=grabber,
                sample_rate=grabber.sample_rate,
                channels=grabber.channels,
                fps=grabber.chunk_ms
            )

    except KeyboardInterrupt:
        print("Stopping audio stream...")

    finally:
        grabber.release()
        await producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
