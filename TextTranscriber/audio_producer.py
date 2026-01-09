"""NATS Producer: Microphone -> Audio Chunks -> NATS"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerProducer
from architecture.library.audio_grabber import AudioGrabber


async def run_producer():
    """Send full micro chunks to NATS"""
    producer = InputLayerProducer(
        source_name="microphone1.audio",
        service="audio",
        broker="152.53.32.66:4222"
    )
    
    grabber = AudioGrabber(chunk_ms=10000) #10s-chunks
    
    await producer.connect()
    
    print("Streaming full audio to NATS...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            await producer.send_audio_chunk(grabber)
    except KeyboardInterrupt:
        print("Stopped audio producer by user.")
    finally:
        grabber.release()
        await producer.disconnect()

if __name__ == "__main__":
    asyncio.run(run_producer())
