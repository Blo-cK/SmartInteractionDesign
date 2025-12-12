import asyncio
import uuid

from library.audio_grabber import AudioGrabber
from library.input_layer import InputLayerProducer


async def main():
    
    "You dont need to specify the Broker its automatically handeled for you"
    broker = "152.53.32.66:4222"
    
    source_name= "stream1"
    service_id = "example_serviceL"
    
    """ 
    Here we add the Producer and the AudioGrabber
    The Grabber is pulling the Audio from your OS Audio Input Device 
    It simulates the Agents Mic with your own hardware
    
    Windows Users: This will ask you to allow Audio Devices
    
    Linux Users: Make sure your Audio Port and Drivers are up to date or this will fail
    """
    producer = InputLayerProducer(broker=broker, source_name = source_name, service= service_id)

    # Create audio grabber
    
    grabber = AudioGrabber(
        sample_rate=16000,
        channels=1,
        chunk_ms=100  # 100ms chunk = fps 10 this is also the parameter used to determine the Cps for Audio (Chunks per second)
        # Determine fps from chunk size so it matches perfectly
    )

    await producer.connect() # you dont need to explicitly connect but its available if needed
    print("Audio producer connected — streaming...")

    try:
        while True:
            await producer.send_audio_chunk(audio_grabber=grabber) #send_audio_chunk will automatically connect you 
    except KeyboardInterrupt:
        print("Stopping audio stream...")
    finally:
        grabber.release()
        await producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
