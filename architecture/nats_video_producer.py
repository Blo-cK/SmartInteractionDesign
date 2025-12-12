import asyncio
import uuid

from library.frame_grabber import FrameGrabber
from library.input_layer import InputLayerProducer





async def main():
    "You dont need to specify the Broker its automatically handeled for you"
    broker = "152.53.32.66:4222"
    source_name= "stream1"
    service_id = "example_serviceL"
    
    """ 
    Here we add the Producer and the FrameGrabber
    The Grabber is pulling the Video Frames from your webcam 
    It simulates the Agents Cameras with your own hardware
    """
    producer = InputLayerProducer(broker=broker,source_name = source_name, service= service_id )
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)
    
    await producer.connect() # you dont need to explicitly connect but its available if needed
    
    """Here we add the loop that triggers the send for Grabbed Frames"""
    try:
        while True:
            # Capture a frame and send to NATS
            await producer.send_frame(grabber,100) #send_frame will automatically connect you 
    except KeyboardInterrupt:
        print("Stopping video stream...")
    finally:
        grabber.release() #let the grabber finish its rw and then finish 
        await producer.disconnect() #disconnect from NATS
        

   
    
asyncio.run(main())


