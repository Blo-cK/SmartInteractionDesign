import asyncio
import uuid

from library.frame_grabber import FrameGrabber
from library.input_layer import InputLayerProducer





async def main():
    "You dont need to specify the Broker its automatically handeled for you"
    broker = "152.53.32.66:4222"
    topic = "cams.cam1" # pls use the same structure and Prefix your cam with cams. and then add your cam1 , 2,3,4...
    myid = str(uuid.uuid4()) #use this for save id selection or specify your own beware of collisions
    
    
    producer = InputLayerProducer(broker=broker,topic=topic,source_name=myid )
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)
    await producer.connect() # you dont need to explicitly connect but its available
    try:
        while True:
            # Capture a frame and send to NATS
            await producer.send_frame(grabber,100) #send_frame will automatically connect you 
    except KeyboardInterrupt:
        print("Stopping video stream...")
    finally:
        grabber.release()
        await producer.disconnect()
    
    
   
    
asyncio.run(main())


