""" 
This is an example workflow - from producing a video using a sensor (e.g your Camera) to processing it and sending
the result to the output layer.
"""
import asyncio
import uuid
import cv2
import numpy as np
from datetime import datetime
import os, sys

from architecture.library.frame_grabber import FrameGrabber
from architecture.library.input_layer import InputLayerConsumer, InputLayerProducer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer





######################################################################
# PRODUCER TASK – captures camera frames and publishes to NATS
######################################################################
async def producer_task(topic: str, source_name: str):
    """
    This function uses the Library to create a InputLayerProducer and a Frame Grabber.
    The FrameGrabber is used to get data from your Camera.
    The InputLayerProducer is used to send the Frames into the NATS (30 FPS)
    This is basically used to simulate the "real" camera of the agent.
    Consumers can subscribe to the topic to get the data out of the NATS.
    """
    producer = InputLayerProducer(topic=topic, source_name=source_name)
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)

    await producer.connect()

    try:
        while True:
            await producer.send_frame(grabber, 30)
    except asyncio.CancelledError:
        print("Producer stopped.")
    finally:
        grabber.release()
        await producer.disconnect()


######################################################################
# CONSUMER TASK – receives frames, processes them, sends metadata
######################################################################

async def consumer_task(topic: str, output_producer: OutputLayerProducer, service_name: str):
    """
    This function will retrieve the Data (in this case frames) which were put in the NATS by the InputLayerProducer.
    A InputLayerConsumer is used to retrieve the data out of the NATS.
    """
    consumer = InputLayerConsumer(topic=topic)

    async def fake_processing(frame):
        """YOUR ML STUFF GOES HERE! - 0.01 TO SIMULATE PROCESSING - RETURN YOUR RESULT AS JSON OR DICT!"""
        await asyncio.sleep(0.01)

        return {"status": "ok", "objects": ["car", "person"]}

    async def handle_message(result: InputResultWrapper):
        """
        process of nats messages
        This method processes only data which comes as jpeg, if another format is send like mp3 this method needs to be adapted.
        It will process the incoming frame and after that it will use the OutputLayerProducer to send the result
        to the Output Hub.
        """
        # JPEG -> numpy
        
        data = np.frombuffer(result.msg.data, np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Show frame
        if frame is not None:
            cv2.imshow("Async Consumer Stream", frame)
            cv2.waitKey(1)

        # ML Processing
        processed_result = await fake_processing(frame)

        # send  Output Layer
        await output_producer.sendData(
            input_result=result,
            result=processed_result,
            service_id=service_name
        )


    await consumer.connect()
    await consumer.consume(onFrame=handle_message)


######################################################################
# MAIN WORKFLOW
######################################################################
async def main():
    topic = "input.cameras.camera1"
    service_name = "example_service"

    output_producer = OutputLayerProducer()

    try:
        await asyncio.gather(
            producer_task(topic, "Sensor1"),
            consumer_task(topic, output_producer, service_name)
        )
    except KeyboardInterrupt:
        print("Shutting down workflow...")
    finally:
        await output_producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
