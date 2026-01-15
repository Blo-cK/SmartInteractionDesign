import asyncio
import cv2
import numpy as np

from architecture.library.frame_grabber import FrameGrabber
from architecture.library.input_layer import InputLayerConsumer, InputLayerProducer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer

from ultralytics import solutions


######################################################################
# PRODUCER TASK – captures camera frames and publishes to NATS
######################################################################
async def producer_task(service_name: str, source_name: str):
    """
    This function uses the Library to create a InputLayerProducer and a Frame Grabber.
    The FrameGrabber is used to get data from your Camera.
    The InputLayerProducer is used to send the Frames into the NATS (30 FPS)
    This is basically used to simulate the "real" camera of the agent.
    Consumers can subscribe to the topic to get the data out of the NATS.
    """
    producer = InputLayerProducer(service=service_name, source_name=source_name)
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

    previous_count = {"total_people": 0}

    async def get_heatmap_data(frame):
        heatmap = solutions.Heatmap(
            show=False,  # display the output
            model="yolo11n.pt",  # model for heatmap
            colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
            classes=[0],  # only detect people
            verbose=False # mute auto console prints
        )
        results = heatmap(frame)
        total_people = results.total_tracks

        return {"total_people": total_people}, frame

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

        # ML Processing
        processed_result, frame = await get_heatmap_data(frame)

        # Show frame
        if frame is not None:
            cv2.imshow("Async Consumer Stream", frame)
            cv2.waitKey(1)

        # only send data if current value differs from previously sent value
        if previous_count["total_people"] != processed_result["total_people"]:
            previous_count["total_people"] = processed_result["total_people"]

            # send Output Layer
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
    source_name = "camera1.heatmap"
    service_name = "heatmap"
    producer_topic = f"input.{source_name}.{service_name}".lower()

    output_producer = OutputLayerProducer()

    try:
        await asyncio.gather(
            producer_task(service_name, source_name),
            consumer_task(producer_topic, output_producer, service_name)
        )
    except KeyboardInterrupt:
        print("Shutting down workflow...")
    finally:
        await output_producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
