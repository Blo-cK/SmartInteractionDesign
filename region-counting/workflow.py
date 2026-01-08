import asyncio
import cv2
import numpy as np

from architecture.library.frame_grabber import FrameGrabber
from architecture.library.input_layer import InputLayerConsumer, InputLayerProducer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer

from ultralytics import YOLO


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

    previous_count = {"people_in_region": 0}
    model = YOLO("yolo11n.pt")
    # region coordinates
    region_x1, region_x2 = 650, 1100
    region_y1, region_y2 = 250, 650

    async def count_people_in_region(frame):
        results = model.track(frame, persist=True, classes=[0], conf=0.35, verbose=False)
        people_in_region = 0

        boxes = results[0].boxes
        # iterate over all tracked boxes in frame
        for box in boxes:
            # grab center of tracked box
            x_center = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

            # check if center of tracked box is inside the region
            if region_x1 <= x_center <= region_x2 and region_y1 <= y_center <= region_y2:
                people_in_region += 1

        # show region in frame
        color = (145, 0, 175)
        thickness = 2
        cv2.rectangle(frame, (region_x1, region_y1), (region_x2, region_y2), color, thickness)
        cv2.putText(frame, f"Count: {people_in_region}", (region_x1, region_y1-10),
        cv2.FONT_ITALIC, 0.9, color, thickness)

        return {"people_in_region": people_in_region}, frame

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
        processed_result, frame = await count_people_in_region(frame)

        # Show frame
        if frame is not None:
            cv2.imshow("Async Consumer Stream", frame)
            cv2.waitKey(1)

        # only send data if current value differs from previously sent value
        if previous_count["people_in_region"] != processed_result["people_in_region"]:
            previous_count["people_in_region"] = processed_result["people_in_region"]

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
    topic = "input.cameras.camera1"
    service_name = "region_counting"

    output_producer = OutputLayerProducer()

    try:
        await asyncio.gather(
            producer_task(topic, "RegionCountingSensor"),
            consumer_task(topic, output_producer, service_name)
        )
    except KeyboardInterrupt:
        print("Shutting down workflow...")
    finally:
        await output_producer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
