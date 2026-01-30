import asyncio
import uuid
import cv2
import numpy as np
from datetime import datetime
import os, sys

from architecture.library.frame_grabber import FrameGrabber
from architecture.library.input_layer import InputLayerConsumer, InputLayerProducer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer

import cv2
import numpy as np
import time

######################################################################
# PRODUCER TASK – captures camera frames and publishes to NATS
######################################################################
async def producer_task(service_name: str, source_name: str):
    producer = InputLayerProducer(service=service_name, source_name=source_name)
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)

    await producer.connect()

    try:
        while True:
            await producer.send_frame(grabber, 1) 
    except asyncio.CancelledError:
        print("Producer stopped.")
    finally:
        grabber.release()
        await producer.disconnect()


######################################################################
# CONSUMER TASK – receives frames, processes them, sends metadata
######################################################################

async def consumer_task(topic: str, output_producer: OutputLayerProducer, service_name: str):
    consumer = InputLayerConsumer(topic=topic)
    previous_state = {"brightness_scaled": -1} 
    last_send_time = 0

    def calculate_brightness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0

    def calculate_brightness_histogram(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        median_val = np.searchsorted(cdf, cdf[-1] * 0.5)
        return median_val / 255.0

    def calculate_brightness_centroid(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        brightness_value = np.sum(hist * np.arange(256)) / np.sum(hist)
        return brightness_value / 255.0

    def map_brightness_to_scale(brightness, min_val=0.0, max_val=1.0):
        normalized = np.clip((brightness - min_val) / (max_val - min_val), 0.0, 1.0)
        return int(round(1 + normalized * 9))

    async def video_brightness(frame):
        brightness = calculate_brightness(frame)
        scale = map_brightness_to_scale(brightness)

        # Brightness values between 0 and 10
        return {
            #"brightness": float(brightness),
            "brightness_scaled": scale
        }

    async def handle_message(result: InputResultWrapper):
        nonlocal last_send_time
        current_time = time.time()

        if current_time - last_send_time < 1.0:
            return

        # JPEG -> numpy
        data = np.frombuffer(result.msg.data, np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return

        processed_result = await video_brightness(frame)
        current_scale = processed_result["brightness_scaled"]

        # Only send if the value has changed
        if current_scale != previous_state["brightness_scaled"]:
            previous_state["brightness_scaled"] = current_scale
            last_send_time = current_time

            # --- Console output ---
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] [Brightness] Sending update: {current_scale}/10")
            # -----------------------------------

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
    source_name = "camera1.video_brightness"
    service_name = "video_brightness"
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