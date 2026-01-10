"""Consumer: NATS → Head Gesture → Kafka"""
import asyncio
import cv2
import numpy as np
import sys
import os
import json
import subprocess
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from architecture.library.output_layer import OutputLayerProducer
from architecture.library.monitor_client import MonitorClient
from headgesturerecognition import HeadGestureRecognition

def handle_async_exception(loop, context):
    print("\n ASYNC EXCEPTION")
    print(context.get("message"))
    exception = context.get("exception")
    if exception:
        traceback.print_exception(
            type(exception),
            exception,
            exception.__traceback__
        )

asyncio.get_event_loop_policy().get_event_loop().set_exception_handler(
    handle_async_exception
)

async def check_producer_online(service_id="camera1", monitor_url="http://152.53.32.66:5000"):
    """Check if face extractor producer service is online via MonitorClient"""
    try:
        client = MonitorClient(base_url=monitor_url)
        status = client.get_online_status(service_id)
        if status:
            return status.online
        return False
    except Exception:
        print("Failed to check producer status:")
        traceback.print_exc()
        return False

async def start_producer():
    """Start the producer as a subprocess"""
    print("Starting face producer")
    producer_path = os.path.join(os.path.dirname(__file__), "headgesture_nats_producer.py")
    conda_env = "emotionsdetektion_elena_ryumina"
    conda_path = "C:/Users/Johannes/miniconda3/Scripts/conda.exe" #"C:/anaconda3/Scripts/conda.exe"
    
    # Start producer as background process
    subprocess.Popen(
        [
            conda_path, "run", "-n", conda_env,
            "--no-capture-output", "python", producer_path
        ],
        cwd=os.path.dirname(__file__),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print("Face extractor Producer started in new console")
    await asyncio.sleep(10)  # Wait for producer to initialize

async def run_consumer():
    print("🔍 Checking if face extractor producer is online")
    is_online = await check_producer_online("camera1")
    
    if not is_online:
        print("Face extractor producer is not online -> starting it")
        await start_producer()
    else:
        print("Face extractor producer is online")


async def run_consumer():
    consumer = InputLayerConsumer(
        topic="input.camera1.faceextractor",
        broker="152.53.32.66:4222"
    )
    
    kafka = OutputLayerProducer(
        broker="152.53.32.66:9094"
    )
    
    headgesture_recognition = HeadGestureRecognition()
    print("🔧 Head Gesture Recognition loaded")
    
    await consumer.connect()
    
    print("🎯 Processing faces...")
    count = [0]
    
    async def handle_message(msg):
        try:
            # msg is InputResultWrapper, actual NATS message is in msg.msg
            json_data = msg.msg.data
            
            # Parse JSON data package
            import base64
            data_package = json.loads(json_data.decode('utf-8'))
            # Optional timestamp from producer payload
            source_timestamp = data_package #data_package.get('timestamp')
            
            # Extract face image from base64
            face_bytes = base64.b64decode(data_package['face_image'])
            face_array = np.frombuffer(face_bytes, dtype=np.uint8)
            face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
            
            if face_img is None:
                return
            
            # Extract metadata from data package
            bbox_info = data_package.get('bbox', {})
            frame_size = data_package.get('frame_size', {})
            face_id = data_package.get('face_id', 'unknown')
            
            bbox = None
            if bbox_info:
                bbox = (bbox_info['x'], bbox_info['y'],
                    bbox_info['w'], bbox_info['h'])
            
            # Get frame dimensions
            w = frame_size.get('width', 1920)
            h = frame_size.get('height', 1080)
            
            # headgesture detection
            gesture_output = headgesture_recognition.process_frame(face_img)

            if not gesture_output:
                print(f"No gesture detected for face {face_id}")
                return

            # Extract first (and only) track result
            track_id, gesture_data = next(iter(gesture_output.items()))

            head_gesture = gesture_data.get("head_gesture")
            confidence = gesture_data.get("confidence")
            timestamp = gesture_data.get("timestamp")

            # Prefer payload timestamp if present
            if source_timestamp is not None:
                payload_timestamp = source_timestamp
            else:
                payload_timestamp = None

            # Tensor → float (VERY IMPORTANT)
            if confidence is not None:
                confidence = float(confidence)

            # --------------------------------------------
            # Prepare Kafka result
            # --------------------------------------------
            result = {
                "face_id": face_id,
                "track_id": track_id,
                "bbox": bbox_info,
                "frame_size": frame_size,
                "head_gesture": head_gesture,
                "confidence": confidence,
                "timestamp": timestamp,
                "source_timestamp": payload_timestamp
            }

            await kafka.sendData(
                msg,
                result,
                "headgesture_recognition"
            )

            print(
                f"✅ Face {face_id} | "
                f"Track {track_id} | "
                f"Gesture={head_gesture} | "
                f"Confidence={confidence:.3f}"
            )
            count[0] += 1

        except Exception:
            print("\n ERROR INSIDE handle_message")
            traceback.print_exc()
    
    # Use InputLayerConsumer.consume() as per README
    await consumer.consume(onFrame=handle_message)
    
    # Keep running indefinitely
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopping consumer... Processed {count[0]} faces")
        await consumer.disconnect()
        await kafka.disconnect()


if __name__ == "__main__":
    print("🚨 ENTERING MAIN")
    asyncio.run(run_consumer())
