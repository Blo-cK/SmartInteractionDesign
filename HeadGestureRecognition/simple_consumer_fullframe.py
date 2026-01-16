"""Consumer: NATS full-frame → Head Gesture → Kafka"""
import asyncio
import cv2
import numpy as np
import sys
import os
import json
import subprocess
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from architecture.library.output_layer import OutputLayerProducer
from architecture.library.monitor_client import MonitorClient
from headgesturerecognition import HeadGestureRecognition


async def check_producer_online(service_id="camera1_fullframe", monitor_url="http://152.53.32.66:5000"):
    """Check if fullframe producer service is online via MonitorClient"""
    try:
        client = MonitorClient(base_url=monitor_url)
        status = client.get_online_status(service_id)
        if status:
            return status.online
        return False
    except Exception as e:
        print(f"Failed to check producer status: {e}")
        return False


async def start_producer():
    """Start the producer as a subprocess"""
    print("Starting frame producer")
    producer_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "GazeDetection",
        "frame_producer.py",
    )

    # Start producer as background process using the current interpreter
    subprocess.Popen(
        [sys.executable, producer_path],
        cwd=os.path.dirname(producer_path),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print("Frame producer started in new console")
    await asyncio.sleep(10)  # Wait for producer to initialize


async def run_consumer():
    print("Checking if fullframe producer is online")
    is_online = await check_producer_online("camera1.fullframe")
    
    if not is_online:
        print("Frame producer is not online -> starting it")
        await start_producer()
    else:
        print("Frame producer is online")
    
    consumer = InputLayerConsumer(
        topic="input.camera1.fullframe.gaze",
        broker="152.53.32.66:4222",
    )
    
    kafka = OutputLayerProducer(broker="152.53.32.66:9094")
    
    headgesture_recognition = HeadGestureRecognition()
    print("Head Gesture Recognition loaded")
    
    await consumer.connect()
    
    print("Processing full frames...")
    count = [0]
    headgesture_none = False
    
    async def handle_message(msg):
        # msg is InputResultWrapper, actual NATS message is in msg.msg
        frame_bytes = msg.msg.data
        headers = msg.msg.headers or {}

        # Decode JPEG bytes → frame
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode frame")
            return

        frame_h, frame_w = frame.shape[:2]

        # Run head gesture recognition on the full frame
        gesture_output = headgesture_recognition.process_frame(frame)

        if not gesture_output:
            print("No gestures detected in frame")
            return

        # Prepare and send one Kafka message per tracked face
        for track_id, gesture_data in gesture_output.items():
            track_faces = headgesture_recognition.face_tracker.tracks_store.get(track_id, [])
            bbox_info = {}
            face_crop = None

            if track_faces:
                last_face = track_faces[-1]
                loc = last_face.loc
                bbox_info = {
                    "x": loc.x1,
                    "y": loc.y1,
                    "w": loc.x2 - loc.x1,
                    "h": loc.y2 - loc.y1,
                }

                x1, y1, x2, y2 = loc.x1, loc.y1, loc.x2, loc.y2
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]

            head_gesture = gesture_data.get("head_gesture")
            confidence = gesture_data.get("confidence")
            timestamp = gesture_data.get("timestamp")
            payload_timestamp = headers.get("time_stamp") or headers.get("timestamp")

            if confidence is not None:
                confidence = float(confidence)

            # if face_crop is not None:
            #     save_folder = os.path.join(os.path.dirname(__file__), "saved_faces_fullframe")
            #     os.makedirs(save_folder, exist_ok=True)
            #     face_filename = os.path.join(save_folder, f"{track_id}.png")
            #     cv2.imwrite(face_filename, face_crop)

            result = {
                "face_id": str(track_id),
                "track_id": track_id,
                "bbox": bbox_info,
                "frame_size": {"width": frame_w, "height": frame_h},
                "head_gesture": head_gesture,
                "confidence": confidence,
                "timestamp": timestamp,
                "source_timestamp": payload_timestamp,
            }

            if head_gesture != 'none':
                await kafka.sendData(msg, result, "headgesture_recognition")
                print(f"✅ Face {track_id} | Gesture={head_gesture} | Confidence={confidence if confidence is not None else 'n/a'}")
                headgesture_none = False
            
            if head_gesture == 'none' and headgesture_none != True:
                await kafka.sendData(msg, result, "headgesture_recognition")
                print(f"✅ Face {track_id} | Gesture={head_gesture} | Confidence={confidence if confidence is not None else 'n/a'}")
                headgesture_none = True

            count[0] += 1
    
    # Use InputLayerConsumer.consume() as per README
    await consumer.consume(onFrame=handle_message)
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopping consumer... Processed {count[0]} faces")
        await consumer.disconnect()
        await kafka.disconnect()


if __name__ == "__main__":
    asyncio.run(run_consumer())
