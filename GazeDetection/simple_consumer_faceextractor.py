"""Consumer: NATS → Gaze Detection → Kafka"""
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
from gazedetection import GazeDetector


async def check_producer_online(service_id="camera1", monitor_url="http://152.53.32.66:5000"):
    """Check if face extractor producer service is online via MonitorClient"""
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
    """Start the producer as a subprocess (only with venv working on windows)"""
    print("Starting frame producer")
    producer_path = os.path.join(os.path.dirname(__file__), "face_extractor_producer.py")
    venv_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    
    # Start producer as background process using venv
    subprocess.Popen(
        [venv_python, producer_path],
        cwd=os.path.dirname(__file__),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print("Frame producer started in new console (using venv)")
    await asyncio.sleep(10)  # Wait for producer to initialize


async def run_consumer():
    print("Checking if frame producer is online")
    is_online = await check_producer_online("camera1")
    
    if not is_online:
        print("Frame producer is not online -> starting it")
        await start_producer()
    else:
        print("Frame producer is online")
    
    consumer = InputLayerConsumer(
        topic="input.camera1.faceextractor",
        broker="152.53.32.66:4222"
    )
    
    kafka = OutputLayerProducer(
        broker="152.53.32.66:9094"
    )
    
    gaze_detector = GazeDetector()
    print("🔧 Gaze Detector (FaceExtractor Interface) loaded")
    
    await consumer.connect()
    
    print("Processing faces...")
    count = [0]
    
    async def handle_message(msg):
        # msg is InputResultWrapper, actual NATS message is in msg.msg
        json_data = msg.msg.data
        
        # Parse JSON data package from face extractor
        import base64
        try:
            data_package = json.loads(json_data.decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Failed to parse JSON: {e}")
            return
        
        # Extract face image from base64
        face_image_b64 = data_package.get('face_image')
        if not face_image_b64:
            print("⚠️ No face_image in data")
            return
        
        face_bytes = base64.b64decode(face_image_b64)
        nparr = np.frombuffer(face_bytes, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_img is None:
            print("⚠️ Failed to decode face image")
            return
        
        # Get metadata
        face_id = data_package.get('face_id', 'unknown')
        bbox_info = data_package.get('bbox', {})
        frame_size = data_package.get('frame_size', {})
        
        w = frame_size.get('width', 1920)
        h = frame_size.get('height', 1080)
        
        # Detect head pose from cropped face using faceextractor's face crop
        faces_data = gaze_detector.detect_gaze(face_img, frame_width=w, frame_height=h)
        
        if len(faces_data) == 0:
            print(f"⚠️ No face detected for {face_id}")
            return
        
        # Use first detected face (should only be one in crop)
        face_data = faces_data[0]
        
        # Transform head position from cropped face coordinates to original frame coordinates
        crop_head_pos = face_data.get('head_position', {'x': 0.5, 'y': 0.5})
        
        # Get bbox coordinates
        bbox_x = bbox_info.get('x', 0)
        bbox_y = bbox_info.get('y', 0)
        bbox_w = bbox_info.get('w', w)
        bbox_h = bbox_info.get('h', h)
        
        # Transform: cropped position (0-1) -> bbox position -> frame position (0-1)
        frame_head_x = (bbox_x + crop_head_pos['x'] * bbox_w) / w
        frame_head_y = (bbox_y + crop_head_pos['y'] * bbox_h) / h
        
        head_position = {
            'x': round(frame_head_x, 4),
            'y': round(frame_head_y, 4)
        }
        
        # Send to Kafka (only for debugging/demo => not really used later (is merged into GazeDetector of simple_consumer_gaze))
        result = {
            'person_id': face_id,
            'head_position': head_position,
            'bbox': bbox_info
        }
        
        await kafka.sendData(msg, result, 'gaze_detector_faceextractor')
        
        print(f"✅ {face_id}: head_position x={head_position['x']:.4f}, y={head_position['y']:.4f} (frame coords) → Kafka")
        
        count[0] += 1
    
    # Use InputLayerConsumer.consume() as per README
    await consumer.consume(onFrame=handle_message)
    
    # Keep running indefinitely
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping consumer... Processed {count[0]} faces")
        await consumer.disconnect()
        await kafka.disconnect()


if __name__ == "__main__":
    asyncio.run(run_consumer())
