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
from gazedetection import GazeDetector


async def check_producer_online(service_id="camera1", monitor_url="http://152.53.32.66:5000"):
    """Check if face extractor producer service is online via REST API"""
    try:
        response = requests.get(f"{monitor_url}/api/services/input/monitor/{service_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # API returns nested dict: {'service_id': {'online': True, 'last_seen': ...}}
            service_data = data.get(service_id, {})
            is_online = service_data.get('online', False)
            return is_online
        print(f"Monitor returned status code: {response.status_code}")
        return False
    except Exception as e:
        print(f"Failed to check producer status: {e}")
        return False


async def start_producer():
    """Start the producer as a subprocess"""
    print("Starting face producer")
    producer_path = os.path.join(os.path.dirname(__file__), "gaze_nats_producer.py")
    conda_env = "emotionsdetektion_elena_ryumina"
    
    # Start producer as background process
    subprocess.Popen(
        [
            "C:/anaconda3/Scripts/conda.exe", "run", "-n", conda_env,
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
    
    consumer = InputLayerConsumer(
        topic="input.faceextractor.frames",
        broker="152.53.32.66:4222"
    )
    
    kafka = OutputLayerProducer(
        broker="152.53.32.66:9094"
    )
    
    gaze_detector = GazeDetector()
    print("🔧 Gaze Detector loaded")
    
    await consumer.connect()
    
    print("🎯 Processing faces...")
    count = [0]
    
    async def handle_message(msg):
        # msg is InputResultWrapper, actual NATS message is in msg.msg
        json_data = msg.msg.data
        
        # Parse JSON data package
        import base64
        data_package = json.loads(json_data.decode('utf-8'))
        
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
        
        # Gaze detection
        gaze = gaze_detector.detect_gaze(face_img, bbox, w, h)
        
        # Send to Kafka
        result = {
            'face_id': face_id,
            'bbox': bbox_info,
            'frame_size': frame_size,
            'gaze': gaze
        }
        
        await kafka.sendData(msg, result, 'gaze_detector')
        
        # Get architecture metadata from headers
        arch_meta = msg.msg.headers or {}
        print(f"✅ {face_id}: pitch={gaze['pitch']}°, yaw={gaze['yaw']}° → Kafka")
        print(f"📋 Architecture metadata: {arch_meta}")
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
