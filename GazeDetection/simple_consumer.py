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
            return data.get('online', False)
        return False
    except Exception as e:
        print(f"Failed check producer status: {e}")
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
        topic="gaze.frames",
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
        face_data = msg.msg.data
        meta = msg.msg.headers or {}
        
        # Decode face image
        face_array = np.frombuffer(face_data, dtype=np.uint8)
        face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
        
        if face_img is None:
            return
        
        # Parse bbox and frame_size from metadata
        bbox_info = json.loads(meta.get('bbox', '{}'))
        frame_size = json.loads(meta.get('frame_size', '{}'))
        
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
        face_id = meta.get('face_id', 'unknown')
        result = {
            'face_id': face_id,
            'bbox': bbox_info,
            'frame_size': frame_size,
            'gaze': gaze
        }
        
        await kafka.sendData(meta, result, 'gaze_detector')
        
        print(f"✅ {face_id}: pitch={gaze['pitch']}°, yaw={gaze['yaw']}° → Kafka")
        print(meta, result)
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
