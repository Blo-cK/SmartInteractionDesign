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
    producer_path = os.path.join(os.path.dirname(__file__), "frame_producer.py")
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
    print("🔍 Checking if frame producer is online")
    is_online = await check_producer_online("camera1_fullframe")
    
    if not is_online:
        print("Frame producer is not online -> starting it")
        await start_producer()
    else:
        print("Frame producer is online")
    
    consumer = InputLayerConsumer(
        topic="input.camera1.fullframe.gaze",
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
        # InputLayerProducer sends raw JPEG bytes (not JSON)
        frame_bytes = msg.msg.data
        
        # Get metadata from headers
        headers = msg.msg.headers or {}
        w = int(headers.get('width', 1920))
        h = int(headers.get('height', 1080))
        
        # Decode JPEG bytes → frame
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("⚠️ Failed to decode frame")
            return
        
        # Detect ALL faces in frame and get their gaze => only return yaw pitch roll using fullframe
        faces_data = gaze_detector.detect_gaze(frame, frame_width=w, frame_height=h)
        
        if len(faces_data) == 0:
            print("⚠️ No faces detected in frame")
            return
        
        # Send each detected face to Kafka
        for face_data in faces_data:
            result = {
                'face_id': face_data['face_id'],
                'gaze': {
                    'pitch': face_data['pitch'],
                    'yaw': face_data['yaw'],
                    'roll': face_data['roll'],
                }
            }
            
            await kafka.sendData(msg, result, 'gaze_detector')
            
            print(f"{face_data['face_id']}: pitch={face_data['pitch']:.2f}°, yaw={face_data['yaw']:.2f}°, roll={face_data['roll']:.2f}° → Kafka")
        
        count[0] += len(faces_data)
    
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
