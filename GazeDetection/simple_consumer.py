"""Consumer: NATS → Gaze Detection"""
import asyncio
import cv2
import numpy as np
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from gazedetection import GazeDetector


async def run_consumer(num_frames=10):
    consumer = InputLayerConsumer(
        topic="gaze.frames",
        broker="152.53.32.66:4222"
    )
    
    gaze_detector = GazeDetector()
    print("🔧 Gaze Detector loaded")
    
    await consumer.connect()
    
    print("🎯 Processing faces...")
    count = [0]
    
    async def process_message(msg):
        face_data = msg.data
        meta = msg.headers or {}
        
        # Decode face image
        face_array = np.frombuffer(face_data, dtype=np.uint8)
        face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
        
        if face_img is None:
            return
        
        # Parse bbox from metadata
        bbox_info = json.loads(meta.get('bbox', '{}'))
        bbox = None
        if bbox_info:
            bbox = (bbox_info['x'], bbox_info['y'],
                   bbox_info['w'], bbox_info['h'])
        
        # Get frame dimensions
        frame_size = bbox_info.get('frame_size', {})
        w = frame_size.get('width', 1920)
        h = frame_size.get('height', 1080)
        
        # Gaze detection
        gaze = gaze_detector.detect_gaze(face_img, bbox, w, h)
        
        # Print result
        face_id = meta.get('face_id', 'unknown')
        print(f"✅ {face_id}: pitch={gaze['pitch']}°, yaw={gaze['yaw']}°")
        
        count[0] += 1
    
    await consumer.consume(onFrame=process_message)
    
    # Wait for all messages
    await asyncio.sleep(30)
    
    try:
        await consumer.disconnect()
    except Exception as e:
        pass
    
    print(f"✅ Done! Processed {count[0]} faces")


if __name__ == "__main__":
    asyncio.run(run_consumer(10))
