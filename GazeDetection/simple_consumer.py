"""Consumer: NATS → Gaze Detection → Kafka"""
import asyncio
import cv2
import numpy as np
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from architecture.library.output_layer import OutputLayerProducer
from gazedetection import GazeDetector


async def run_consumer(num_frames=10):
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
        
        # Send to Kafka
        face_id = meta.get('face_id', 'unknown')
        result = {
            'face_id': face_id,
            'bbox': bbox_info,
            'gaze': gaze
        }
        
        await kafka.sendMetadata(meta, result, 'gaze_detector')
        
        print(f"✅ {face_id}: pitch={gaze['pitch']}°, yaw={gaze['yaw']}° → Kafka")
        print(meta, result)
        count[0] += 1
    
    # Use InputLayerConsumer.consume() as per README
    await consumer.consume(onFrame=handle_message)
    
    # Wait for all messages
    await asyncio.sleep(30)
    
    try:
        await consumer.disconnect()
        await kafka.disconnect()
    except Exception:
        pass
    
    print(f"✅ Done! Processed {count[0]} faces")


if __name__ == "__main__":
    asyncio.run(run_consumer(10))
