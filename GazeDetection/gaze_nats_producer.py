"""NATS Producer: Face Extractor → NATS"""
import asyncio
import sys
import os
import cv2
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerProducer
from face_extractor import WebcamFaceExtractor


async def run_producer(num_frames=10):
    producer = InputLayerProducer(
        topic="gaze.frames",
        source_name="camera1",
        broker="152.53.32.66:4222"
    )
    
    extractor = WebcamFaceExtractor(
        frames_folder="frames",
        faces_folder="faces",
        max_frames=num_frames,
        capture_interval=0.5
    )
    
    await producer.connect()
    extractor.reset_directories()
    
    print(f"🎥 Capturing {num_frames} frames...")
    extractor.start_capture(camera_index=0)
    
    while extractor.is_running:
        await asyncio.sleep(0.5)
    
    print("📤 Sending faces to NATS...")
    
    import json
    from pathlib import Path
    
    sent = 0
    for person_folder in Path("faces").glob("person_*"):
        face_files = sorted(person_folder.glob("*.jpg"))
        for face_file in face_files:
            face_img = cv2.imread(str(face_file))
            if face_img is None:
                continue
            
            # Load metadata
            meta_file = face_file.with_suffix('.json')
            bbox_info = {}
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    bbox_info = json.load(f).get('bbox', {})
            
            _, buffer = cv2.imencode('.jpg', face_img)
            metadata = {
                'time_stamp': str(int(time.time())),
                'source_id': 'camera1',
                'face_id': person_folder.name,
                'bbox': json.dumps(bbox_info),
                'encoding': 'jpeg'
            }
            
            await producer._send_message(buffer.tobytes(), metadata)
            sent += 1
            print(f"✅ {sent} faces")
    
    await producer.disconnect()
    print(f"✅ {sent} faces → NATS")


if __name__ == "__main__":
    asyncio.run(run_producer(10))