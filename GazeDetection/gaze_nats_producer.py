"""NATS Producer: Face Extractor → NATS"""
import asyncio
import sys
import os
import cv2
import time
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerProducer
from face_extractor import WebcamFaceExtractor


class FaceFrameGrabber:
    """Adapter to provide face images with metadata"""
    def __init__(self, face_img, bbox_info, face_id):
        self.face_img = face_img
        self.bbox_info = bbox_info
        self.face_id = face_id
        h, w = face_img.shape[:2]
        self.width = w
        self.height = h
        
    def read_frame(self):
        _, buffer = cv2.imencode('.jpg', self.face_img)
        return buffer.tobytes()


async def run_producer(num_frames=10):
    producer = InputLayerProducer(
        topic="gaze.frames",
        source_name="camera1",
        broker="152.53.32.66:4222"
    )
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frames_folder = os.path.join(base_dir, "frames")
    faces_folder = os.path.join(base_dir, "faces")
    
    extractor = WebcamFaceExtractor(
        frames_folder=frames_folder,
        faces_folder=faces_folder,
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
    
    sent = 0
    for person_folder in Path(faces_folder).glob("person_*"):
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
            
            # Send frame with custom metadata
            frame_bytes = cv2.imencode('.jpg', face_img)[1].tobytes()
            h, w = face_img.shape[:2]
            
            # Create base metadata using InputLayerMetadata structure
            from architecture.library.input_layer import InputLayerMetadata
            import time as time_module
            metadata = InputLayerMetadata(
                time_stamp=int(time_module.time()),
                source_id='camera1',
                encoding='jpeg',
                width=w,
                height=h
            ).as_dict()
            
            # Add custom metadata
            metadata['face_id'] = person_folder.name
            metadata['bbox'] = json.dumps(bbox_info)
            
            # Send using _send_message to include custom metadata
            await producer._send_message(frame_bytes, metadata)
            sent += 1
            print(f"✅ {sent} faces")
    
    await producer.disconnect()
    print(f"✅ {sent} faces → NATS")


if __name__ == "__main__":
    asyncio.run(run_producer(10))
