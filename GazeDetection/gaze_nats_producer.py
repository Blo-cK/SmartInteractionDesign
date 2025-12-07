"""NATS Producer: Face Extractor → NATS"""
import asyncio
import sys
import os
import cv2
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import (
    InputLayerProducer, InputLayerMetadataVideo
)
from architecture.library.frame_grabber import FrameGrabber
from face_extractor import WebcamFaceExtractor


async def send_face_with_data(producer, face_img, bbox_info, face_id, frame_size):
    """Send face image with metadata in data block (JSON format)"""
    
    # Encode face image to JPEG
    _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    face_bytes = buffer.tobytes()
    
    # Create data package with image and metadata
    import base64
    data_package = {
        'face_image': base64.b64encode(face_bytes).decode('utf-8'),
        'face_id': face_id,
        'bbox': bbox_info,
        'frame_size': frame_size
    }
    
    # Convert to JSON bytes
    json_bytes = json.dumps(data_package).encode('utf-8')
    
    # Send using standard InputLayerProducer (architecture handles metadata)
    h, w = face_img.shape[:2]
    metadata = InputLayerMetadataVideo(
        time_stamp=int(time.time()),
        source_id=producer.id,
        encoding='json',
        width=w,
        height=h
    ).as_dict()
    
    await producer._send_message(json_bytes, metadata)


async def run_producer(num_frames=-1):
    producer = InputLayerProducer(
        topic="input.faceextractor.frames",
        source_name="camera1",
        broker="152.53.32.66:4222"
    )
    
    await producer.connect()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frames_folder = os.path.join(base_dir, "frames")
    faces_folder = os.path.join(base_dir, "faces")
    
    sent = [0]  # Use list for closure
    
    # Callback to send face immediately after extraction
    def on_face_extracted(face_path, meta_path, face_id):
        # Read face and metadata
        face_img = cv2.imread(face_path)
        if face_img is None:
            return
        
        with open(meta_path, 'r') as f:
            file_metadata = json.load(f)
        
        bbox_info = file_metadata.get('bbox', {})
        frame_size = file_metadata.get('frame_size', {})
        
        # Schedule async send with data in JSON format
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_face_with_data(producer, face_img, bbox_info, face_id, frame_size))
        loop.close()
        
        sent[0] += 1
        print(f"📤 Sent face {sent[0]}: {face_id}")
        
        # Delete files after sending
        try:
            if os.path.exists(face_path):
                os.remove(face_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
        except Exception as e:
            print(f"⚠️ Could not delete files: {e}")
    
    extractor = WebcamFaceExtractor(
        frames_folder=frames_folder,
        faces_folder=faces_folder,
        max_frames=num_frames,
        capture_interval=0.01,
        on_face_extracted=on_face_extracted
    )
    
    extractor.reset_directories()
    
    if num_frames == -1:
        print("🎥 Capturing continuously (infinite mode)...")
    else:
        print(f"🎥 Capturing {num_frames} frames (send immediately after extraction)...")
    extractor.start_capture(camera_index=0)
    
    while extractor.is_running:
        await asyncio.sleep(0.5)
    
    await asyncio.sleep(0.5)  # Allow final sends
    await producer.disconnect()
    print(f"✅ {sent[0]} faces → NATS (real-time)")


if __name__ == "__main__":
    asyncio.run(run_producer(-1))  # -1 = infinite mode
