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


class FaceFrameGrabber(FrameGrabber):
    """Custom FrameGrabber for face images with additional metadata"""
    def __init__(self, face_img, bbox_info, face_id, frame_size):
        # Don't call super().__init__() since we don't use camera
        self.face_img = face_img
        self.bbox_info = bbox_info
        self.face_id = face_id
        self.frame_size = frame_size
        
        h, w = face_img.shape[:2]
        self.width = w
        self.height = h
        self.jpeg_quality = 80
        
    def read_frame(self):
        """Encode face image as JPEG bytes"""
        _, buffer = cv2.imencode('.jpg', self.face_img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()


class ExtendedInputLayerProducer(InputLayerProducer):
    """Extended producer that supports custom metadata fields"""
    
    async def send_frame_with_metadata(self, frame_grabber: FaceFrameGrabber):
        """
        Send frame with extended metadata support.
        Public method that follows architecture pattern but allows custom fields.
        """
        frame_bytes = frame_grabber.read_frame()
        if not frame_bytes:
            return
        
        # Create base metadata using InputLayerMetadataVideo
        metadata = InputLayerMetadataVideo(
            time_stamp=int(time.time()),
            source_id=self.id,
            encoding='jpeg',
            width=frame_grabber.width,
            height=frame_grabber.height
        ).as_dict()
        
        # Add custom fields
        metadata['face_id'] = frame_grabber.face_id
        metadata['bbox'] = json.dumps(frame_grabber.bbox_info)
        metadata['frame_size'] = json.dumps(frame_grabber.frame_size)
        
        # Use protected method for actual sending (same as send_frame does)
        await self._send_message(frame_bytes, metadata)


async def run_producer(num_frames=-1):
    producer = ExtendedInputLayerProducer(
        topic="gaze.frames",
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
        
        # Create custom FrameGrabber with metadata
        face_grabber = FaceFrameGrabber(
            face_img=face_img,
            bbox_info=bbox_info,
            face_id=face_id,
            frame_size=frame_size
        )
        
        # Schedule async send using extended producer method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(producer.send_frame_with_metadata(face_grabber))
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
