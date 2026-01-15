import time
import cv2
from face_extractor import WebcamFaceExtractor
from headgesturerecognition import HeadGestureRecognition
import json
from pathlib import Path
import shutil

def main():
    """Run face extraction and gaze detection pipeline"""
    # Clear old data
    for folder in ["faces", "frames"]:
        if Path(folder).exists():
            shutil.rmtree(folder)
    print("🗑️ Cleared old data")
    
    # Start face extraction (auto-stops at max_frames)
    extractor = WebcamFaceExtractor(
        frames_folder="frames",
        faces_folder="faces",
        max_frames=10,
        capture_interval=0.1
    )
    
    try:
        extractor.start_capture(camera_index=0)
        print("📸 Capturing frames (Ctrl+C to stop)...")
        
        # Wait for auto-stop
        while extractor.is_running:
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        
    finally:
        extractor.stop_capture()
        
    # Process faces for headgesture detection
    if extractor.frame_counter < extractor.max_frames:
        return
    
    print(f"\n✅ Captured {extractor.frame_counter} frames")
    print("🔍 Starting headgesture recognition...")
    
    headgesture_recognition = HeadGestureRecognition()
    headgesture_results = []
    
    # Process all face images
    for face_file in Path("faces").glob("**/*.jpg"):
        try:
            # Load face image
            face_image = cv2.imread(str(face_file))
            if face_image is None:
                continue
            
            # Load metadata (bbox and frame size)
            meta_file = face_file.with_suffix('.json')
            bbox_info = None
            frame_width, frame_height = 1920, 1080
            
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    bbox_info = metadata.get('bbox')
                    frame_size = metadata.get('frame_size', {})
                    frame_width = frame_size.get('width', 1920)
                    frame_height = frame_size.get('height', 1080)
            
            # Prepare bbox tuple
            face_bbox = None
            if bbox_info:
                face_bbox = (bbox_info['x'], bbox_info['y'],
                           bbox_info['w'], bbox_info['h'])
            
            # Detect headgesture
            headgesture_data = headgesture_recognition.process_frame(face_image)
            
            # Extract metadata from filename
            parts = face_file.stem.split('_')
            timestamp = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
            
            # Store result
            headgesture_results.append({
                "person_id": face_file.parent.name,
                "timestamp": timestamp,
                "filename": face_file.name,
                "head_rotation": {
                    "pitch": headgesture_data["pitch"],
                    "yaw": headgesture_data["yaw"]
                },
                "head_position": headgesture_data["head_position"]
            })
            
            print(f"📍 {face_file.name}: pitch={headgesture_data['pitch']}°, yaw={headgesture_data['yaw']}°")
            
        except Exception as e:
            print(f"❌ Error: {face_file.name}: {e}")
    
    # Save results
    with open("headgesture_results.json", 'w') as f:
        json.dump(headgesture_results, f, indent=2)
    
    print(f"💾 Saved {len(headgesture_results)} results to headgesture_results.json")


if __name__ == "__main__":
    main()