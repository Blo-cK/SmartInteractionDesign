import time
import cv2
from face_extractor import WebcamFaceExtractor
from gazedetection import GazeDetector
import json
from pathlib import Path
import shutil

def main():
    # Clear faces and frames folders
    faces_folder = Path("faces")
    frames_folder = Path("frames")

    if faces_folder.exists():
        shutil.rmtree(faces_folder)
        print("🗑️ Cleared faces folder")

    if frames_folder.exists():
        shutil.rmtree(frames_folder)
        print("🗑️ Cleared frames folder")
        
    """Run the face extractor with auto-stop functionality"""
    print("🚀 Starting Face Extractor from main.py")
    
    # Create extractor with custom settings
    extractor = WebcamFaceExtractor(
        frames_folder="frames",
        faces_folder="faces", 
        max_frames=10,
        capture_interval=0.1
    )
    
    try:
        # Start capture
        extractor.start_capture(camera_index=0)
        
        print("📸 Capturing frames - will auto-stop at 10 frames")
        print("Press Ctrl+C to stop early...")
        
        # Wait until auto-stop or manual interrupt
        while extractor.is_running:
            time.sleep(2)
            stats = extractor.get_current_stats()
            print(f"📊 Progress: {stats['frames_captured']}/{extractor.max_frames} frames")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        
    finally:
        extractor.stop_capture()
        
    # Check if auto-stop worked
    if extractor.frame_counter >= extractor.max_frames:
        print(f"\n✅ SUCCESS: Auto-stop worked! Captured {extractor.frame_counter} frames")
        # Process faces for gaze detection
        
        print("🔍 Starting gaze detection on captured faces...")
        
        gaze_detector = GazeDetector()
        faces_folder = Path("faces")
        gaze_results = []
        
        if faces_folder.exists():
            # Process all face images in person subdirectories
            for face_file in faces_folder.glob("**/*.jpg"):
                try:
                    # Extract person ID from parent folder name
                    person_id = face_file.parent.name
                    # Extract timestamp from filename
                    filename_parts = face_file.stem.split('_')
                    timestamp = "_".join(filename_parts[:2]) if len(filename_parts) >= 2 else "unknown"
                    
                    # Load the face image
                    face_image = cv2.imread(str(face_file))
                    if face_image is None:
                        print(f"⚠️  Could not load image: {face_file}")
                        continue
                    
                    # Load bounding box metadata
                    meta_file = face_file.with_suffix('.json')
                    bbox_info = None
                    frame_width = 1920
                    frame_height = 1080
                    
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                            bbox_info = metadata.get('bbox')
                            frame_size = metadata.get('frame_size', {})
                            frame_width = frame_size.get('width', 1920)
                            frame_height = frame_size.get('height', 1080)
                    
                    # Get gaze data with bbox for position calculation
                    face_bbox = None
                    if bbox_info:
                        face_bbox = (bbox_info['x'], bbox_info['y'], 
                                    bbox_info['w'], bbox_info['h'])
                    
                    gaze_data = gaze_detector.detect_gaze(
                        face_image,
                        face_bbox=face_bbox,
                        frame_width=frame_width,
                        frame_height=frame_height
                    )
                    
                    if gaze_data:
                        result = {
                            "person_id": person_id,
                            "timestamp": timestamp,
                            "filename": face_file.name,
                            "head_rotation": {
                                "pitch": gaze_data["pitch"],
                                "yaw": gaze_data["yaw"]
                            },
                            "head_position": gaze_data["head_position"]
                        }
                        gaze_results.append(result)
                        pos = gaze_data['head_position']
                        if pos:
                            print(f"📍 {face_file.name}: pitch={gaze_data['pitch']}°, yaw={gaze_data['yaw']}°, pos=({pos['center_x']:.2f}, {pos['center_y']:.2f})")
                        else:
                            print(f"📍 {face_file.name}: pitch={gaze_data['pitch']}°, yaw={gaze_data['yaw']}°")
                        
                except Exception as e:
                    print(f"❌ Error processing {face_file.name}: {e}")
            
            # Save results to JSON
            output_file = "gaze_results.json"
            with open(output_file, 'w') as f:
                json.dump(gaze_results, f, indent=2)
            
            print(f"💾 Saved {len(gaze_results)} gaze detections to {output_file}")
        else:
            print("❌ Faces folder not found")


if __name__ == "__main__":
    main()