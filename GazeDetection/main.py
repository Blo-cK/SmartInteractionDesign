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
                    
                    # Get gaze coordinates
                    gaze_coords = gaze_detector.detect_gaze(face_image)
                    
                    if gaze_coords:
                        gaze_results.append({
                            "person_id": person_id,
                            "timestamp": timestamp,
                            "gaze_x": gaze_coords[0],
                            "gaze_y": gaze_coords[1],
                            "filename": face_file.name
                        })
                        print(f"📍 Processed {face_file.name}: gaze at ({gaze_coords[0]:.2f}, {gaze_coords[1]:.2f})")
                        
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