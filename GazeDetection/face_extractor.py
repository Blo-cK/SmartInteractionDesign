import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime
import json
import face_recognition
from PIL import Image


class FaceRecognitionTracker:
    """Robust face recognition system using deep learning models"""
    
    def __init__(self, similarity_threshold=0.6, timeout_seconds=60):
        self.known_faces = {}  # {face_id: {'encodings': [...], 'last_seen': timestamp}}
        self.face_counter = 0
        self.similarity_threshold = similarity_threshold
        self.timeout_seconds = timeout_seconds
        print("🤖 Initialized robust face recognition with deep learning")
        
    def extract_face_encoding(self, face_img):
        """Extract deep learning face encoding using face_recognition library"""
        try:
            if face_img is None or face_img.size == 0:
                return None
            
            # Convert BGR to RGB (face_recognition expects RGB)
            if len(face_img.shape) == 3:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            
            # Get face encodings using the robust deep learning model
            face_encodings = face_recognition.face_encodings(rgb_img)
            
            if len(face_encodings) > 0:
                # Return the first (and usually only) face encoding
                return face_encodings[0]
            else:
                print("⚠️ No face encodings found in image")
                return None
                
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def find_or_assign_face_id(self, face_img):
        """Find existing face ID or assign new one using robust face recognition"""
        current_time = time.time()
        
        # Extract face encoding using deep learning
        face_encoding = self.extract_face_encoding(face_img)
        if face_encoding is None:
            return None
            
        # Clean up old faces
        self._cleanup_old_faces(current_time)
        
        # Compare with known faces using face_recognition's distance function
        best_match_id = None
        best_distance = float('inf')
        face_distances = {}
        
        for face_id, face_data in self.known_faces.items():
            # Calculate distances to all stored encodings for this person
            known_encodings = face_data['encodings']
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(distances) > 0:
                min_distance = min(distances)
                avg_distance = np.mean(distances)
                
                face_distances[face_id] = {
                    'min_distance': min_distance,
                    'avg_distance': avg_distance,
                    'all_distances': distances.tolist()
                }
                
                # Use minimum distance for best match
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_match_id = face_id
        
        # Show detailed distances for debugging
        for face_id, distances in face_distances.items():
            print(f"🔍 {face_id}: min_dist={distances['min_distance']:.3f}, "
                  f"avg_dist={distances['avg_distance']:.3f}")
        
        print(f"🎯 Best match: {best_match_id} "
              f"(distance: {best_distance:.3f}, threshold: "
              f"{self.similarity_threshold})")
        
        # Face recognition uses distance (lower is better), threshold should be low
        # Typical threshold is 0.6 - lower means more similar
        if best_match_id and best_distance < self.similarity_threshold:
            # Update existing face
            self.known_faces[best_match_id]['encodings'].append(face_encoding)
            self.known_faces[best_match_id]['last_seen'] = current_time
            
            # Limit stored encodings per person (keep more for better matching)
            if len(self.known_faces[best_match_id]['encodings']) > 5:
                self.known_faces[best_match_id]['encodings'].pop(0)
                
            print(f"✅ Matched to existing person: {best_match_id}")
            return best_match_id
        else:
            # New face
            new_face_id = f"person_{self.face_counter:03d}"
            self.known_faces[new_face_id] = {
                'encodings': [face_encoding],
                'last_seen': current_time
            }
            self.face_counter += 1
            print(f"✨ New person detected: {new_face_id} "
                  f"(distance to closest: {best_distance:.3f})")
            return new_face_id
    
    def _cleanup_old_faces(self, current_time):
        """Remove faces that haven't been seen for too long"""
        to_remove = []
        for face_id, face_data in self.known_faces.items():
            if current_time - face_data['last_seen'] > self.timeout_seconds:
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.known_faces[face_id]
            print(f"🗑️ Removed old face ID: {face_id}")
    
    def get_stats(self):
        """Get current statistics"""
        total_encodings = sum(len(data['encodings']) for data in self.known_faces.values())
        return {
            'total_known_faces': len(self.known_faces),
            'face_ids': list(self.known_faces.keys()),
            'total_encodings_stored': total_encodings
        }


class WebcamFaceExtractor:
    """Main class for webcam face extraction with rolling frame buffer"""
    
    def __init__(self, frames_folder="frames", faces_folder="faces", 
                 max_frames=100, capture_interval=1.0, on_face_extracted=None):
        self.frames_folder = frames_folder
        self.faces_folder = faces_folder
        self.max_frames = max_frames
        self.capture_interval = capture_interval
        self.on_face_extracted = on_face_extracted
        
        # Note: Directories should be created/cleared by the caller
        # before instantiating this class
        
        # Initialize robust face recognition system
        self.face_tracker = FaceRecognitionTracker()
        
        # Capture control
        self.is_running = False
        self.cap = None
        self.capture_thread = None
        
        # Frame tracking
        self.frame_counter = 0
        self.saved_frames = []  # List of frame filenames for cleanup
        
    def _create_directories(self):
        """Create necessary directories"""
        for folder in [self.frames_folder, self.faces_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 Created directory: {folder}")
    
    def reset_directories(self):
        """Clean existing directories for fresh start"""
        import shutil
        
        for folder in [self.frames_folder, self.faces_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"🗑️ Cleaned directory: {folder}")
        
        self._create_directories()
        print("🔄 Directories reset for fresh start")
    
    def _cleanup_old_frames(self):
        """Remove oldest frames if we exceed max_frames"""
        try:
            # Remove oldest frames while we have more than max_frames
            while len(self.saved_frames) > self.max_frames:
                old_frame = self.saved_frames.pop(0)
                frame_path = os.path.join(self.frames_folder, old_frame)
                
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        print(f"🗑️ Removed old frame: {old_frame}")
                except Exception as e:
                    print(f"Error removing frame {old_frame}: {e}")
        except Exception as e:
            print(f"Error in _cleanup_old_frames: {e}")
    
    def _normalize_frame_format(self, frame):
        """Convert frame to standard 8-bit BGR format for face_recognition compatibility"""
        try:
            if frame is None or frame.size == 0:
                return None
            
            # Check frame dtype and convert if needed
            if frame.dtype != np.uint8:
                print(f"⚠️ Converting frame dtype from {frame.dtype} to uint8")
                if frame.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    frame = cv2.convertScaleAbs(frame, alpha=255.0/65535.0)
                elif frame.dtype == np.float32 or frame.dtype == np.float64:
                    # Convert float to 8-bit
                    frame = cv2.convertScaleAbs(frame, alpha=255.0)
                else:
                    # Fallback: generic conversion
                    frame = frame.astype(np.uint8)
            
            # Check number of channels
            if len(frame.shape) == 2:
                # Grayscale - convert to BGR
                print(f"⚠️ Converting grayscale to BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:
                channels = frame.shape[2]
                if channels == 4:
                    # RGBA or BGRA - convert to BGR
                    print(f"⚠️ Converting RGBA/BGRA ({channels} channels) to BGR")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif channels != 3:
                    # Unexpected number of channels
                    print(f"⚠️ Unexpected {channels} channels, attempting conversion")
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error normalizing frame format: {e}")
            return None
    
    def _extract_and_save_faces(self, frame, frame_filename):
        """Extract faces from frame using robust face detection and save with IDs"""
        try:
            # Normalize frame format for compatibility with different cameras
            frame = self._normalize_frame_format(frame)
            if frame is None:
                print("⚠️ Skipping frame due to format normalization error")
                return
            
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use face_recognition's robust face detection
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if len(face_locations) == 0:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Calculate face dimensions
                face_width = right - left
                face_height = bottom - top
                
                # Skip very small faces (likely false positives)
                if face_width < 80 or face_height < 80:
                    continue
                
                # Add some padding around the face
                padding = max(20, int(min(face_width, face_height) * 0.2))
                y1 = max(0, top - padding)
                x1 = max(0, left - padding)
                y2 = min(frame.shape[0], bottom + padding)
                x2 = min(frame.shape[1], right + padding)
                
                # Extract face region from original BGR frame
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Get face ID through robust recognition
                face_id = self.face_tracker.find_or_assign_face_id(face_img)
                
                if face_id:
                    # Create person-specific folder
                    person_folder = os.path.join(self.faces_folder, face_id)
                    if not os.path.exists(person_folder):
                        os.makedirs(person_folder)
                        print(f"📁 Created folder for {face_id}")
                    
                    # Save face image
                    face_filename = f"{timestamp}_{i:02d}.jpg"
                    face_path = os.path.join(person_folder, face_filename)
                    
                    cv2.imwrite(face_path, face_img)
                    
                    # Save bounding box metadata
                    meta_filename = f"{timestamp}_{i:02d}.json"
                    meta_path = os.path.join(person_folder, meta_filename)
                    
                    frame_height, frame_width = frame.shape[:2]
                    metadata = {
                        "bbox": {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1},
                        "frame_size": {"width": frame_width, "height": frame_height}
                    }
                    
                    import json
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)
                    
                    print(f"💾 Saved face: {face_id}/{face_filename} "
                          f"({face_width}x{face_height})")
                    
                    # Trigger callback immediately after extraction
                    if self.on_face_extracted:
                        self.on_face_extracted(face_path, meta_path, face_id)
                    
        except Exception as e:
            print(f"Error extracting faces: {e}")
    
    def _capture_loop(self):
        """Main capture loop"""
        print(f"🎥 Starting capture loop (interval: {self.capture_interval}s)")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_filename = f"frame_{timestamp}_{self.frame_counter:06d}.jpg"
                    frame_path = os.path.join(self.frames_folder, frame_filename)
                    
                    # Save frame
                    cv2.imwrite(frame_path, frame)
                    self.saved_frames.append(frame_filename)
                    
                    print(f"📸 Captured frame: {frame_filename}")
                    
                    # Extract faces
                    self._extract_and_save_faces(frame, frame_filename)
                    
                    # Delete frame after face extraction
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    except Exception as e:
                        print(f"⚠️ Could not delete frame: {e}")
                    
                    # Cleanup old frames
                    self._cleanup_old_frames()
                    
                    self.frame_counter += 1
                    # If we've reached the configured max frames, stop gracefully
                    # (skip check if max_frames is -1 for infinite capture)
                    if self.max_frames > 0 and self.frame_counter >= self.max_frames:
                        print("⏭️ Reached max frames: "
                              f"{self.frame_counter}/{self.max_frames}."
                              " Stopping capture.")
                        # Perform graceful stop actions from within the
                        # capture thread
                        self._on_max_frames_reached()
                        break
                    
                    # Show current stats
                    if self.frame_counter % 10 == 0:
                        stats = self.face_tracker.get_stats()
                        print(f"📊 Stats - Frames: {len(self.saved_frames)}, "
                              f"Known faces: {stats['total_known_faces']}")
                
                else:
                    print("❌ Failed to capture frame")
                
                # Wait for next capture
                time.sleep(self.capture_interval)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def start_capture(self, camera_index=0):
        """Start webcam capture"""
        if self.is_running:
            print("⚠️ Capture already running")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_index}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📷 Camera initialized - Resolution: {width}x{height}")
        print(f"🎯 Max frames: {self.max_frames}")
        print(f"⏱️ Capture interval: {self.capture_interval}s")
        print(f"📁 Frames folder: {os.path.abspath(self.frames_folder)}")
        print(f"👥 Faces folder: {os.path.abspath(self.faces_folder)}")
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("✅ Face extraction started!")
    
    def stop_capture(self):
        """Stop webcam capture"""
        if not self.is_running:
            print("⚠️ Capture not running")
            return
        
        print("🛑 Stopping capture...")
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread:
            # Avoid joining from the capture thread itself
            if threading.current_thread() is not self.capture_thread:
                self.capture_thread.join(timeout=5)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Final statistics
        stats = self.face_tracker.get_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames captured: {self.frame_counter}")
        print(f"   Current frames stored: {len(self.saved_frames)}")
        print(f"   Unique persons detected: {stats['total_known_faces']}")
        print(f"   Face IDs: {stats['face_ids']}")
        
        # Save session info
        self._save_session_info(stats)
        
        print("✅ Capture stopped")

    def _on_max_frames_reached(self):
        """Handle graceful shutdown when max frames limit is reached.

        This is safe to call from within the capture thread: it will
        set the running flag to False, release the camera, save session
        info and print final stats. We avoid joining the capture thread
        here because this method is called from that thread itself.
        """
        try:
            # Stop the loop
            self.is_running = False

            # Release camera early
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass

            # Final statistics
            stats = self.face_tracker.get_stats()
            print(f"\n📊 Final Statistics (auto-stop):")
            print(f"   Total frames captured: {self.frame_counter}")
            print(f"   Current frames stored: {len(self.saved_frames)}")
            print(f"   Unique persons detected: {stats['total_known_faces']}")
            print(f"   Face IDs: {stats['face_ids']}")

            # Save session info
            self._save_session_info(stats)
            print("✅ Auto-stop complete (max frames reached)")

        except Exception as e:
            print(f"Error in auto-stop handler: {e}")
    
    def _save_session_info(self, stats):
        """Save session information to JSON"""
        try:
            session_info = {
                'session_end': datetime.now().isoformat(),
                'total_frames_captured': self.frame_counter,
                'frames_stored': len(self.saved_frames),
                'max_frames_limit': self.max_frames,
                'capture_interval_seconds': self.capture_interval,
                'face_recognition_stats': stats,
                'folders': {
                    'frames': os.path.abspath(self.frames_folder),
                    'faces': os.path.abspath(self.faces_folder)
                }
            }
            
            info_file = os.path.join(self.faces_folder, 'session_info.json')
            with open(info_file, 'w') as f:
                json.dump(session_info, f, indent=2)
            
            print(f"💾 Session info saved: {info_file}")
            
        except Exception as e:
            print(f"Error saving session info: {e}")
    
    def get_current_stats(self):
        """Get current statistics"""
        stats = self.face_tracker.get_stats()
        return {
            'frames_captured': self.frame_counter,
            'frames_stored': len(self.saved_frames),
            'is_running': self.is_running,
            'face_stats': stats
        }


    def start(frames_folder="frames",
            faces_folder="faces",
            max_frames=100,  # Keep only 100 most recent frames
            capture_interval=2.0  # Capture every 2 seconds
        ):
        """Main function to run the face extractor"""
        print("🚀 Starting Webcam Face Extractor - ROBUST DEEP LEARNING VERSION")
        print("🧠 Using state-of-the-art face recognition with dlib/face_recognition")
        print("=" * 70)
        
        # Clear existing directories first
        import shutil
        
        folders_to_clear = [frames_folder, faces_folder]
        print("🔄 Clearing directories for fresh start...")
        
        for folder in folders_to_clear:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"🗑️ Cleaned directory: {folder}")
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created directory: {folder}")
        
        print("✅ Directories cleared and ready")
        
        # Create face extractor
        extractor = WebcamFaceExtractor(
            frames_folder=frames_folder,
            faces_folder=faces_folder,
            max_frames=max_frames,
            capture_interval=capture_interval
        )
        
        try:
            # Start capture
            extractor.start_capture(camera_index=0)
            
            print("\n🔄 System running...")
            print("📸 Capturing frames every 2 seconds")
            print("👥 Extracting and recognizing faces")
            print("🗂️ Organizing faces by person ID")
            print("🔄 Maintaining rolling buffer of 100 frames")
            print("\nPress Ctrl+C to stop...")
            
            # Keep main thread alive until capture stops or interrupted
            while extractor.is_running:
                time.sleep(10)
                stats = extractor.get_current_stats()
                print(f"📊 Status - Frames: {stats['frames_stored']}/{extractor.max_frames}, "
                      f"Persons: {stats['face_stats']['total_known_faces']}")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            
        finally:
            extractor.stop_capture()
            print("👋 Program terminated")