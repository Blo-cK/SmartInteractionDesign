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
    """face recognition system using deep learning models"""
    
    def __init__(self, similarity_threshold=0.6, timeout_seconds=60):
        self.known_faces = {}  # {face_id: {'encodings': [...], 'last_seen': timestamp}}
        self.face_counter = 0
        self.similarity_threshold = similarity_threshold
        self.timeout_seconds = timeout_seconds
        
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
                print("No face encodings found in image")
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
            print(f"{face_id}: min_dist={distances['min_distance']:.3f}, "
                  f"avg_dist={distances['avg_distance']:.3f}")
        
        print(f"Best match: {best_match_id} "
              f"(distance: {best_distance:.3f}, threshold: "
              f"{self.similarity_threshold})")
        
        if best_match_id and best_distance < self.similarity_threshold:
            # Update existing face
            self.known_faces[best_match_id]['encodings'].append(face_encoding)
            self.known_faces[best_match_id]['last_seen'] = current_time
            
            # Limit stored encodings per person (keep more for better matching)
            if len(self.known_faces[best_match_id]['encodings']) > 5:
                self.known_faces[best_match_id]['encodings'].pop(0)
                
            print(f"Matched to existing person: {best_match_id}")
            return best_match_id
        else:
            # New face
            new_face_id = f"person_{self.face_counter:03d}"
            self.known_faces[new_face_id] = {
                'encodings': [face_encoding],
                'last_seen': current_time
            }
            self.face_counter += 1
            print(f"New person detected: {new_face_id} "
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
            print(f"Removed old face ID: {face_id}")
    

class WebcamFaceExtractor:
    """Main class for webcam face extraction with rolling frame buffer"""
    
    def __init__(self, frames_folder="frames", faces_folder="faces", 
                 max_frames=100, capture_interval=0.5, on_face_extracted=None):
        self.frames_folder = frames_folder
        self.faces_folder = faces_folder
        self.max_frames = max_frames
        self.capture_interval = capture_interval
        self.on_face_extracted = on_face_extracted
        
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
                print(f"Created directory: {folder}")
    
    def reset_directories(self):
        """Clean existing directories for fresh start"""
        import shutil
        
        for folder in [self.frames_folder, self.faces_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Cleaned directory: {folder}")
        
        self._create_directories()
        
    def _cleanup_old_frames(self):
        """Remove oldest frames if we exceed max_frames"""
        try:
            # Remove oldest frames
            while len(self.saved_frames) > self.max_frames:
                old_frame = self.saved_frames.pop(0)
                frame_path = os.path.join(self.frames_folder, old_frame)
                
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        print(f"Removed old frame: {old_frame}")
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
                print(f"Converting frame dtype from {frame.dtype} to uint8")
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
                print(f"Converting grayscale to BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:
                channels = frame.shape[2]
                if channels == 4:
                    # RGBA or BGRA - convert to BGR
                    print(f"Converting RGBA/BGRA ({channels} channels) to BGR")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif channels != 3:
                    # Unexpected number of channels
                    print(f"Unexpected {channels} channels, attempting conversion")
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            return frame
            
        except Exception as e:
            print(f"Error normalizing frame format: {e}")
            return None
    
    def _extract_and_save_faces(self, frame, frame_filename):
        """Extract faces from frame using face detection and save with IDs"""
        try:
            # Normalize frame format for compatibility with different cameras
            frame = self._normalize_frame_format(frame)
            if frame is None:
                print("Skipping frame due to format normalization error")
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
                        print(f"Created folder for {face_id}")
                    
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
                    
                    print(f"Saved face: {face_id}/{face_filename} "
                          f"({face_width}x{face_height})")
                    
                    # Trigger callback immediately after extraction
                    if self.on_face_extracted:
                        self.on_face_extracted(face_path, meta_path, face_id)
                    
        except Exception as e:
            print(f"Error extracting faces: {e}")
    
    def _capture_loop(self):
        """Main capture loop"""
        print(f"Starting capture loop (interval: {self.capture_interval}s)")
        
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
                    
                    print(f"Captured frame: {frame_filename}")
                    
                    # Extract faces
                    self._extract_and_save_faces(frame, frame_filename)
                    
                    # Delete frame after face extraction
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                            # Remove from tracking list since we deleted it immediately
                            if frame_filename in self.saved_frames:
                                self.saved_frames.remove(frame_filename)
                    except Exception as e:
                        print(f"Could not delete frame: {e}")
                    
                    # Cleanup old frames (only needed if frames weren't deleted immediately)
                    # self._cleanup_old_frames()
                    
                    self.frame_counter += 1
                    # if non infinite mode
                    if self.max_frames > 0 and self.frame_counter >= self.max_frames:
                        print("Reached max frames: "
                              f"{self.frame_counter}/{self.max_frames}."
                              " Stopping capture.")
                        self._on_max_frames_reached()
                        break
                else:
                    print("Failed to capture frame")
                
                # Wait for next capture
                time.sleep(self.capture_interval)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def start_capture(self, camera_index=0):
        """Start webcam capture"""
        if self.is_running:
            print("Capture already running")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"Could not open camera {camera_index}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized - Resolution: {width}x{height}")
        print(f"Max frames: {self.max_frames}")
        print(f"Capture interval: {self.capture_interval}s")
        print(f"Frames folder: {os.path.abspath(self.frames_folder)}")
        print(f"Faces folder: {os.path.abspath(self.faces_folder)}")
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("Face extraction started!")
    
    def stop_capture(self):
        """Stop webcam capture"""
        if not self.is_running:
            print("Capture not running")
            return
        self.is_running = False
        
        # Wait for thread to finish (fail safe)
        if self.capture_thread:
            # Avoid joining from the capture thread itself
            if threading.current_thread() is not self.capture_thread:
                self.capture_thread.join(timeout=5)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        
    def _on_max_frames_reached(self):
        try:
            # Stop the loop
            self.is_running = False

            # Release camera early
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error in auto-stop handler: {e}")