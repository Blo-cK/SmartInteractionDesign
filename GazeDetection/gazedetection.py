import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not installed. Install with: pip install mediapipe")

class GazeDetector:
    def __init__(self, model_path=None):
        """Initialize gaze detector with MediaPipe Face Landmarker"""
        
        if not MEDIAPIPE_AVAILABLE:
            print("⚠️ MediaPipe not available, gaze detection will return dummy values")
            self.detector = None
            return
        
        # Set default model path - check multiple locations
        if model_path is None:
            # Try local models folder first
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker.task')
            
            # If not found, use HeadGestureRecognition model
            if not os.path.exists(model_path):
                base_dir = os.path.dirname(os.path.dirname(__file__))
                model_path = os.path.join(
                    base_dir, 
                    'HeadGestureRecognition', 
                    'src', 
                    'model_checkpoints', 
                    'face_landmarker_v2_with_blendshapes.task'
                )
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            print("Expected locations:")
            print("  1. GazeDetection/models/face_landmarker.task")
            print("  2. HeadGestureRecognition/src/model_checkpoints/face_landmarker_v2_with_blendshapes.task")
            self.detector = None
            return
        
        # Initialize MediaPipe Face Landmarker
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=5,  # Detect up to 5 faces
            min_face_detection_confidence=0.3,  # Lower threshold for better detection
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.detector = FaceLandmarker.create_from_options(options)
        print(f"✅ MediaPipe Face Landmarker loaded from {model_path}")
    
    def detect_gaze(self, frame, face_bbox=None, frame_width=None, frame_height=None):
        """
            Detect gaze/head pose from frame using MediaPipe Face Landmarker        
        Args:
            frame: Full BGR frame from camera or cropped face image
            face_bbox: bbox
            frame_width: Auto-detected from frame
            frame_height: Auto-detected from frame
            
        Returns:
            dict with pitch, yaw, roll, head_position for each detected face (different values for the different use cases)
        """
        
        if self.detector is None:
            return []
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        if frame_width is None:
            frame_width = w
        if frame_height is None:
            frame_height = h
        
        print(f"🔍 Processing frame: {w}x{h}, dtype={frame_rgb.dtype}")
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            # Actually run the detection!
            result = self.detector.detect(mp_image)
            
            print(f"📊 MediaPipe result: {len(result.facial_transformation_matrixes)} faces, {len(result.face_landmarks)} landmarks")
            
            # Check if we have landmarks but no transformation matrices
            if len(result.face_landmarks) > 0 and len(result.facial_transformation_matrixes) == 0:
                print("⚠️ MediaPipe: Detected landmarks but no transformation matrix")
                print("    This usually means face detection confidence is too low or lighting is poor")
                # Try to compute pose from landmarks using solvePnP
                return self._estimate_pose_from_landmarks(result.face_landmarks, w, h)
            
            if len(result.facial_transformation_matrixes) == 0:
                # No faces detected at all
                print("⚠️ MediaPipe: No faces detected in frame")
                return []
            
            # Process all detected faces
            faces_data = []
            for idx, rotation_matrix in enumerate(result.facial_transformation_matrixes):
                pitch, yaw, roll = self._matrix_to_euler(rotation_matrix)
                
                # Get face landmarks to calculate head position
                if idx < len(result.face_landmarks):
                    landmarks = result.face_landmarks[idx]
                    # Use nose tip (landmark 1) as head center
                    nose = landmarks[1]
                    head_position = {
                        "x": round(nose.x, 4),
                        "y": round(nose.y, 4)
                    }
                else:
                    head_position = None
                
                faces_data.append({
                    "pitch": round(pitch, 2),
                    "yaw": round(yaw, 2),
                    "roll": round(roll, 2),
                    "head_position": head_position,
                    "face_id": f"face_{idx}"
                })
            
            return faces_data
            
        except Exception as e:
            print(f"⚠️ MediaPipe error: {e}")
            return []
    
    def _estimate_pose_from_landmarks(self, face_landmarks_list, img_w, img_h):
        """Fallback: estimate head pose using facial landmarks and solvePnP"""
        faces_data = []
        
        # 3D model points of key facial landmarks (nose, chin, eyes, mouth corners)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (assuming centered principal point)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        for idx, landmarks in enumerate(face_landmarks_list):
            # Extract 2D image points for key landmarks
            # MediaPipe face mesh indices: 1=nose, 152=chin, 33=left eye, 263=right eye, 61=left mouth, 291=right mouth
            key_indices = [1, 152, 33, 263, 61, 291]
            
            image_points = np.array([
                (landmarks[i].x * img_w, landmarks[i].y * img_h)
                for i in key_indices
            ], dtype=np.float64)
            
            # Solve for pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                continue
            
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self._matrix_to_euler(rotation_mat)
            
            # Get head position from nose landmark
            nose = landmarks[1]
            head_position = {
                "x": round(nose.x, 4),
                "y": round(nose.y, 4)
            }
            
            faces_data.append({
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "head_position": head_position,
                "face_id": f"face_{idx}"
            })
        
        return faces_data
    
    def detect_gaze_legacy(self, face_image, face_bbox=None, frame_width=1920, frame_height=1080):
        """Legacy method for backward compatibility with cropped faces"""
        
        # Calculate head position (normalized 0-1)
        head_position = None
        if face_bbox is not None:
            x, y, w, h = face_bbox
            center_x = (x + w / 2) / frame_width
            center_y = (y + h / 2) / frame_height
            head_position = {
                "x": round(center_x, 4),
                "y": round(center_y, 4)
            }
        
        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2),
            "head_position": head_position
        }
    
    def _matrix_to_euler(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees
        
        Args:
            rotation_matrix: Either 3x3 (from solvePnP) or 4x4 (from MediaPipe)
        """
        # Check if it's a 3x3 or 4x4 matrix
        if rotation_matrix.size == 9:
            # 3x3 matrix from cv2.solvePnP/Rodrigues
            rot_matrix = rotation_matrix.reshape(3, 3)
            # Apply coordinate transformation (OpenCV to standard)
            rotation_x_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            rot_matrix = rot_matrix @ rotation_x_pi
        else:
            # 4x4 matrix from MediaPipe
            re = rotation_matrix.reshape(4, 4)
            rotation_x_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            re[1:3, :] = -re[1:3, :]
            rot_matrix = re[:3, :3] @ rotation_x_pi
        
        euler = R_.from_matrix(rot_matrix).as_euler("xyz", degrees=True)
        
        pitch = euler[0]
        yaw = euler[1]
        roll = euler[2]
        
        return pitch, yaw, roll