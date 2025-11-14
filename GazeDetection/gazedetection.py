import torch
import cv2
import numpy as np
from torchvision import transforms
import math

import torch.nn as nn

class GazeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the gaze detector with L2CS model
        
        Args:
            model_path: Path to the pretrained L2CS model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path):
        """Load the L2CS model"""
        # Simplified L2CS model architecture
        model = L2CSModel()
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for gaze detection
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        face_tensor = self.transform(face_rgb)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def detect_gaze(self, face_image, face_bbox=None, frame_width=1920, frame_height=1080):
        """
        Detect gaze direction and return comprehensive data
        
        Args:
            face_image: Face image as numpy array
            face_bbox: Optional tuple (x, y, w, h) of face location in original frame
            frame_width: Width of the original webcam frame
            frame_height: Height of the original webcam frame
            
        Returns:
            dict: Dictionary with head rotation, position, and gaze data
        """
        # Preprocess the face image
        face_tensor = self.preprocess_face(face_image)
        
        with torch.no_grad():
            # Get gaze predictions (pitch and yaw angles in degrees)
            pitch, yaw = self.model(face_tensor)
            
            # Convert to numpy
            pitch = float(pitch.cpu().numpy()[0])
            yaw = float(yaw.cpu().numpy()[0])
        
        # Calculate relative head position if bbox provided
        head_position = None
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Calculate center of face relative to frame (0-1 normalized)
            center_x = (x + w / 2) / frame_width
            center_y = (y + h / 2) / frame_height
            
            head_position = {
                "x": round(center_x, 4),
                "y": round(center_y, 4)
            }
        
        # Return comprehensive data
        return {
            "pitch": round(pitch, 2),  # Head rotation up/down (degrees)
            "yaw": round(yaw, 2),      # Head rotation left/right (degrees)
            "head_position": head_position,
            "frame_width": frame_width,
            "frame_height": frame_height
        }
    
    def _angles_to_coordinates(self, pitch, yaw, width, height):
        """
        Convert pitch and yaw angles to screen coordinates
        
        Args:
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            width: Screen width
            height: Screen height
            
        Returns:
            tuple: (x, y) screen coordinates
        """
        # Convert degrees to radians
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        # Assume a viewing distance and convert to screen coordinates
        # These constants may need adjustment based on your setup
        x = width / 2 + (math.tan(yaw_rad) * width / 2)
        y = height / 2 - (math.tan(pitch_rad) * height / 2)
        
        # Clamp to screen boundaries
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        
        return int(x), int(y)


class L2CSModel(nn.Module):
    """Simplified L2CS model implementation"""
    
    def __init__(self):
        super(L2CSModel, self).__init__()
        
        # Use ResNet18 as backbone
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=True)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Gaze estimation heads
        self.pitch_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.yaw_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Predict pitch and yaw
        pitch = self.pitch_head(features)
        yaw = self.yaw_head(features)
        
        return pitch, yaw