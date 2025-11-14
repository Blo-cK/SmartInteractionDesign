import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn

class GazeDetector:
    def __init__(self, model_path=None):
        """Initialize gaze detector with L2CS model"""
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
        """Convert BGR image to normalized tensor"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        face_tensor = self.transform(face_rgb)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def detect_gaze(self, face_image, face_bbox=None, frame_width=1920, frame_height=1080):
        """Detect head rotation (pitch/yaw) and position in frame"""
        # Get model predictions
        face_tensor = self.preprocess_face(face_image)
        
        with torch.no_grad():
            pitch, yaw = self.model(face_tensor)
            pitch = float(pitch.cpu().numpy()[0])
            yaw = float(yaw.cpu().numpy()[0])
        
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
            "head_position": head_position
        }


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