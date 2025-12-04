import cv2
import asyncio

class FrameGrabber:
    def __init__(self, device=0, width=640, height=480, jpeg_quality=80):
        self.device = device
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality

        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self) -> bytes:
        """Capture a single frame and return JPEG bytes"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ret:
            return None

        return buffer.tobytes()

    def release(self):
        self.cap.release()
        
        