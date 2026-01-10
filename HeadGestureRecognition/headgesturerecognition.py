import cv2
import time
import torch

from src.utils_demo import (
    FaceDetectorCV2,
    FaceDetectorYUNET,
    FaceTracker,
    MediapipePredictor,
    HGPredictor,
    TrackHandler,
    Visualizer,
)


class HeadGestureRecognition:
    """Head gesture detector with clean API similar to GazeDetector."""

    def __init__(
        self,
        face_detector="CV2",
        draw_bbox=True,
        draw_landmarks=True,
        draw_head_gesture=True,
        device="cpu"
    ):
        """Initialize all components for head gesture detection."""
        self.device = torch.device(device)

        # Face detector
        if face_detector == "YUNET":
            self.face_detector = FaceDetectorYUNET()
        elif face_detector == "CV2":
            self.face_detector = FaceDetectorCV2()
        else:
            raise ValueError("Invalid face detector type")

        # Tracking & prediction pipeline
        self.face_tracker = FaceTracker()
        self.face_predictor = MediapipePredictor()
        self.hg_predictor = HGPredictor(device)
        self.track_handler = TrackHandler(self.face_tracker)

        # Drawing helper
        self.visualizer = Visualizer(
            draw_bbox=draw_bbox,
            draw_landmarks=draw_landmarks,
            draw_head_gesture=draw_head_gesture,
        )

    # --------------------------------------------------------------
    # Main processing of a single frame
    # --------------------------------------------------------------
    def process_frame(self, frame):
        """
        Processes a frame and returns:
        - Annotated Frame
        - List of gesture predictions per tracked face
        """

        start = time.time()
        frame_time = int(round(time.time() * 1000))

        # -------------------------
        # 1) Detect faces
        # -------------------------
        detection = self.face_detector.process_image(frame)

        # -------------------------
        # 2) Update tracker
        # -------------------------
        self.face_tracker.update(detection, frame_time)
        track_ids = self.face_tracker.get_tracks()

        # -------------------------
        # 3) Mediapipe face landmarks
        # -------------------------
        for track_id in track_ids:
            track = self.face_tracker.tracks_store[track_id][-1]  # last state
            face_prediction = self.face_predictor.process_face(frame, track)
            track.add_prediction(face_prediction)

        # -------------------------
        # 4) Gesture prediction
        # -------------------------
        gesture_output = self.hg_predictor.process(self.face_tracker, track_ids)
        self.track_handler.add_track_prediction(gesture_output)

        # -------------------------
        # 5) Visualize output
        # -------------------------
        # annotated_frame = self.visualizer.process(frame, self.face_tracker, gesture_output)

        # -------------------------
        # 6) FPS calculation
        # -------------------------
        # fps = 1.0 / (time.time() - start)
        # cv2.putText(
        #     annotated_frame, f"FPS {int(fps)}",
        #     (1650, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX, 2,
        #     (0, 255, 0), 2, cv2.LINE_AA
        # )

        return gesture_output

    # --------------------------------------------------------------
    # Webcam demo function (equivalent to original main loop)
    # --------------------------------------------------------------
    def run_webcam(self, cam_id=0):
        """Starts the webcam and displays live detection results."""
        vid = cv2.VideoCapture(cam_id)

        while True:
            ret, frame = vid.read()
            if not ret:
                break

            gesture_output = self.process_frame(frame)
            # print("[INFO] Gesture output:", gesture_output.keys())
            # print("[INFO] Gesture output:", gesture_output.values())
            print("[INFO] Gesture output:", gesture_output.items())
            # cv2.imshow("Head Gesture Demo", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        vid.release()
        cv2.destroyAllWindows()
