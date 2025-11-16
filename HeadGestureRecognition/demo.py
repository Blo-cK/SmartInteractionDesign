import argparse
import cv2
from SmartInteractionDesign.HeadGestureRecognition.headgesturerecognition import HeadGestureDetector  # importiere die Klasse aus deinem Modul


class HeadGestureDemo:
    """Main class to test HeadGestureDetector."""

    def __init__(self, args):
        self.detector = HeadGestureDetector(
            face_detector=args.face_detector,
            draw_bbox=args.draw_bbox,
            draw_landmarks=args.draw_landmarks,
            draw_head_gesture=args.draw_head_gesture,
            device=args.device
        )
        self.cam_id = args.cam_id
        self.test_image_path = args.test_image

    def run_webcam(self):
        """Run live webcam demo."""
        print("[INFO] Starting webcam...")
        self.detector.run_webcam(self.cam_id)

    def run_test_image(self):
        """Run head gesture detection on a single image."""
        if not self.test_image_path:
            print("[ERROR] No test image provided.")
            return

        frame = cv2.imread(self.test_image_path)
        if frame is None:
            print(f"[ERROR] Could not load image {self.test_image_path}")
            return

        annotated_frame, gesture_output = self.detector.process_frame(frame)
        print("[INFO] Gesture output:", gesture_output)

        cv2.imshow("Head Gesture Test", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HeadGestureDetector")
    parser.add_argument("--face_detector", type=str, default="CV2", choices=["CV2", "YUNET"])
    parser.add_argument("--draw_bbox", type=bool, default=True)
    parser.add_argument("--draw_landmarks", type=bool, default=True)
    parser.add_argument("--draw_head_gesture", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cam_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--test_image", type=str, default=None, help="Path to test image")
    args = parser.parse_args()

    demo = HeadGestureDemo(args)

    if args.test_image:
        demo.run_test_image()
    else:
        demo.run_webcam()
