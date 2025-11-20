"""
Starter program for demo, uses local camera
- for each frame calls deepFace.analyze_frame and prints results as json
- shows a visualization window if IS_VISUALIZE_ENABLED is TRUE
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import cv2
import deepFaceDetection

IS_VISUALIZE_ENABLED = True  # set False to disable cv2 window


async def main_loop():
    camera = None
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("No NATS stream and no local camera found. Exiting.")
        return
    print("Using local camera for frames.")

    # thread executor for CPU-bound DeepFace calls
    executor = ThreadPoolExecutor(max_workers=6)

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break
            personID = "unknown"

            # run deepFace analysis in a thread so we don't block asyncio loop
            result = await asyncio.get_event_loop().run_in_executor(executor, deepFaceDetection.analyze_frame, frame, personID)

            # print JSON result
            try:
                print(json.dumps(result, default=str))
            except Exception:
                print(result)

            # visualization
            if IS_VISUALIZE_ENABLED:
                vis = frame.copy()
                stable = result.get("stable_emotion")
                dom = result.get("dominant_emotion")
                age = result.get("age")
                gender = result.get("gender")
                text = f"ID:{personID} DOM:{dom} STABLE:{stable} AGE:{age} G:{gender}"
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("DeepFace - Live", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit requested.")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if camera:
            camera.release()
        if IS_VISUALIZE_ENABLED:
            cv2.destroyAllWindows()



if __name__ == '__main__':
    asyncio.run(main_loop())
