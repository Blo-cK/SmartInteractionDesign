"""Consumer: NATS → Gaze Detection → Kafka"""
import asyncio
import cv2
import numpy as np
import deepFaceDetection
import json
from concurrent.futures import ThreadPoolExecutor
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from architecture.library.output_layer import OutputLayerProducer

NUM_FRAMES=10
IS_VISUALIZE_ENABLED = True  # set False to disable cv2 window
NATS_TOPIC="gaze.frames"
NATS_BROKER="152.53.32.66:4222"
KAFKA_BROKER="152.53.32.66:9094"

# Trys to connect to Nats SRC. If timeout is reached -> use local camera
async def try_nats_queue(consumer,timeout=1.0):
    """Attempt to connect to subscribe to NATS_Topic"""
    try:
        await asyncio.wait_for(consumer.connect(),timeout)
        print("Connected to nats stream")
        return True
    except asyncio.TimeoutError:
        return False

async def main():
    async def handle_message(msg):
        face_data = msg.data
        meta = msg.headers or {}

        # Decode face image
        face_array = np.frombuffer(face_data, dtype=np.uint8)
        face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
        
        if face_img is None:
            return
        
        # Parse bbox from metadata
        bbox_info = json.loads(meta.get('bbox', '{}'))
        bbox = None
        if bbox_info:
            bbox = (bbox_info['x'], bbox_info['y'],
                   bbox_info['w'], bbox_info['h'])
        
        # Get frame dimensions
        frame_size = bbox_info.get('frame_size', {})
        w = frame_size.get('width', 1920)
        h = frame_size.get('height', 1080)
        
        result = deepFaceDetection.analyze_frame(face_img,"UNKNOWN")
        print(result)
        await kafka.sendData(msg.headers, result, 'model_deepFace')
    
    consumer = InputLayerConsumer(
        topic=NATS_TOPIC,
        broker=NATS_BROKER
    )
    
    kafka = OutputLayerProducer(
        broker=KAFKA_BROKER
    )
    
    use_nats = await try_nats_queue(consumer,3.0)
    camera = None
    if use_nats:
        try:
            while True:
                await consumer.consume(onFrame=handle_message)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            await consumer.disconnect()
            await kafka.disconnect()

    else:
        executor = ThreadPoolExecutor(max_workers=1)
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("No NATS stream and no local camera found. Exiting.")
            return
        print("Using local camera for frames.")
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Failed to read frame from camera. Exiting.")
                    break
                personID = "unknown"
                result = await asyncio.get_event_loop().run_in_executor(executor, deepFaceDetection.analyze_frame, frame, personID)
                meta = {"time_stamp":result.get('timestamp'), "source_id":"model_deepFace"}
                await kafka.sendData(meta, result, 'model_deepFace')
                try:
                    print(json.dumps(result, default=str))
                except Exception:
                    print(result)

                # visualization
                if IS_VISUALIZE_ENABLED and result:
                    vis = frame.copy()
                    stable = result.get("stable_emotion")
                    dom = result.get("dominant_emotion")
                    age = result.get("age")
                    gender = result.get("gender")
                    text = f"ID:{personID} CURRENT:{dom} STABLE:{stable} AGE:{age} G:{gender}"
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("DeepFace - Live", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested.")
                        break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            kafka.disconnect()
            if camera:
                camera.release()
            if IS_VISUALIZE_ENABLED:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())