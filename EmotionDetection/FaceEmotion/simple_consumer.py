"""Consumer: NATS (FaceExtractor) → DeepFace Detection → Kafka"""
import asyncio
import cv2
import numpy as np
import deepFaceDetection
import json
import base64
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer

NATS_TOPIC = "camera1.faceextractor"
NATS_BROKER = "152.53.32.66:4222"
KAFKA_BROKER = "152.53.32.66:9094"

IS_VISUALIZE_ENABLED = True  # set False to disable cv2 window

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_face_image(msg_data):
    """Extract and decode face image from NATS message (JSON format)"""
    try:
        json_data = json.loads(msg_data.decode('utf-8'))
        face_b64 = json_data.get('face_image', None)
        
        if face_b64 is None:
            logger.warning("No face_image in message")
            return None, None
        
        # Decode base64 image
        face_bytes = base64.b64decode(face_b64)
        face_array = np.frombuffer(face_bytes, dtype=np.uint8)
        face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
        
        # Extract metadata
        metadata = {
            'face_id': json_data.get('face_id', 'unknown'),
            'bbox': json_data.get('bbox', {}),
            'frame_size': json_data.get('frame_size', {})
        }
        
        return face_img, metadata
    except Exception as e:
        logger.error(f"Error decoding face image: {e}")
        return None, None

async def main():
    logger.info(f"Listening to NATS topic: {NATS_TOPIC}")
    logger.info("Waiting for face frames... Press Ctrl+C to stop.\n")

    face_consumer = InputLayerConsumer(
        topic=NATS_TOPIC,
        broker=NATS_BROKER
    )
    kafka_producer = OutputLayerProducer(
        broker=KAFKA_BROKER
    )
    
    async def handle_face_frame(input_result: InputResultWrapper):
        try:
            msg = input_result.msg
            face_img, metadata = decode_face_image(msg.data)
            
            if face_img is None:
                logger.warning("Invalid or empty face frame received")
                return
            
            face_id = metadata.get('face_id', 'unknown')
            logger.info(f"Received face frame: {face_id}")
            
            result = deepFaceDetection.analyze_frame(face_img, face_id)
            logger.info(f"[DeepFace] Analysis complete for {face_id}")
            if IS_VISUALIZE_ENABLED and result:
                vis = face_img.copy()
                stable = result.get("stable_emotion", "N/A")
                dom = result.get("dominant_emotion", "N/A")
                age = result.get("age", "N/A")
                gender = result.get("gender", "N/A")
                text = f"ID:{face_id} CURRENT:{dom} STABLE:{stable} AGE:{age} G:{gender}"
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("DeepFace - Stream", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested by user")
                    raise KeyboardInterrupt()
    
            await kafka_producer.sendData(input_result, result, 'model_deepFace')
            logger.info(f"DeepFace result sent to Kafka for {face_id}")
            
        except Exception as e:
            logger.error(f"Error processing face frame: {e}", exc_info=True)
    
    try:
        await face_consumer.connect()
        logger.info("Connected to NATS")
        await face_consumer.consume(onFrame=handle_face_frame)
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Connection failed: {e}", exc_info=True)
    finally:
        await face_consumer.disconnect()
        await kafka_producer.producer.stop()
        if IS_VISUALIZE_ENABLED:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())