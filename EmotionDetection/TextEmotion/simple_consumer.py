import asyncio
import torch
import json
from german_emotion_classifier import GermanEmotionClassifier
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer, OutputLayerMetadata

EMOTION_MODEL_NAME = "facebook/bart-large-mnli"

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

NATS_TOPIC = "microphone1.text_transcriber" 
NATS_BROKER = "152.53.32.66:4222"
KAFKA_BROKER = "152.53.32.66:9094"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing emotion classification model...")
emotion_classifier = GermanEmotionClassifier(model_name=EMOTION_MODEL_NAME)
DEBUG_MODE = True

async def extract_text_from_message(msg_data):
    """Extract text from NATS message"""
    try:
        json_data = json.loads(msg_data.decode('utf-8'))
        return json_data.get('text', None)
    except Exception as e:
        logger.error(f"Error parsing message: {e}")
        return None


async def main():
    logger.info(f"Device: {device} (GPU: {'Enabled' if USE_GPU else 'Disabled'})")
    logger.info(f"Emotion model: {EMOTION_MODEL_NAME}")
    logger.info(f"Listening to NATS topic: {NATS_TOPIC}")
    logger.info("Waiting for transcribed text... Press Ctrl+C to stop.\n")
    text_consumer = InputLayerConsumer(
        topic=NATS_TOPIC,
        broker=NATS_BROKER
    )
    kafka_producer = OutputLayerProducer(
        broker=KAFKA_BROKER
    )
    async def handle_transcribed_text(input_result: InputResultWrapper):
        try:
            msg = input_result.msg
            text = await extract_text_from_message(msg.data)
            
            if text is None or text.strip() == "":
                logger.warning("Empty or invalid text received")
                return
            
            logger.info(f"Received text: '{text}'")
            emotion_result = emotion_classifier.predict(text)
            logger.info(f"[Emotion] Detected: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
            if DEBUG_MODE:
                logger.debug(f"[Emotion] All scores: {emotion_result['all']}")
            await kafka_producer.sendData(input_result, emotion_result, 'model_emotion_text')
            logger.info(f"Emotion result sent to Kafka")
            
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)

    try:
        await text_consumer.connect()
        await text_consumer.consume(onFrame=handle_transcribed_text)
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Connection failed: {e}", exc_info=True)
    finally:
        await text_consumer.disconnect()
        await kafka_producer.producer.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())