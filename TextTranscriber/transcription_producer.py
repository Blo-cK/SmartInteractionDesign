"""NATS Consumer: Audio Input → Transcribe → NATS Producer"""
import asyncio
import sys
import os
import torch
import time
import json
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import (
    InputLayerProducer, InputLayerConsumer, BaseInputMetadata, InputResultWrapper
)
from german_whisper_transcriber import GermanWhisperTranscriber

# Configuration
SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
LANGUAGE = "de"
USE_GPU = torch.cuda.is_available()
NATS_BROKER = "152.53.32.66:4222"
SERVICE = "text_transcriber"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transcriber = GermanWhisperTranscriber(
    model_size=WHISPER_MODEL_SIZE, 
    device="cuda" if USE_GPU else "cpu"
)


async def send_transcription(producer, text, source_metadata):
    """Send transcribed text to NATS"""
    if text is None or text.strip() == "":
        return
    
    data_package = {
        'text': text
    }
    
    # Convert to JSON bytes
    json_bytes = json.dumps(data_package).encode('utf-8')
    
    metadata = BaseInputMetadata(
        time_stamp=str(time.time()),
        source_id=source_metadata.get('source_id', 'microphone1'),
        service_id='text_transcriber',
        encoding='json',
    ).as_dict()
    
    await producer._send_message(json_bytes, metadata)
    logger.info(f"📤 Transcribed & Sent: {text}")


def decode_audio_chunk(audio_bytes, sample_rate=16000):
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        return audio_float
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        return None


async def transcribe_audio_chunk(audio_bytes, metadata_dict):
    try:
        audio_array = decode_audio_chunk(audio_bytes)
        if audio_array is None or len(audio_array) == 0:
            return None
        
        # Transcribe
        text = transcriber.transcribe(
            audio_array, 
            sample_rate=SAMPLE_RATE, 
            language=LANGUAGE
        )
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None


async def run_consumer_producer():
    audio_consumer = InputLayerConsumer(
        topic="microphone1.audio",
        broker=NATS_BROKER
    )
    text_producer = InputLayerProducer(
        source_name="microphone1",
        service=SERVICE,
        broker=NATS_BROKER
    )
    
    await audio_consumer.connect()
    await text_producer.connect()
    
    logger.info("Connected to NATS. Listening for audio chunks...")
    
    async def on_audio_received(input_result: InputResultWrapper):
        try:
            msg = input_result.msg
            audio_bytes = msg.data
            metadata_dict = dict(msg.headers) if msg.headers else {}
            
            logger.info(f"Received audio chunk ({len(audio_bytes)} bytes)")
            text = await transcribe_audio_chunk(audio_bytes, metadata_dict)
            if text and text.strip() != "":
                await send_transcription(text_producer, text, metadata_dict)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    await audio_consumer.consume(onFrame=on_audio_received)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await audio_consumer.disconnect()
        await text_producer.disconnect()

if __name__ == "__main__":
    asyncio.run(run_consumer_producer())
