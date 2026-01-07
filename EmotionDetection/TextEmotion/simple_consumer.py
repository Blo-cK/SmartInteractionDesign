import asyncio
import torch
import wave
import numpy as np
from datetime import datetime
from german_whisper_transcriber import GermanWhisperTranscriber
from german_emotion_classifier import GermanEmotionClassifier
import sys
import os

SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerConsumer
from architecture.library.output_layer import OutputLayerProducer

WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
EMOTION_MODEL_NAME = "facebook/bart-large-mnli"  


LANGUAGE = "de"
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

NATS_TOPIC="audio_microphone"
NATS_BROKER="152.53.32.66:4222"
KAFKA_BROKER="152.53.32.66:9094"

print("Initializing models...")
transcriber = GermanWhisperTranscriber(model_size=WHISPER_MODEL_SIZE, device="cuda" if USE_GPU else "cpu")
emotion_classifier = GermanEmotionClassifier(model_name=EMOTION_MODEL_NAME)
print("Models initialized successfully.")

DEBUG_MODE = True

async def try_nats_queue(consumer,timeout=1.0):
    try:
        await asyncio.wait_for(consumer.connect(),timeout)
        print("Connected to nats stream")
        return True
    except asyncio.TimeoutError:
        return False

async def save_wav(filename: str, audio: np.ndarray):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

async def transcribe_audio(audio):
    print("[Transcription] Processing audio...")
    try:
        text = transcriber.transcribe(audio, sample_rate=SAMPLE_RATE, language=LANGUAGE)
        if text and text.strip():
            print(f"[Transcription]: '{text}'")
            return text
        else:
            print("[Transcription] No speech detected in audio chunk.")
            return None
    except Exception as e:
        print(f"[Transcription] Error: {e}")
        return None


async def main():
    os.makedirs("./temp", exist_ok=True)

    print(f"Device: {device} (GPU: {'Enabled' if USE_GPU else 'Disabled'})")
    print(f"Whisper model: {WHISPER_MODEL_SIZE}")
    print(f"Emotion model: {EMOTION_MODEL_NAME}")
    print(f"Chunk duration: {CHUNK_DURATION}s")
    print(f"NATS topic: {NATS_TOPIC}")
    print(f"Kafka broker: {KAFKA_BROKER}")
    print("="*60)
    print("🎧 Listening for audio... Press Ctrl+C to stop.\n")
    async def handle_message(msg):
        audio = msg.astype(np.float32)
        if len(audio) < SAMPLE_RATE * CHUNK_DURATION:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_name = f"chunk_{timestamp}.wav"
        wav_path = f"./temp/{wav_name}"
        save_wav(wav_path, audio)
        text = await transcribe_audio(audio)
        if text == None:
            return
        
        # Detect emotion using transformer model
        emotion_result = emotion_classifier.predict(text)
        
        # Enhanced logging with all emotion scores
        print(f"[Emotion] Detected: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
        if DEBUG_MODE:
            print(f"[Emotion] All scores: {emotion_result['all']}")
        
        # Send to Kafka
        await kafka.sendData(msg.headers, emotion_result, 'model_emotion_text')
    
    consumer = InputLayerConsumer(
        topic=NATS_TOPIC,
        broker=NATS_BROKER
    )
    
    kafka = OutputLayerProducer(
        broker=KAFKA_BROKER
    )
    
    use_nats = await try_nats_queue(consumer,3.0)
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
        print("Try running main.py")


if __name__ == "__main__":
    asyncio.run(main())