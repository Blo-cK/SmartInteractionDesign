import sounddevice as sd
import numpy as np
import torch
import wave
import asyncio
from datetime import datetime
import os
from voiceEmotion import VoiceEmotionModel

SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#using https://huggingface.co/steveway/wav2vec2-large-emotion-detection-german_onnx


def save_wav(filename: str, audio: np.ndarray):
    """Save recorded audio chunk into a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        

def audio_callback(indata, frames, time, status):
    global model

    if status:
        print("Audio status:", status)
        # Ensure ./temp exists
    os.makedirs("./temp", exist_ok=True)
    
    # Flatten to mono
    audio = indata[:, 0].astype(np.float32)    
    # Predict emotion
    emotion, confidence = model.predict(audio)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_name = f"chunk_{timestamp}.wav"
    wav_path = f"./temp/{wav_name}"
    
    save_wav(wav_path, audio)
    with open("./temp/output.txt", "a", encoding="utf-8") as f:
            f.write(f"{wav_name} | {timestamp} | {emotion} | {confidence:.2f}\n")
    print(f"Emotion: {emotion} | Confidence: {confidence:.2f}")


if __name__ == "__main__":
    MODEL_PATH = "./wav2vec_model"  # directory containing .onnx, .processor, .tokens

    print("Initializing emotion model...")
    model = VoiceEmotionModel(MODEL_PATH)

    print("Starting microphone stream...")
    with sd.InputStream(
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype="float32"
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
