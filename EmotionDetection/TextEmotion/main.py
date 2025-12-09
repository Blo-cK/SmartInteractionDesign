import torch
import torch.nn.functional as F
import wave
import numpy as np
from datetime import datetime
import sounddevice as sd
import os
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


def audio_callback(indata, frames, time, status):
    global model
    if status:
        print("Audio status:", status)
        # Ensure ./temp exists
    os.makedirs("./temp", exist_ok=True)
    
    # Flatten to mono
    audio = indata[:, 0].astype(np.float32)   
    
    # transcribe
    
    # analyze emotion
    

if __name__ == "__main__":
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