
import torch
import wave
import numpy as np
from datetime import datetime
import sounddevice as sd
import os
from german_whisper_transcriber import GermanWhisperTranscriber
from german_emotion_classifier import GermanEmotionClassifier

SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


# Model configuration
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
EMOTION_MODEL_NAME = "facebook/bart-large-mnli"  # Zero-shot emotion classification
# Supports 28 emotions: admiration, amusement, anger, annoyance, approval, caring,
# confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment,
# excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, 
# realization, relief, remorse, sadness, surprise, neutral

LANGUAGE = "de"
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")

transcriber = GermanWhisperTranscriber(model_size=WHISPER_MODEL_SIZE, device="cuda" if USE_GPU else "cpu")
emotion_classifier = GermanEmotionClassifier(model_name=EMOTION_MODEL_NAME)


def transcribe_audio(audio):
    """
    Transcribe audio chunk and detect emotion via transformer model.
    Prints the transcription, emotion, and all emotion scores.
    """
    print("[Transcription] Processing audio...")
    try:
        text = transcriber.transcribe(audio, sample_rate=SAMPLE_RATE, language=LANGUAGE)
        if text and text.strip():
            print(f"[Transcription]: '{text}'")
            emotion_result = emotion_classifier.predict(text)
            print(f"[Emotion] Detected: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
            # Optionally show all emotion scores
            print(f"[Emotion] All scores: {emotion_result['all']}")
        else:
            print("[Transcription] No speech detected.")
    except Exception as e:
        print(f"[Transcription] Error: {e}")

def save_wav(filename: str, audio: np.ndarray):
    """audio chunk into WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    os.makedirs("./temp", exist_ok=True)
    # Flatten to mono
    audio = indata[:, 0].astype(np.float32)
    if len(audio) < SAMPLE_RATE * CHUNK_DURATION:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_name = f"chunk_{timestamp}.wav"
    wav_path = f"./temp/{wav_name}"
    save_wav(wav_path, audio)
    transcribe_audio(audio)


if __name__ == '__main__':
    print("="*60)
    print("🎙️  Local Audio → Transcription → Emotion Detection")
    print("="*60)
    print(f"Device: {device} (GPU: {'Enabled' if USE_GPU else 'Disabled'})")
    print(f"Whisper model: {WHISPER_MODEL_SIZE}")
    print(f"Emotion model: {EMOTION_MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Chunk duration: {CHUNK_DURATION}s")
    print("="*60)
    print("Listening... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(
            channels=1,
            callback=audio_callback,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype="float32"
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped by user.")
