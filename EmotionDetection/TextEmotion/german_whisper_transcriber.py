import torch
import numpy as np
import whisper

class GermanWhisperTranscriber:
    def __init__(self, model_size="medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio, sample_rate=16000, language="de"):
        # Whisper expects float32 numpy array, sample_rate=16000
        result = self.model.transcribe(audio, language=language, fp16=(self.device=="cuda"))
        return result.get("text", "")
