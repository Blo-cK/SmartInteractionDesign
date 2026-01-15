"""NATS Producer: NATS Audio Chunks -> Text transcriber -> NATS"""
import torch
import numpy as np
import whisper
import re

class GermanWhisperTranscriber:
    def __init__(self, model_size="medium", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Loaded Whisper model '{model_size}' on device: {self.device}")

    def transcribe(self, audio, sample_rate=16000, language="de"):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio) + 1e-8)
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=(self.device == "cuda"),
            verbose=False,
            temperature=0.0
        )        
        text = result.get("text", "").strip()
        # Post-process for spaCy
        if text:
            text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
        text = text.replace('„', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text.strip()
    
    def transcribe_with_segments(self, audio, sample_rate=16000, language="de"):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio) + 1e-8)
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=(self.device == "cuda"),
            verbose=False
        )
        full_text = self._clean_text(result.get("text", ""))
        segments = result.get("segments", [])
        cleaned_segments = []
        for seg in segments:
            cleaned_segments.append({
                "text": self._clean_text(seg.get("text", "")),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })
        
        return {
            "text": full_text,
            "segments": cleaned_segments,
            "language": result.get("language", language)
        }
