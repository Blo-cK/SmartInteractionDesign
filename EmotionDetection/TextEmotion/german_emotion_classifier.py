from transformers import pipeline
import numpy as np

class GermanEmotionClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        print(f"Loading emotion model: {model_name}")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self._cuda_available() else -1
        )
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
            'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral'
        ]
        print(f"Supporting {len(self.emotion_labels)} emotion labels")
    
    @staticmethod
    def _cuda_available():
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def predict(self, text):
        if not text or not text.strip():
            return {
                "emotion": "neutral",
                "confidence": 1.0,
                "all": {label: (1.0 if label == "neutral" else 0.0) for label in self.emotion_labels}
            }
        result = self.classifier(text[:512], self.emotion_labels)
        emotion_scores = dict(zip(result['labels'], result['scores']))
        top_emotion = result['labels'][0]
        top_confidence = float(result['scores'][0])
        
        return {
            "emotion": top_emotion,
            "text-input": text,
            "confidence": round(top_confidence, 4),
            "all": {k: round(v, 4) for k, v in emotion_scores.items()}
        }
