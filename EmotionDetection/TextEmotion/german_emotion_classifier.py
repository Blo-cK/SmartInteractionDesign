from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class GermanEmotionClassifier:
    def __init__(self, model_name="cardiffnlp/twitter-xlm-roberta-base-emotion-multilingual", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.labels = [
            "anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"
        ]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        result = {label: float(prob) for label, prob in zip(self.labels, probs)}
        top_label = self.labels[np.argmax(probs)]
        return {"emotion": top_label, "confidence": float(np.max(probs)), "all": result}
