import onnxruntime as ort
import numpy as np
from transformers import Wav2Vec2Processor
import torch


class VoiceEmotionModel:
    def __init__(self, model_path: str):
        """
        model_path: directory where the HuggingFace model files (.onnx, .processor, etc.) are stored
        """
        print("Loading processor...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)

        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            f"{model_path}/model.onnx",
            providers=["CPUExecutionProvider"]
        )

        # Read emotion labels
        self.labels = ["anger","boredom","disgust","fear","happiness","sadness","neutral"]

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Model loaded.")

    def predict(self, audio_buffer: np.ndarray):
        """
        Takes raw audio samples (float32, 16kHz) and returns the predicted emotion.
        """

        # Preprocess with processor
        inputs = self.processor(
            audio_buffer,
            sampling_rate=16000,
            return_tensors="np",
            padding=True
        )

        # Run ONNX model
        logits = self.session.run(
            [self.output_name],
            {self.input_name: inputs["input_values"]}
        )[0]

        # Get probabilities
        probs = torch.softmax(torch.tensor(logits[0]), dim=-1)
        idx = torch.argmax(probs).item()

        return self.labels[idx], probs[idx].item()
