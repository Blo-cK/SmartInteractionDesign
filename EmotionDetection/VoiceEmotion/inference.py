import torch
import torch.nn.functional as F
import wave
import numpy as np
from datetime import datetime
import sounddevice as sd
import os
from transformers import AutoConfig, Wav2Vec2Processor
from Wav2Vec2ForSpeechClassification import Wav2Vec2ForSpeechClassification
#https://github.com/padmalcom/wav2vec2-emotion-detection-ger
MY_MODEL = "padmalcom/wav2vec2-large-emotion-detection-german"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(MY_MODEL)
processor = Wav2Vec2Processor.from_pretrained(MY_MODEL)
sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(MY_MODEL).to(device)

# def speech_file_to_array_fn(path, sampling_rate):
# 	speech_array, _sampling_rate = torchaudio.load(path)
# 	resampler = torchaudio.transforms.Resample(_sampling_rate)
# 	speech = resampler(speech_array).squeeze().numpy()
# 	return speech

def save_wav(filename: str, audio: np.ndarray):
    """Save recorded audio chunk into a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())



def predict(test, sampling_rate):
    # Normalize audio input
    test = test / np.max(np.abs(test))  # Normalize audio
    features = processor(test, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs


def audio_callback(indata, frames, time, status):
    global model
    if status:
        print("Audio status:", status)
        # Ensure ./temp exists
    os.makedirs("./temp", exist_ok=True)
    
    # Flatten to mono
    audio = indata[:, 0].astype(np.float32)
    if len(audio) < SAMPLE_RATE * CHUNK_DURATION:  # Ensure audio is of expected length
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_name = f"chunk_{timestamp}.wav"
    wav_path = f"./temp/{wav_name}"
    save_wav(wav_path, audio)
    
    res = predict(audio, SAMPLE_RATE)
    if(res):
        temp = max(res, key=lambda x: x['Score'])
        print("Expected:", temp)
        with open("./temp/output.txt", "a", encoding="utf-8") as f:
            f.write(f"{wav_name} | {timestamp} | {temp} | {res} \n")

    # print(res)




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


# res = predict("test2.wav", SAMPLE_RATE)
# max = max(res, key=lambda x: x['Score'])
# print("Expected anger:", max)