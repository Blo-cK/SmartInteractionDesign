#TextEmotion
Predicts emotion on transcribed text. Currently tries to guess
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
python=3.11
Input:
    Nats 
Output:
    To Kafka
    "emotion": dominant predicted emotion 
    "text": transcribed text
    "confidence": the confidence for predicted emotion