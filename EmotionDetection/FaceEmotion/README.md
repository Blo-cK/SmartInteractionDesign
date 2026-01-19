# DeepFace Emotion Recognition
python=3.11
Input:
    Nats FaceExtractor

Outputs:
    TO Kafka
    "age": age of person
    "gender": gender of person
    "stable_emotion": emotion predicted through max polling buffer
    "dominant_emotion": emotion predicted without buffer
    "emotion_changed": if the emotion did change
    "previous_emotion": the emotion predicted before the current one


Following models could be used for finetuning:
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",

Following Emotions can be possibly detected   
    Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral