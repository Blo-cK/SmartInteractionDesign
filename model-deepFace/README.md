# DeepFace Emotion Recognition
Input:

    Webcam Stream

Outputs:
    
    "age": str(analysis['age']),
    "gender": str(analysis['dominant_gender']),
    "emotion": str(analysis['dominant_emotion']),
    "race": str(analysis['dominant_race']),

Following models could be used:
 

    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",

Following Emotions can be detected
   
    Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral