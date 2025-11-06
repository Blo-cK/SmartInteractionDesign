# AffectNet Emotion Recognition

Generally needed Input:
    
    Short snippet of Webcam Stream

Output:
    
    frame,Neutral,Happiness,Sadness,Surprise,Fear,Disgust,Anger
    000001,0.6259228,0.2763268,0.033525277,0.042410657,0.005775456,0.0058860816,0.010152857
    000002,0.6259228,0.2763268,0.033525277,0.042410657,0.005775456,0.0058860816,0.010152857

    -> acummulation of emotions could be used to predict overall emotion?

- Different types of temporal / backbone model can be used (~ 6 different types)
- The models themselves could also be tried out analyzing live webcam feed without snippets (example check_backbone_models_by_webcam.ipynb) 


Problem:

    Only the first Face gets predicted until it gets out of sight --> maybe wrong face when multiple people are standing in front of it
    --> Maybe cut webcam snippet into faces and predict the active speaker 

