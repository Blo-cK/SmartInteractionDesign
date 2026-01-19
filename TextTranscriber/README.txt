# Usage of TextTranscriber
python=3.11

## audio_producer
- edit USE_VISUALIZATION = True 
    - True: create push to talk window. Record during button press and send after release (for testing)
    - False: continously produce adio chunks (10s windows) #audio transcriber sometimes has issues transcribing "empty" chunks to '' text
Output:
    Records audio and sends it to nats


## transcription_producer
- edit WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
- edit LANGUAGE = "de" # option: en, de (probably some more possible)

Input:
    nats audio chunk 
Output:
    transcribed text to nats