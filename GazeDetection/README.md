# Usgae of GazeDetection

There are 2 Modes/Services: FaceExtractor or "Gaze"-Detection (Head position and Head rotation) 

FaceExtractor: Crop faces and send them to NATS => start face_extractor_producer.py
A simple example where it shows how to use this service is in simple_consumer_faceextractor.py (it reads the images from NATS and then sends them 1to1 to kafka (demo purpose))

GazeDetection: Detect Head and send data to Kafka => Start frame_producer.py (sends full frames to NATS). Then start simple_consumer_gaze to start Gaze Detection (uses frames sent to NATS) and sends json-info to Kafka.