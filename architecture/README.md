# Setup
**Create a venv for Python:**
```
python -m venv venv
```

**Activate the venv**
```
venv\Scripts\activate
```

**Install the packages:**
```
pip install -r requirements.txt
```
**You can run example scripts such as:**

For Video sending and receiving 
```
python .\architecture\nats_video_producer.py

python .\architecture\nats_video_consumer.py
```
For Audio sending and receiving 
```
python .\architecture\nats_audio_producer.py

python .\architecture\nats_audio_consumer.py
```

# Input & Output Layer / Processing Layer - Updated Architecture Overview

The system is built around two core messaging components:

* Input Layer: handles real-time ingestion of camera frames and audio streams
* Processing Layer: distributes processed results (e.g. ML detections) into Kafka
* Output Layer: Collects all results from Kafak and provides a single source of truth


The Input Layer uses **NATS**.  
The Output Layer uses **Kafka**.

Both services are already deployed on the server, so you do not need to install them manually.  
Docker-compose files are provided for optional custom deployments.

The new examples use the **updated consumer**:

**NEW:** InputLayerConsumerThread  
**OLD (deprecated):** InputLayerConsumer  

The new threaded consumer is required for real-time audio playback and OpenCV video display.

## New vs Old Consumer

| Feature | InputLayerConsumer (old) | InputLayerConsumerThread (new) |
|--------|---------------------------|--------------------------------|
| Threaded | ❌ | ✅ |
| Works with OpenCV imshow | ❌ Often blocks | ✅ Smooth |
| Works with audio playback | ❌ No | ✅ Yes |
| Latency | High | Low |
| Recommended | ❌ Deprecated | ✅ Use this |

# Input Layer

The Input Layer is responsible for:

* Capturing raw sensor data (video frames or audio)
* Attaching metadata automatically (timestamps, encoding, sizes, fps…)
* Publishing data to NATS
* Providing threaded low-latency consumers for real-time work

A camera or microphone publishes to a topic such as:

cams.cam1  
audio.stream1

## Components

### InputLayerProducer  
Unified producer for **both video and audio**.  
Captures frames or audio chunks and publishes them to NATS.

### InputLayerMetadata  
Generated automatically with source ID, timestamp, dimensions, etc.

### InputLayerConsumerThread (NEW)
The new recommended consumer for **both video and audio**..  
Uses a background thread for receiving frames or audio with:

* low latency  
* smooth OpenCV `imshow`  
* smooth audio playback  
* async-compatible  
* non-blocking  

### InputLayerConsumer (OLD — deprecated)
Older async-only version.  
Kept for backwards compatibility but **no longer recommended**.

# Video Streaming

The working example is located here: `.\architecture\`
For Video look in the `nats_video_producer.py` and `nats_video_consumer.py`

The following code is needed to send video
```
    topic = "cams.cam1"
    myid = str(uuid.uuid4())

    producer = InputLayerProducer(broker=broker,topic=topic,source_name=myid )
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)

    await producer.send_frame(grabber,100)
```
***The full explenation is in the working example***


# Audio Streaming

The working example is located here: `.\architecture\`
For Video look in the `nats_audio_producer.py` and `nats_audio_consumer.py`


The following code is needed to send audio
```
    topic = "audio.stream1"
    myid = str(uuid.uuid4())

    producer = InputLayerProducer(broker=broker, topic=topic,source_name=myid)
    grabber = AudioGrabber(sample_rate=16000, channels=1, chunk_ms=100)

    await producer.send_frame(grabber,100)
```
***The full explenation is in the working example***

* Windows Users: This will ask you to allow Audio Devices
    
* Linux Users: Make sure your Audio Port and Drivers are up to date or this will fail

## Audio Producer Example

Captures microphone audio and streams it to NATS:



# Output Layer

The Output Layer handles:

* Collecting results from processing services (AI, detection, analysis, etc.)
* Mapping metadata from the Input Layer
* Publishing results to Kafka  
* Enabling dashboards or downstream services to consume data

Example metadata:

