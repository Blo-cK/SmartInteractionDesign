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
python .\architecture\nats_producer.py

python .\architecture\nats_consumer.py
```
For Audio sending and receiving 
```
python .\architecture\nats_producer.py

python .\architecture\nats_consumer.py
```

# Input & Output Layer - Updated Architecture Overview (2025)

The system is built around two core messaging components:

* Input Layer: handles real-time ingestion of camera frames and audio streams
* Output Layer: distributes processed results (e.g. ML detections) into Kafka

The Input Layer uses **NATS**.  
The Output Layer uses **Kafka**.

Both services are already deployed on the server, so you do not need to install them manually.  
Docker-compose files are provided for optional custom deployments.

The new examples use the **updated consumer**:

**NEW:** InputLayerConsumerThread  
**OLD (deprecated):** InputLayerConsumer  

The new threaded consumer is required for real-time audio playback and OpenCV video display.

# Input Layer

The Input Layer is responsible for:

* Capturing raw sensor data (video frames or audio)
* Attaching metadata automatically (timestamps, encoding, sizes, fps…)
* Publishing data to NATS
* Providing threaded low-latency consumers for real-time work

A camera or microphone publishes to a topic such as:

cams.cam1  
input.audio-stream

## Components

### InputLayerProducer  
Unified producer for **both video and audio**.  
Captures frames or audio chunks and publishes them to NATS.

### InputLayerMetadata  
Generated automatically with source ID, timestamp, dimensions, etc.

### InputLayerConsumerThread (NEW)
The new recommended consumer.  
Uses a background thread for receiving frames or audio with:

* low latency  
* smooth OpenCV `imshow`  
* smooth audio playback  
* async-compatible  
* non-blocking  

### InputLayerConsumer (OLD — deprecated)
Older async-only version.  
Kept for backwards compatibility but **no longer recommended**.

# Audio Streaming

## Audio Producer Example

Captures microphone audio and streams it to NATS:


# New vs Old Consumer

| Feature | InputLayerConsumer (old) | InputLayerConsumerThread (new) |
|--------|---------------------------|--------------------------------|
| Threaded | ❌ | ✅ |
| Works with OpenCV imshow | ❌ Often blocks | ✅ Smooth |
| Works with audio playback | ❌ No | ✅ Yes |
| Latency | High | Low |
| Recommended | ❌ Deprecated | ✅ Use this |

# Output Layer

The Output Layer handles:

* Collecting results from processing services (AI, detection, analysis, etc.)
* Mapping metadata from the Input Layer
* Publishing results to Kafka  
* Enabling dashboards or downstream services to consume data

Example metadata:

