from abc import ABC
from dataclasses import dataclass, asdict
import asyncio
import queue
from queue import Queue
import threading
import time
import nats
import numpy as np
from typing import Callable, Literal, Optional
import cv2
import threading
import logging
import sounddevice as sd

from enum import Enum

from .frame_grabber import FrameGrabber
from .audio_grabber import AudioGrabber

class SampleFormat(str, Enum):
    PCM16 = "pcm16"
    FLOAT32 = "float32"
    OPUS = "opus"
@dataclass
class BaseInputMetadata(ABC):
    time_stamp:str
    source_id:str
    encoding:str
    def as_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

@dataclass
class InputLayerMetadataVideo(BaseInputMetadata):
    width:int
    height:int
    
@dataclass
class InputLayerMetadataSound(BaseInputMetadata):
    sample_rate: int
    channels: int
    sample_format: SampleFormat
    chunk_ms: int
    def as_dict(self):
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result



class InputLayerProducer:
    def __init__(self, topic:str, source_name:str, broker:str = "152.53.32.66:4222"):
        self.broker = broker
        self.topic = self.build_topic_name(topic)
        self._connected= False
        self.producer = None
        self.id= source_name
    
    def build_topic_name(self, topic: str) -> str:
        if not topic.startswith("input."):
            return f"input.{topic}"
        return topic
    
    async def connect(self):
        """
        connects to the NATS Backend 
        """
        if self._connected:
            return
        try:
            self.producer = await nats.connect(f"nats://{self.broker}")
            print('NATS Producer connected')
            print("Consumer connected: ",self.producer.is_connected )
            self._connected= self.producer.is_connected
            
        except Exception as e:
            print("Error happened: ",e)
                
    async def _send_message(self, data, metadata:dict):
        """
        sends generic messages to NATS  
        """
        if not self._connected:
            await self.connect()    
        try:
            await self.producer.publish(self.topic,data, headers=metadata)
            print('Message sent successfully')
        except Exception as e:
            print(f"Error sending message: {e}")
    
    #TODO: Implement sending Audio Chunks
    async def send_audio_chunk(self, audio_grabber:AudioGrabber, sample_rate=16000, channels=1):
        """
        sends audio chunk to NATS using the _send_messages 
        """
        if not self._connected:
            await self.connect()
        sample_rate_str = str(getattr(audio_grabber, "sample_rate", sample_rate) or sample_rate)
        channels_str = str(getattr(audio_grabber, "channels", channels) or channels)
        chunk_ms_str = str(getattr(audio_grabber, "chunk_ms", 100))  # default 100ms if missing
        
        audio_bytes = audio_grabber.read_chunk()
        metadata = InputLayerMetadataSound(
            time_stamp=str(time.time()),
            source_id=str(self.id),
            encoding= "int16",
            sample_rate= sample_rate_str,
            channels= channels_str,
            sample_format= SampleFormat.PCM16,
            chunk_ms= chunk_ms_str,
        ).as_dict()

        await self._send_message(audio_bytes, metadata=metadata)
        await asyncio.sleep(1.0/audio_grabber.chunk_ms)
    
    async def send_frame(self, frame_grabber:FrameGrabber, fps=30):
        """
        Capture a frame from FrameGrabber and send to NATS
        """
        frame_bytes = frame_grabber.read_frame()
        if frame_bytes:
            metadata = InputLayerMetadataVideo(
                time_stamp=int(time.time()),
                source_id=self.id,
                encoding="jpeg",
                width=frame_grabber.width,
                height=frame_grabber.height
            ).as_dict()
            
            await self._send_message(frame_bytes, metadata=metadata)
            await asyncio.sleep(1.0/fps)
            
    async def disconnect(self):
        """
        disconnects from NATS  
        """
        if self._connected and self.producer:
            try:
                await self.producer.drain()
                await self.producer.close()
            except Exception as e:
                print("Error while disconnecting", e)
        else:
            print("Error cannot disconnect no connection was found ")
        
        if self.producer.is_closed:
            self._connected= False



class InputResultWrapper():
    """
    Wraps the incomming DATA to InputResult Type    
    """
    def __init__(self, msg):
        self.msg = msg



class InputLayerConsumer:
    "This is the old Consumer Class this wont get updated for audio"
    def __init__(self, topic:str, broker:str= "152.53.32.66:4222"):
        self.broker= broker
        self.topic= self.build_topic_name(topic)
        self._connected= False
        self.consumer= None
        self.subscription = None
    
    def build_topic_name(self, topic: str) -> str:
        if not topic.startswith("input."):
            return f"input.{topic}"
        return topic
    
    async def connect(self):
        if self._connected:
            return
        try:
            self.consumer = await nats.connect(f"nats://{self.broker}")
            self._connected= self.consumer.is_connected
            print(f"Connected to NATS topic '{self.topic}'")
        except Exception as e:
            print("Error happened: ",e)
    
    async def consume(self, onFrame: Callable):
        if not self._connected and self.consumer:
            await self.connect()   
            
        async def message_handler(msg):
            try:
                if onFrame:
                    await onFrame(InputResultWrapper(msg)) # Callback every Gruppe can write their own Callback fucntion so we have decoupled the functionality
            except Exception as e:
                logging.exception("Error while consuming")    
                
        
        self.subscription = await self.consumer.subscribe(self.topic, cb=message_handler)
    
    async def consume_video(self):
        #Consume frames from NATS and display video in OpenCV window.
        if not self._connected or not self.consumer:
            await self.connect()

        async def show_frame(msg):
            frame_bytes: bytes = msg.data
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("NATS Video Stream", frame)
                cv2.waitKey(1)
                
        await self.consume(onFrame=show_frame)
    
    async def disconnect(self):
        if self.consumer and self._connected:
            await self.subscription.unsubscribe()
            await self.consumer.close()
            cv2.destroyAllWindows()
            print(f"[NATS] Disconnected")
        else:
            print("Error cannot disconnect no connection was found ")
        if self.consumer.is_closed:
            self._connected= False
            
     



 
class InputLayerConsumerThread:
    """
    Async NATS consumer that hands off frames to lightweight background threads.
    - A display thread shows the newest frame (minimal latency)
    - An optional user callback thread processes each newest frame
    - The same behaviour is applied to the audio threads
    """

    def __init__(self, topic:str, broker:str= "152.53.32.66:4222"):
        self.broker = broker
        self.topic = self.build_topic_name(topic)
        self._connected = False
        self.consumer = None
        self.subscription = None
        
        
        #Frame Section
        self.latest_frame = None
        self.latest_msg = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Optional user callback
        self.user_callback: Optional[Callable[[np.ndarray], None]] = None
        self.callback_thread: Optional[threading.Thread] = None
        
        #Audio Section
        self.shared_aduio_queue = Queue()
        self.callback_queue = Queue()
        self.audio_player = None
        self.audio_thread = None
        
    def build_topic_name(self, topic: str) -> str:
        "builds the Topic prefix 'Input.' for the Monitor to recognize the messages "
        if not topic.startswith("input."):
            return f"input.{topic}"
        return topic
    
    async def connect(self):
        """
        Connects the Consumer to NATS
        """
        if self._connected:
            return
        try:
            self.consumer = await nats.connect(f"nats://{self.broker}")
            self._connected = self.consumer.is_connected
            print(f"[Consumer] Connected to NATS topic '{self.topic}'")
        except Exception as e:
            print("[Consumer] Error connecting:", e)

    def on_message(self, callback: Callable):
        """
        Register a user-defined function that will receive the newest frame.
        The function is executed in its own background thread.
        """
        self.user_callback = callback

    async def consume_video(self, play_video = False):
        """
        Consumes video and sets the Dispaly and callback
        """
        if not self._connected or not self.consumer:
            await self.connect()

        async def message_handler(msg):
            frame_bytes = msg.data
            
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_msg = msg

        self.subscription = await self.consumer.subscribe(self.topic, cb=message_handler)

        
        # Start display thread
        if play_video:
            display_thread = threading.Thread(target=self._display_loop, daemon=True)
            display_thread.start()

        # Start callback thread if provided
        if self.user_callback:
            self.callback_thread = threading.Thread(target=self._callback_loop, daemon=True)
            self.callback_thread.start()

        print("[Consumer] Started zero-delay display & callback threads.")
        # Keep coroutine alive
        while self.running:
            await asyncio.sleep(0.001)

    
    async def consume_audio(self, play_audio:bool = False):
        """
        Consumes the Audio Chunks and sets the Callback
        """
        if not self._connected or not self.consumer:
            await self.connect()

        async def audio_message_handler(msg):
            with self.frame_lock:
                self.shared_aduio_queue.put_nowait(msg)  # store the latest audio message
                self.callback_queue.put_nowait(msg)
                self.latest_msg = msg

        # Subscribe to the audio topic
        self.subscription = await self.consumer.subscribe(self.topic, cb=audio_message_handler)
        print(f"[Consumer] Subscribed to audio topic '{self.topic}'")

        if self.user_callback:
            self.callback_thread = threading.Thread(target=self._callback_loop_audio_queue, args=(), daemon=True)
            self.callback_thread.start()

        if play_audio and self.audio_player is None:
            self.audio_player = AudioPlayer()
            self.audio_player.start(queue=self.shared_aduio_queue)
        # Keep coroutine alive while running
        while self.running:
            await asyncio.sleep(0.001)
    
    def _display_loop(self):
        """
        This is the Display Loop that is offloaded into a seperate Thread
        TODO: This could be seperated into its own VideoPlayer Class like i did with the AudioPlayer
        """
        print("[Display] Thread started.")
        while self.running:
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is not None:
                cv2.imshow("NATS Zero-Delay Stream", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("[Display] 'q' pressed. Stopping display.")
                self.running = False
                break

        cv2.destroyAllWindows()

    def _callback_loop(self):
        """Continuously calls the user callback with the latest frame. This is also offloaded into a seperate Thread"""
        print("[Callback] Thread started.")
        while self.running:
            frame = None
            msg = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    msg = self.latest_msg
                    

            if frame is not None and self.user_callback:
                try:
                    self.user_callback(msg, frame)
                except Exception as e:
                    print("[Callback] Error in user callback:", e)

            # small sleep to prevent tight loop
            time_wait = 0.001
            threading.Event().wait(time_wait)

        print("[Callback] Thread exiting.")

    def _callback_loop_audio_queue(self):
        """Continuously calls the user callback with the latest audio chunk. This is also offloaded into a seperate Thread"""
        print("[Callback] Thread started.")
        time_wait = None
        while self.running:
            try:
                msg = self.callback_queue.get_nowait()
                time_wait = 1.0 / int(msg.headers["chunk_ms"])
                print("queue was read")
            except queue.Empty:
                msg = None
                print("queue was empty")

            if msg is not None and self.user_callback:
                try:
                    self.user_callback(msg)
                except Exception as e:
                    print("[Callback] Error in user callback:", e)
            if time_wait is not None:
                threading.Event().wait(time_wait)
            else:
                threading.Event().wait(0.01)

        print("[Callback] Thread exiting.")
    
    async def disconnect(self):
        self.running = False
        if self.consumer and self._connected:
            if self.subscription:
                await self.subscription.unsubscribe()
            await self.consumer.close()
            self._connected = False
            print("[Consumer] Disconnected cleanly.")
        else:
            print("[Consumer] No active connection to disconnect.")

 
import time
import asyncio
import nats
import logging
from typing import Dict


class TopicActivityMonitorMulti:
    """
    Tracks which services (source_id) have sent messages into a NATS topic.
    """

    def __init__(self, topic: str = "*", broker: str = "152.53.32.66:4222", window_seconds: int = 10):
        self.topic = topic
        self.broker = broker
        self.window = window_seconds

        self._connected = False
        self.nc = None
        self.subscription = None

        # { source_id: last_timestamp }
        self.service_activity: Dict[str, float] = {}

    async def connect(self):
        if self._connected:
            return

        try:
            self.nc = await nats.connect(f"nats://{self.broker}")
            self._connected = self.nc.is_connected
            print(f"[Monitor] Connected to '{self.topic}'")

            async def handler(msg):
            
                headers = msg.headers or {}
                source_id = headers.get("source_id")

                if source_id:
                    self.service_activity[source_id] = time.time()

            self.subscription = await self.nc.subscribe(self.topic, cb=handler)

        except Exception as e:
            logging.exception("[Monitor] Connection error")
            raise e

    def get_status(self):
        """
        Returns a dict with info who is active.
        """
        now = time.time()
        result = {}
      
        for service_id, ts in self.service_activity.items():
            is_online = (now - ts) <= self.window
            result[service_id] = {
                "last_seen": ts,
                "online": is_online
            }

        return result

    async def disconnect(self):
        if self.nc and self._connected:
            await self.subscription.unsubscribe()
            await self.nc.close()
            self._connected = False
            print("[Monitor] Disconnected.")


class AudioPlayer:
    def __init__(self, samplerate=16000, channels=1, dtype="int16"):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.running = False
        self.thread = None

    def start(self, queue):
        """Start audio playback from an external queue."""
        self.queue = queue
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        stream.start()

        silence = np.zeros(int(self.samplerate * 0.05), dtype=self.dtype)

        while self.running:
            try:
                msg = self.queue.get(timeout=0.05)
                pcm = np.frombuffer(msg.data, dtype=self.dtype)
                stream.write(pcm)

            except queue.Empty:
                stream.write(silence)

        stream.stop()
        stream.close()
