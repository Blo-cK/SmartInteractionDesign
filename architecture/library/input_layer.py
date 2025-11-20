from abc import ABC
from dataclasses import dataclass, asdict
import asyncio
import threading
import time
import nats
import numpy as np
from typing import Callable, Literal, Optional
import cv2
import threading

from .frame_grabber import FrameGrabber

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
    sample_format: Literal["pcm16", "float32", "opus"]

class InputLayerProducer:
    def __init__(self, topic:str, source_name:str, broker:str = "152.53.32.66:4222"):
        self.broker = broker
        self.topic = topic
        self._connected= False
        self.producer = None
        self.id= source_name
        
    async def connect(self):
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
        if not self._connected:
            await self.connect()    
        try:
            await self.producer.publish(self.topic,data, headers=metadata)
            print('Message sent successfully')
        except Exception as e:
            print(f"Error sending message: {e}")
    
    #TODO: Implement sending Audio Chunks
    async def send_audio_chunk():
        return
    
    async def send_frame(self, frame_grabber:FrameGrabber, fps=30):
        """Capture a frame from FrameGrabber and send to NATS"""
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





class InputLayerConsumer:
    def __init__(self, topic:str, broker:str= "152.53.32.66:4222"):
        self.broker= broker
        self.topic= topic
        self._connected= False
        self.consumer= None
        self.subscription = None
        
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
                    await onFrame(msg) # Callback every Gruppe can write their own Callback fucntion so we have decoupled the functionality
            except Exception as e:
                print("Error while consuming", e)    
                
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
    """

    def __init__(self, topic:str, broker:str= "152.53.32.66:4222"):
        self.broker = broker
        self.topic = topic
        self._connected = False
        self.consumer = None
        self.subscription = None

        self.latest_frame = None
        self.latest_msg = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Optional user callback
        self.user_callback: Optional[Callable[[np.ndarray], None]] = None
        self.callback_thread: Optional[threading.Thread] = None

    async def connect(self):
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

    async def consume_video(self):
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
        display_thread = threading.Thread(target=self._display_loop, daemon=True)
        display_thread.start()

        # Start callback thread if provided
        if self.user_callback:
            self.callback_thread = threading.Thread(target=self._callback_loop, daemon=True)
            self.callback_thread.start()

        print("[Consumer] Started zero-delay display & callback threads.")
        # Keep coroutine alive
        while self.running:
            await asyncio.sleep(0.01)

    def _display_loop(self):
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
        """Continuously calls the user callback with the latest frame."""
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
 