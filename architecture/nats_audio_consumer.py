
import asyncio
import numpy as np


import sounddevice as sd

from library.input_layer import InputLayerConsumerThread
from library.input_layer import InputResultWrapper



async def main():
    
    broker = "152.53.32.66:4222"
    audio_topic = "audio.stream1"
    
    consumer = InputLayerConsumerThread(topic=audio_topic, broker=broker)
        
    def handle_audio(msg: InputResultWrapper):
        """this is the user defined callback"""
        print("Headers:", msg.headers)
        
    consumer.on_message(handle_audio)
    await consumer.connect()
    await consumer.consume_audio(play_audio=True)
  
    await asyncio.Future()  # keep running
    
    
asyncio.run(main())
