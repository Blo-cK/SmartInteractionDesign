
import asyncio
import numpy as np


import sounddevice as sd

from library.input_layer import InputLayerConsumerThread
from library.input_layer import AudioPlayer


async def main():
    
    broker = "152.53.32.66:4222"
    audio_topic = "audio.stream1"
    
    consumer = InputLayerConsumerThread(topic=audio_topic, broker=broker)
        
    def handle_audio(msg):
        """this is the user defined callback"""
        print("Headers:", msg.headers)
        
    consumer.on_message(handle_audio)
    await consumer.connect()
    #asyncio.create_task(consumer.consume_audio())
    await consumer.consume_audio(play_audio=True)
    
    #player = AudioPlayer()
    #player.start(consumer.shared_aduio_queue)
  
    await asyncio.Future()  # keep running
    
    
asyncio.run(main())
