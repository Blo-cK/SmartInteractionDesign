
import asyncio
import cv2
import numpy as np


import sounddevice as sd

from library.input_layer import InputLayerConsumerThread
from library.input_layer import AudioPlayer


async def main():
    
    broker = "152.53.32.66:4222"
    topic = "cams.cam1"
    audio_topic = "input.audio-stream"
    
    consumer = InputLayerConsumerThread(topic=audio_topic, broker=broker)

    def handle_frame(msg, frames):
        
        """ print("Subject:", msg.subject)
        print("Reply:", msg.reply)
        print("Subscription ID:", msg.sid)
        print("Timestamp:", msg.timestamp)
        print("Headers:", msg.headers)
        print("Subject parts:", msg.subject_parts)
        print("Data length:", len(msg.data)) """
        print("Headers:", msg.headers)
        data = np.frombuffer(msg.data, np.uint8)
        
        
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Now frame is a numpy array you can feed to your model or share with other teams
        cv2.imshow(msg.subject, frame)
        cv2.waitKey(1)
        
    def handle_audio(msg):
        print("Headers:", msg.headers)
        #print("queue item ", msg)
        # Convert byte data to int16 numpy array
        # Set sample rate (adjust if your audio stream uses a different rate) deprecated dont use this ever again
        #sample_rate = 16000
        
        # Play audio (non-blocking)
        #sd.play(audio, samplerate=sample_rate)
        #sd.wait()
        
        # Optional: block until this chunk finishes playing
        # sd.wait()
        
    consumer.on_message(handle_audio)
    await consumer.connect()
    #await consumer.consume_video()
    asyncio.create_task(consumer.consume_audio())
    print(",hjvljhvljhvljhvljhvjlhgvljhv ")
    player = AudioPlayer()
    player.start(consumer.shared_aduio_queue)
  
    
    
    await asyncio.Future()  # keep running
    
    
asyncio.run(main())
