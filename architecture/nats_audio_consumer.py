
import asyncio
import numpy as np


import sounddevice as sd

from library.input_layer import InputLayerConsumerThread
from library.input_layer import InputResultWrapper



async def main():
    
    broker = "152.53.32.66:4222"
    source_name= "stream1"
    service_id = "example_serviceL"
    
    consumer = InputLayerConsumerThread(source_name = source_name, service= service_id, broker=broker)
        
    def handle_audio(msg: InputResultWrapper):
        """this is the user defined callback"""
        ##############################
        #   YOUR CODE GOES HERE
        ##############################
        print("Headers:", msg.headers)
        
    consumer.on_message(handle_audio)
    await consumer.connect() # you dont need to connect manually its done automatically but its a option if you need to or want to for style reasons
    await consumer.consume_audio(play_audio=True) 
    
    await asyncio.Future()  # keep running this is needed bc the script would end here  
    # the api will work as long as the main is running (so the consume will stop as soon as this terminates/returns) 
    # if you dont want this to happen then keep it running forever or put it in a coroutine or thread if needed
    
    
asyncio.run(main())
