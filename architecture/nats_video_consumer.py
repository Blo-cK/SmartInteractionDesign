
import asyncio
import cv2
import numpy as np

from library.input_layer import InputLayerConsumerThread



async def main():
    "You dont need to specify the Broker its automatically handeled for you"
    broker = "152.53.32.66:4222"
    
    source_name= "stream1"
    service_id = "example_serviceL"
    
    """ Here we set the Consumer"""
    consumer = InputLayerConsumerThread(source_name = source_name, service= service_id, broker=broker)

    """"
    This is the user defind Callback 
    You can name this however you want just add the two parameters to it
    msg is the whole message Object in the following you can see what you have acess to
     
    """
    def handle_frame(msg, frames):
        
        """ print("Subject:", msg.subject)
        print("Reply:", msg.reply)
        print("Subscription ID:", msg.sid)
        print("Timestamp:", msg.timestamp)
        print("Headers:", msg.headers)
        print("Subject parts:", msg.subject_parts)
        print("Data length:", len(msg.data)) """
        print("Headers:", msg.headers)
        
        ##############################
        #   YOUR CODE GOES HERE
        ##############################
        """ 
        Video Display is a feature already built in to the Producer you dont have to implememt it if you need it just use the "play_video" Prop in the consume_video
        """

    consumer.on_message(handle_frame)
    await consumer.connect() # you dont need to connect manually its done automatically but its a option if you need to or want to for style reasons
    await consumer.consume_video(play_video=True)

    await asyncio.Future()  # keep running this is needed bc the script would end here  
    # the api will work as long as the main is running (so the consume will stop as soon as this terminates/returns) 
    # if you dont want this to happen then keep it running forever or put it in a coroutine or thread if needed

asyncio.run(main())
