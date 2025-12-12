
import asyncio
import cv2
import numpy as np

from library.input_layer import InputLayerConsumerThread



async def main():
    "You dont need to specify the Broker its automatically handeled for you"
    broker = "152.53.32.66:4222"
    
    topic = "cams.cam1" # set your topic to listen to cams.cam____
    
    """ Here we set the Consumer"""
    consumer = InputLayerConsumerThread(topic=topic, broker=broker)

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
        In this example i display the video
        You do what you want with the data form this point on
        Video Display is a feature already built in to the Producer you dont have to implememt it if you need it just use the "play_video" Prop in the consume_video
        """
        data = np.frombuffer(msg.data, np.uint8)
        
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Now frame is a numpy array you can feed to your model
        cv2.imshow(msg.subject, frame)
        cv2.waitKey(1)

    consumer.on_message(handle_frame)
    await consumer.connect() # you dont need to connect manually its done automatically but its a option if you need to or want to for style reasons
    await consumer.consume_video(play_video=False)

    await asyncio.Future()  # keep running

asyncio.run(main())
