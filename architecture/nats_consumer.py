
import asyncio
import cv2
import numpy as np

from library.input_layer import InputLayerConsumerThread



async def main():
    
    broker = "152.53.32.66:4222"
    topic = "cams.cam1"
    
    consumer = InputLayerConsumerThread(topic=topic, broker=broker)

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
        # Now frame is a numpy array you can feed to your model
        cv2.imshow(msg.subject, frame)
        cv2.waitKey(1)

    consumer.on_message(handle_frame)
    await consumer.connect()
    await consumer.consume_video()

    await asyncio.Future()  # keep running

asyncio.run(main())
