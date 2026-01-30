import asyncio
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architecture.library.input_layer import InputLayerConsumerThread, InputResultWrapper
from architecture.library.output_layer import OutputLayerProducer

async def main():
    broker = "152.53.32.66:4222"
    source_name = "stream1"
    service_id = "environment_loudness"
    
    output_producer = OutputLayerProducer()
    loop = asyncio.get_running_loop()
    
    previous_loudness = [-1] 
    
    consumer = InputLayerConsumerThread(source_name=source_name, service=service_id, broker=broker)
        
    def handle_audio(msg: InputResultWrapper):
        audio_data = np.frombuffer(msg.data, dtype=np.int16)
        
        if len(audio_data) > 0:
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            
            # Scale from 1 to 10
            loudness_scale = int(np.clip(rms / 500, 1, 10))
            
            # Only send if value has changed
            if loudness_scale != previous_loudness[0]:
                previous_loudness[0] = loudness_scale
                
                result = {
                    "loudness_scaled": loudness_scale
                }

                msg.msg = msg 

                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(
                        output_producer.sendData(
                            input_result=msg, 
                            result=result, 
                            service_id="environment_loudness"
                        )
                    )
                )
                
                bar = "!" * loudness_scale
                print(f"Update: Level {loudness_scale} {bar} (RMS: {rms:.2f})")
        
    consumer.on_message(handle_audio)
    await consumer.connect()
    
    print("Audio Consumer aktiv - Sende nur Änderungen an das Dashboard...")
    await consumer.consume_audio(play_audio=True) 
    
    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeendet.")