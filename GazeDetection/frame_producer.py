"""NATS Producer: Webcam → Full Frames → NATS (no face extraction)"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerProducer
from architecture.library.frame_grabber import FrameGrabber


async def run_producer():
    """Send full webcam frames to NATS"""
    producer = InputLayerProducer(
        source_name="camera1.fullframe",
        service="gaze",
        broker="152.53.32.66:4222"
    )
    
    # Use FrameGrabber to capture webcam frames)
    grabber = FrameGrabber(device=0, width=1920, height=1080, jpeg_quality=40)
    
    await producer.connect()
    
    print("Streaming full frames to NATS...")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    fps = 2  
    
    try:
        while True:
            # Send frame using FrameGrabber
            await producer.send_frame(grabber, fps)
            
            frame_count += 1
            if frame_count % 5 == 0:
                print(f"Sent {frame_count} frames")
            
            # Additional delay: 500ms between frames = 2 FPS max
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames")
    finally:
        grabber.release()
        await producer.disconnect()


if __name__ == "__main__":
    asyncio.run(run_producer())
