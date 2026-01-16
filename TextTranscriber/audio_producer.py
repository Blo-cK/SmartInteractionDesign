"""NATS Producer: Microphone -> Audio Chunks -> NATS"""
import asyncio
import sys
import os
import tkinter as tk
from tkinter import ttk
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.library.input_layer import InputLayerProducer, InputLayerMetadataSound, SampleFormat
from architecture.library.audio_grabber import AudioGrabber

# Global configuration
USE_VISUALIZATION = True


class PushToTalkUI:
    """UI Window with push-to-talk button"""
    
    def __init__(self, producer, grabber, loop):
        self.producer = producer
        self.grabber = grabber
        self.loop = loop
        self.recording = False
        self.audio_buffer = []
        
        # Setup UI
        self.root = tk.Tk()
        self.root.title("Audio Producer - Push to Talk")
        self.root.geometry("300x200")
        
        # Status label
        self.status_label = ttk.Label(
            self.root, 
            text="Ready - Press and hold button to record",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=20)
        
        # Record button
        self.record_button = tk.Button(
            self.root,
            text="Hold to Record",
            width=20,
            height=3,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            activebackground="#45a049"
        )
        self.record_button.pack(pady=20)
        
        # Bind mouse events
        self.record_button.bind("<ButtonPress-1>", self.start_recording)
        self.record_button.bind("<ButtonRelease-1>", self.stop_recording)
        
    def start_recording(self, event):
        if self.recording:
            return
            
        self.recording = True
        self.status_label.config(text="Recording...", foreground="red")
        self.record_button.config(bg="#f44336")
        self.grabber.start_recording()
        
    def stop_recording(self, event):
        if not self.recording:
            return
            
        self.recording = False
        self.status_label.config(text="Sending audio...", foreground="orange")
        self.record_button.config(bg="#4CAF50")
        combined_audio = self.grabber.stop_recording()
        if combined_audio:
            asyncio.run_coroutine_threadsafe(
                self._send_accumulated_audio(combined_audio),
                self.loop
            )
        else:
            self.status_label.config(text="No audio recorded", foreground="gray")
    
    async def _send_accumulated_audio(self, combined_audio):
        bytes_per_second = self.grabber.sample_rate * self.grabber.channels * 2
        duration_ms = int((len(combined_audio) / bytes_per_second) * 1000)
        
        # Create metadata
        metadata = InputLayerMetadataSound(
            time_stamp=str(time.time()),
            source_id=str(self.producer.source_id),
            service_id=str(self.producer.service_id),
            encoding="int16",
            sample_rate=str(self.grabber.sample_rate),
            channels=str(self.grabber.channels),
            sample_format=SampleFormat.PCM16,
            chunk_ms=str(duration_ms),
        ).as_dict()
        
        # Send to NATS
        try:
            await self.producer._send_message(combined_audio, metadata=metadata)
            self.status_label.config(
                text=f"Sent {duration_ms/1000.0:.1f}s of audio",
                foreground="green"
            )
        except Exception as e:
            self.status_label.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )
        
        # Reset status after 2 seconds
        self.root.after(2000, lambda: self.status_label.config(
            text="Ready - Press and hold button to record",
            foreground="black"
        ))
    
    def run(self):
        """Start the UI main loop"""
        self.root.mainloop()

async def run_producer_continuous(producer, grabber):
    print("Streaming full audio to NATS...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            await producer.send_audio_chunk(grabber)
    except KeyboardInterrupt:
        print("Stopped audio producer by user.")
    finally:
        grabber.release()
        await producer.disconnect()


async def run_producer_continuous_mode(): #Used for sending continous 10s chunks
    producer = InputLayerProducer(
        source_name="microphone1",
        service="audio",
        broker="152.53.32.66:4222"
    )
    
    await producer.connect()    
    grabber = AudioGrabber(chunk_ms=10000) 
    await run_producer_continuous(producer, grabber)


def run_producer_ui_mode():
    producer = InputLayerProducer(
        source_name="microphone1",
        service="audio",
        broker="152.53.32.66:4222"
    )
    
    grabber = AudioGrabber(chunk_ms=100)
    loop = asyncio.new_event_loop()
    def run_async_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    async_thread = threading.Thread(target=run_async_loop, daemon=True)
    async_thread.start()
    
    # Connect to NATS in the background loop
    future = asyncio.run_coroutine_threadsafe(producer.connect(), loop)
    future.result()  # Wait for connection

    print("Push-to-talk UI started")
    print("Press and hold the button to record, release to send")
    ui = PushToTalkUI(producer, grabber, loop)
    
    try:
        ui.run() 
    except KeyboardInterrupt:
        print("Stopped audio producer by user.")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        grabber.release()
        future = asyncio.run_coroutine_threadsafe(producer.disconnect(), loop)
        try:
            future.result(timeout=5)
        except:
            pass


if __name__ == "__main__":
    if USE_VISUALIZATION:
        run_producer_ui_mode()
    else:
        asyncio.run(run_producer_continuous_mode())
