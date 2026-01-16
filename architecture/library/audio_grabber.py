# audio_grabber.py
import sounddevice as sd
import numpy as np
import threading

class AudioGrabber:
    """
    Captures PCM16 audio chunks at a fixed sample rate and chunk size.
    Supports both fixed-chunk reading and continuous recording modes.
    """

    def __init__(self, sample_rate=16000, channels=1, chunk_ms=100):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.sample_per_chunk = int((chunk_ms / 1000.0) * sample_rate)

        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16' #apparently most audio libs use int 16 for this idk why just pcm magic stuff
        )
        self.stream.start()
        self._recording = False
        self._recording_buffer = []
        self._recording_thread = None
        self._recording_lock = threading.Lock()

    def read_chunk(self) -> bytes:
        """
        Reads a single chunk of raw PCM16 audio and returns it as bytes.
        """
        data, overflowed = self.stream.read(self.sample_per_chunk)
        if(overflowed):
            print("Warning: Audio buffer overflowed")
        return data.tobytes()
    
    def start_recording(self):
        """
        Start continuous recording. Audio chunks are accumulated in a buffer.
        """
        with self._recording_lock:
            if self._recording:
                return
            
            self._recording = True
            self._recording_buffer = []
            
            # Start recording thread
            self._recording_thread = threading.Thread(target=self._record_loop, daemon=True)
            self._recording_thread.start()
    
    def stop_recording(self) -> bytes:
        """
        Stop continuous recording and return all accumulated audio as bytes.
        Returns empty bytes if no audio was recorded.
        """
        with self._recording_lock:
            if not self._recording:
                return b''
            
            self._recording = False
        
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
        combined = b''.join(self._recording_buffer)
        self._recording_buffer = []
        return combined
    
    def is_recording(self) -> bool:
        """
        Check if currently recording.
        """
        with self._recording_lock:
            return self._recording
    
    def get_recording_duration_ms(self) -> int:
        """
        Get the duration of current recording in milliseconds.
        """
        with self._recording_lock:
            return len(self._recording_buffer) * self.chunk_ms
    
    def _record_loop(self):
        """
        Internal method: continuously record chunks while recording flag is set.
        """
        while True:
            with self._recording_lock:
                if not self._recording:
                    break
            chunk = self.read_chunk()
            with self._recording_lock:
                if self._recording:  # Double-check before adding
                    self._recording_buffer.append(chunk)

    def release(self):
        if self._recording:
            self.stop_recording()
        
        self.stream.stop()
        self.stream.close()
