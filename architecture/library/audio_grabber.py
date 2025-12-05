# audio_grabber.py
import sounddevice as sd
import numpy as np

class AudioGrabber:
    """
    Simple audio grabber similar to FrameGrabber.
    Captures PCM16 audio chunks at a fixed sample rate and chunk size.
    """

    def __init__(self, sample_rate=16000, channels=1, chunk_ms=100):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.frames_per_chunk = int((chunk_ms / 1000.0) * sample_rate)

        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16'
        )
        self.stream.start()

    def read_chunk(self) -> bytes:
        """
        Reads a single chunk of raw PCM16 audio and returns it as bytes.
        """
        data, overflowed = self.stream.read(self.frames_per_chunk)
        print("||| overflow paramtent",overflowed)
        return data.tobytes()

    def release(self):
        self.stream.stop()
        self.stream.close()
