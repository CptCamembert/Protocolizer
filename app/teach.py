import sys
import numpy as np
from utils.diarization_client import DiarizationClient

# Constants for audio configuration
RATE = 16000
CHUNK = 512

# Initialize diarization client
diarization_client = DiarizationClient("http://192.168.222.2:8000/diarize")

def record_audio(duration, sample_rate=RATE):
    """
    Record audio for a given duration and return audio data.
    
    Args:
        duration (int): Duration in seconds to record audio.
        sample_rate (int, optional): Sample rate for recording. Defaults to 16000.
    
    Returns:
        np.ndarray: Recorded audio data as a NumPy array.
    """
    import pyaudio
    # Open audio stream
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    # Read audio data from the stream
    frames = []
    print(f"Recording for {duration} seconds...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    # Close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert the audio data to a NumPy array
    audio_data = b''.join(frames)
    return np.frombuffer(audio_data, dtype=np.int16)

if __name__ == "__main__":
    # Ensure speaker name is provided as command-line argument
    name = ""
    if len(sys.argv) < 2:
        name = input("Please enter a name: ")
    else:
        name = sys.argv[1]
    if name == "":
        sys.exit(1)

    # Record audio for 15 seconds
    audio_data = record_audio(duration=15)
    
    # Use DiarizationClient to teach the speaker
    success = diarization_client.teach_speaker(name, audio_data)
    
    if success:
        print(f"✅ Successfully taught speaker: {name}")
    else:
        print(f"❌ Failed to teach speaker: {name}")
        sys.exit(1)