import pyaudio
import numpy as np
from queue import Queue
import wave
import io
import time
import threading
import os
import glob
import logging

logger = logging.getLogger(__name__)

class AudioHelper:
    def __init__(self, rate=16000, chunk=512, channels=1, format=pyaudio.paInt16, input=False, output=False):
        self.RATE = rate
        self.CHUNK = chunk
        self.CHANNELS = channels
        self.FORMAT = format

        self.input = input
        self.output = output
        assert self.input or self.output, "Either input or output must be enabled"
        
        self.audio_queue = Queue()
        self.audio = pyaudio.PyAudio()
        
        # Flags to control the continuous recording and playback threads
        self.is_playing = False
        self.is_recording = False
        self.playback_thread = None
        self.recording_thread = None
        
        # Setup input/output streams based on configuration
        self._setup_streams()

    def _setup_streams(self):
        """Set up audio streams based on input/output configuration."""
        if self.input:
            self.input_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                #input_device_index=0
            )
        
        if self.output:
            self.output_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True,
                frames_per_buffer=self.CHUNK
            )

    def start_recording(self):
        """Start a background thread that continuously records audio into the queue."""
        assert self.input, "Input stream is not enabled"
        if self.recording_thread is None or not self.recording_thread.is_alive():
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._continuous_recording_worker)
            self.recording_thread.daemon = True
            self.recording_thread.start()
    
    def stop_recording(self):
        """Stop the continuous recording thread."""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
    
    def _continuous_recording_worker(self):
        """Worker thread that continuously records audio into the queue."""
        while self.is_recording:
            try:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                time.sleep(0.1)  # Sleep briefly before retrying

    def get_frame(self, blocking=True):
        """Get the latest audio frame from the queue.
        
        If the queue is very full, it will be cleared to prevent lag
        between recorded audio and processed audio.
        """
        assert self.input, "Input stream is not enabled"
        # Only get latest chunk to prevent processing backlog
        if len(self.audio_queue.queue) > self.RATE/self.CHUNK:
            # Too much data in queue, clear it to prevent lag
            logger.debug(f"Audio queue too full ({len(self.audio_queue.queue)} frames), clearing...")
            self.audio_queue.queue.clear()

        frames = None
        if not self.audio_queue.empty():
            frames = self.audio_queue.get()
        if frames:
            frames
        if blocking:
            return self.audio_queue.get()
    
    def get_all(self, blocking=True):
        """Get all audio frames from the queue and clear it."""
        assert self.input, "Input stream is not enabled"
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())        
        if frames:
            return b''.join(frames)
        if blocking:
            return self.audio_queue.get()

    
    def put_all(self, audio_data):
        """Put audio data into the queue for playback."""
        self.audio_queue.put(audio_data)
    
    def start_continuous_playback(self):
        """Start a background thread that continuously plays audio from the queue."""
        assert self.output, "Output stream is not enabled"
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._continuous_playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def stop_continuous_playback(self):
        """Stop the continuous playback thread."""
        self.is_playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def _continuous_playback_worker(self):
        """Worker thread that continuously plays audio from the queue."""
        while self.is_playing:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                try:
                    if isinstance(audio_data, bytes):
                        # Directly write raw PCM data to the output stream
                        self.output_stream.write(audio_data)
                except Exception as e:
                    logger.error(f"Error playing audio from queue: {e}")
            else:
                # Sleep a bit to prevent CPU overuse when queue is empty
                time.sleep(0.01)

    def cleanup(self):
        """Clean up all resources used by the AudioHelper."""
        # Stop background threads
        self.stop_continuous_playback()
        self.stop_recording()
        
        # Close streams
        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        # Terminate PyAudio instance
        self.audio.terminate()

    @staticmethod
    def save_buffer_to_file(audio_buffer, filename, rate=16000, channels=1, sample_width=2):
        """
        Save audio buffer to a WAV file.
        
        Args:
            audio_buffer (np.ndarray): Audio data as int16 numpy array
            filename (str): Path to save the WAV file
            rate (int): Sample rate, defaults to 16000
            channels (int): Number of channels, defaults to 1 (mono)
            sample_width (int): Sample width in bytes, defaults to 2 (16-bit)
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(rate)
                wav_file.writeframes(audio_buffer.astype(np.int16).tobytes())
            
            logger.debug(f"Audio saved to {filename} ({len(audio_buffer)} samples, {len(audio_buffer)/rate:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to save audio to {filename}: {e}")
    
    @staticmethod
    def cleanup_temp_files(temp_dir="temp", max_files=10):
        """
        Keep only the most recent audio files in the temp directory.
        
        Args:
            temp_dir (str): Directory containing temporary audio files
            max_files (int): Maximum number of files to keep (default: 10)
        """
        try:
            # Get all .wav files in the temp directory
            pattern = os.path.join(temp_dir, "*.wav")
            wav_files = glob.glob(pattern)
            
            if len(wav_files) <= max_files:
                return  # No cleanup needed
            
            # Sort files by modification time (newest first)
            wav_files.sort(key=os.path.getmtime, reverse=True)
            
            # Keep only the most recent max_files, delete the rest
            files_to_delete = wav_files[max_files:]
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed old audio file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
            
            if files_to_delete:
                logger.debug(f"🧹 Cleaned up {len(files_to_delete)} old audio files, keeping {len(wav_files) - len(files_to_delete)} most recent")
                
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")