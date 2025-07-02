import silero_vad
import numpy as np
import torch

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

class VADModel():

    def __init__(self, threshold=0.5, rate=16000):
        self.threshold = threshold
        self.rate = rate
        self.vad = silero_vad.load_silero_vad()

    def is_speech(self, frame) -> bool:
        # Convert audio data for VAD processing
        audio_int16 = np.frombuffer(frame, np.int16)
        audio_float32 = int2float(audio_int16)
        
        # Process audio in 512 sample chunks for VAD
        confidence = self.vad(torch.from_numpy(audio_float32), self.rate).item()
        is_speech = confidence >= self.threshold
        return is_speech