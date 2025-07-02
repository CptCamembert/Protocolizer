"""
Satellite node application for audio publishing.

This module captures audio data and publishes it using CycloneDDS.
It handles initialization of audio input, voice activity detection,
and message publishing in a clean, maintainable structure.
"""
from utils.audio_helper import AudioHelper
from utils.vad_helper import VADModel
import numpy as np
import logging
from streamz import Stream
import requests
import time

# Constants for audio configuration
RATE = 16000
CHUNK = 512
audio_helper = AudioHelper(rate=RATE, chunk=CHUNK, input=True)
audio_source = Stream()

# Initialize VAD model
vad_model = VADModel(threshold=0.5, rate=RATE)
VAD_BEFORE = .25
VAD_AFTER = .1
VAD_FRAMES_ON = int(np.ceil(VAD_BEFORE * RATE / CHUNK))
VAD_FRAMES_OFF = -int(np.ceil(VAD_AFTER * RATE / CHUNK))

# Server URL for audio diarization
diarize_server_url = "http://localhost:8000/diarize"
ASR_WINDOW = 1
ASR_INTERVAL = .1
ASR_THRESHOLD = 0.5
ASR_ON_TIME = .5
ASR_OFF_TIME = 1
asr_process_time = ASR_INTERVAL
TOP_N_SPEAKERS = 1  # Default to top 1 speaker, can be set to -1 for all speakers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sliding window for VAD
def accumulate_vad(buffer, stream): # -> vad_np_buffer:np.ndarray
    buffer = np.concatenate([buffer, stream])
    chunks_to_keep = int(np.ceil(VAD_BEFORE * RATE / CHUNK))
    return buffer[-chunks_to_keep * CHUNK:]

# VAD processing stream
def process_vad(vad_state, np_stream): # -> is_speech:bool
    # Unpack previous state
    is_speech, vad_frames = vad_state
    
    # Split the window into chunks of CHUNK size
    frames = [np_stream[i:i+CHUNK] for i in range(0, len(np_stream), CHUNK)]
    
    # Apply VAD to each frame
    vad_results = [vad_model.is_speech(chunk) for chunk in frames]
    
    # Count speech frames (add one for speech, subtract one for non-speech)
    # Apply hysteresis to avoid rapid toggling
    for frame_is_speech in vad_results:
        if frame_is_speech:
            vad_frames = min(0 if is_speech else VAD_FRAMES_ON, vad_frames + 1)
        else:
            vad_frames = max(VAD_FRAMES_OFF if is_speech else 0, vad_frames - 1)
    if vad_frames == VAD_FRAMES_ON:
        is_speech = True
    elif vad_frames == VAD_FRAMES_OFF:
        is_speech = False

    # Return updated state and result
    #logger.info(f"VAD state: {is_speech}, vad_frames: {vad_frames}")
    
    return (is_speech, vad_frames), is_speech

# Stitch audio frames for ASR
def stitch_audio_frames(stitch_state, input_data): # -> np_stream:np.ndarray
    was_speech = stitch_state
    np_stream, vad_buffer, is_speech = input_data

    result = []
    is_end = False
    if is_speech:
        if not was_speech:
            result = vad_buffer
        else:
            result = np_stream
    else:
        if was_speech:
            is_end = True
    
    #logger.info(f"Stitching audio frames: {len(result) if result is not None else 0}, was_speech: {was_speech}, is_speech: {is_speech}")
    return is_speech, (result, is_end)

# Chunk audio frames for ASR
def chunk_audio_frames(chunk_state, input_data): # -> np_stream:np.ndarray, is_chunk:bool, is_end:bool
    global asr_process_time
    buffer, is_chunk, _ = chunk_state
    stream, is_end = input_data

    # Reset the buffer
    if is_chunk:
        buffer = []

    if len(stream) != 0:
        # Concatenate the new stream to the buffer
        buffer = np.concatenate([buffer, stream])
        chunks_to_collect = int(np.ceil(asr_process_time * RATE / CHUNK))
        is_chunk = (len(buffer) >= (chunks_to_collect * CHUNK))
    else:
        is_chunk = is_end

    return buffer, is_chunk, is_end

# Sliding Window for ASR
def accumulate_asr(asr_buffer_state, input_data): # -> asr_np_buffer:np.ndarray
    buffer, was_end = asr_buffer_state
    stream, is_end = input_data

    # Reset the buffer if the last chunk is silent
    if was_end:
        buffer = []
    
    # Concatenate the new stream to the buffer
    buffer = np.concatenate([buffer, stream])

    # Keep only the last ASR_WINDOW seconds of audio
    # Calculate the number of chunks to keep based on the ASR_WINDOW
    chunks_to_keep = int(np.ceil(ASR_WINDOW * RATE / CHUNK))
    buffer = buffer[-chunks_to_keep * CHUNK:]

    return (buffer, is_end), buffer

# diarize complete audio package
def process_asr(asr_np_buffer): # -> name:str, score:float
    global asr_process_time

    start_time = time.time()
    
    # Always send top_n parameter as a query parameter
    response = requests.post(
        diarize_server_url,
        data=asr_np_buffer.astype(np.int16).tobytes(),
        params={"top_n": TOP_N_SPEAKERS},
        headers={"Content-Type": "application/octet-stream"}
    )
    if response.status_code == 200:
        response_data = response.json()
        
        # Response will always be a dictionary with a "speakers" list
        speakers = response_data["speakers"]
        
        if speakers:  # Check if any speakers were detected
            logger.info("Top speakers:")
            for idx, speaker_data in enumerate(speakers, 1):
                name = speaker_data["speaker"]
                score = speaker_data["score"]
                logger.info(f"{idx}. Speaker: {name}, Score: {score:.2f}")
        else:
            logger.info("No speakers detected")
    else:
        logger.error(f"Error: Received status code {response.status_code}")
        logger.error(response.text)
    
    end_time = time.time() - start_time

def main():
    """Main entry point for the application."""    

    # Set up audio recording and VAD
    audio_helper.start_recording()

    # Configure audio processing pipeline:
    # Convert raw audio bytes to numpy array
    np_stream = audio_source.map(lambda stream: np.frombuffer(stream, dtype=np.int16))
    
    # Accumulate audio for VAD look-ahead/behind
    vad_np_buffer = np_stream.scan(accumulate_vad, start=[])
    
    # Apply VAD to detect speech
    vad_stream = np_stream.scan(
        process_vad,
        returns_state=True,
        start=(False, 0),
    )
    
    # Combine streams
    stitched_stream = np_stream.zip_latest(vad_np_buffer, vad_stream).scan(
        stitch_audio_frames,
        start=(False),
        returns_state=True,
    )

    # Chunk streams to only emit full chunks
    chunked_stream = stitched_stream.scan(
        chunk_audio_frames,
        start=([], False, False),
        returns_state=False,
    ).filter(lambda result: result[1]
    ).map(lambda result: (result[0], result[2]))

    # Make ASR sliding window
    asr_np_buffer = chunked_stream.scan(
        accumulate_asr,
        start=([], False),
        returns_state=True,
    )

    # Apply ASR to get best speaker
    asr_np_buffer.sink(process_asr)
    
    # Run the main audio capture loop
    logger.info("Starting Satellite Node...")
    
    try:
        while True:
            # Process raw audio data
            raw_audio_data = audio_helper.get_all()
            if raw_audio_data:
                audio_source.emit(raw_audio_data)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, exiting...")
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        audio_helper.cleanup()

if __name__ == "__main__":
    main()
