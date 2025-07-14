"""
Satellite node application for audio publishing.

This module captures audio data and publishes it using CycloneDX.
It handles initialization of audio input, voice activity detection,
and message publishing in a clean, maintainable structure.
"""
from utils.audio_helper import AudioHelper
from utils.vad_helper import VADModel
from utils.diarization_client import DiarizationClient
from utils.whisper_client import WhisperClient
import numpy as np
import logging
from streamz import Stream
import time
from datetime import datetime

# Constants for audio configuration
RATE = 16000
CHUNK = 512
audio_helper = AudioHelper(rate=RATE, chunk=CHUNK, input=True)
audio_source = Stream()

# Initialize Diarization client
diarization_client = DiarizationClient("http://192.168.222.2:8000/diarize", top_n_speakers=-1)

# Initialize Whisper client
whisper_client = WhisperClient("http://192.168.222.2:8001/transcribe")

# Initialize VAD model
vad_model = VADModel(threshold=0.5, rate=RATE)
VAD_BEFORE = .25
VAD_AFTER = .25
VAD_FRAMES_ON = int(np.ceil(VAD_BEFORE * RATE / CHUNK))
VAD_FRAMES_OFF = -int(np.ceil(VAD_AFTER * RATE / CHUNK))

# Transcription settings
TRANSCRIPTION_MAX_LENGTH = 60.0  # Maximum seconds before forcing transcription

# Teaching settings - use the previous transcription minimum length for teaching
TEACHING_MIN_LENGTH = 10.0  # Minimum seconds of audio before teaching a speaker

ASR_WINDOW = 1
ASR_INTERVAL = .05
ASR_THRESHOLD = 0.25
ASR_ON_TIME = .5
ASR_OFF_TIME = 1
asr_process_time = ASR_INTERVAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for transcription file - will be set when main() starts
TRANSCRIPTION_FILE = None

# Global variable to track last speaker for transcription formatting
LAST_TRANSCRIPTION_SPEAKER = None

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

    return (buffer, is_end), (buffer, is_end)

# diarize complete audio package
def process_asr(buffer_bool): # -> speaker_scores:dict, is_end:bool
    global asr_process_time
    buffer, is_end = buffer_bool

    start_time = time.time()
    
    # Use the diarization client instead of making direct requests
    speakers = diarization_client.get_speakers(buffer)
    
    # Update processing time for next iteration
    asr_process_time = max(time.time() - start_time, ASR_INTERVAL)
    
    # Convert speakers list to scores dictionary
    speaker_scores = {}
    if speakers:
        for speaker_data in speakers:
            speaker_scores[speaker_data["speaker"]] = speaker_data["score"]
    
    logger.debug(f"ASR process time: {asr_process_time:.2f} seconds")

    return speaker_scores, is_end

# ASR processing function using hysteresis to get current speaker
def get_current_speaker(asr_state, speaker_scores_bool): # -> current_speaker:str
    # Unpack previous state
    current_speaker, speaker_times, is_reset = asr_state
    speaker_scores, is_end = speaker_scores_bool
    
    # Handle empty speaker scores
    if not speaker_scores:
        return (current_speaker, speaker_times, False), current_speaker

    # Add speaker names to speaker_times if they don't already exist
    for speaker in speaker_scores:
        if speaker not in speaker_times:
            speaker_times[speaker] = 0

    # Reset speaker times if ASR is reset
    if is_reset:
        current_speaker = ""
        for speaker in speaker_times:
            speaker_times[speaker] = 0

    # Find the speaker with highest score and check threshold
    max_speaker = max(speaker_scores, key=speaker_scores.get)
    max_score = speaker_scores[max_speaker]
    if max_score < ASR_THRESHOLD:
        max_speaker = "Unknown"

    # Increase count of max_speaker by one, decrease count of others by one
    for speaker in speaker_times.keys():
        # Code to Activate the speaker
        if max_speaker == speaker and (not speaker == "Unknown" or current_speaker in ("", "Unknown")):
            speaker_times[speaker] = min(0 if current_speaker == speaker else ASR_ON_TIME, speaker_times[speaker] + asr_process_time)
            if speaker_times[speaker] == ASR_ON_TIME:
                current_speaker = speaker
        # Code to Deactivate the speaker
        else:
            speaker_times[speaker] = max(-ASR_OFF_TIME if current_speaker == speaker else 0, speaker_times[speaker] - asr_process_time)
            if speaker_times[speaker] == -ASR_OFF_TIME:
                current_speaker = "Unknown"

    # Pack state
    asr_state = (current_speaker, speaker_times, is_end)

    # Return updated state and result
    return asr_state, current_speaker

# Display speaker information with bar chart
def display_speaker_info(speaker_data):
    # Unpack the data - speaker_data is now ((speaker_scores, is_end), current_speaker)
    speaker_scores_with_end, current_speaker = speaker_data
    speaker_scores, is_end = speaker_scores_with_end
    
    if not speaker_scores:
        return

        # Generate bar chart
    chart_lines = [""]
    chart_lines.append(f"Current Speaker: {current_speaker}")
    bar_width = 100  # Reduced width for cleaner output
    bar_max = 1
    
    # Create threshold indicator line
    threshold_pos = int(ASR_THRESHOLD * bar_width / bar_max)
    threshold_line = "Threshold: " + " " * (10 + threshold_pos) + "| " + str(ASR_THRESHOLD)
    chart_lines.append(threshold_line)
    
    # Sort speakers alphabetically by name
    for speaker in sorted(speaker_scores.keys()):
        score = speaker_scores[speaker]
        bar_len = int(abs(score) / bar_max * bar_width)
        bar_char = "█" if speaker == current_speaker else "▒"
        bar = bar_char * min(bar_len, bar_width) + " " * (bar_width - bar_len) + "|" # Limit bar length
        
        # Mark the current speaker
        chart_lines.append(f"  {speaker:<10}: {score:>5.2f} |{bar}")
    
    chart_lines.append("-" * 50)
    logger.info("\n".join(chart_lines))

# Accumulate audio by speaker for transcription
def accumulate_speaker_audio(transcription_state, input_data): # -> audio_buffer:np.ndarray, should_transcribe:bool
    """
    Accumulate audio from the current speaker and determine when to trigger transcription.
    Triggers transcription when:
    1. Speaker changes (with ASR_ON_TIME backtrack to avoid hysteresis delay)
    2. Speech segment ends (is_end flag from VAD/chunking pipeline)
    3. Maximum length reached
    
    Special handling for speaker detection:
    - Audio recorded while current_speaker is "" gets transferred to the newly detected speaker
    - "Unknown" is treated like any other speaker
    """
    # Unpack state: (current_buffer, active_speaker, buffer_start_time)
    current_buffer, active_speaker, buffer_start_time = transcription_state
    
    # Unpack input: (audio_chunk, current_speaker, is_end)
    audio_chunk, current_speaker, is_end = input_data
    
    # Initialize variables
    should_transcribe = False
    transcribe_buffer = np.array([], dtype=np.int16)
    transcribe_speaker = active_speaker
    
    # Check if we should start a new buffer (new speaker or after transcription)
    if active_speaker != current_speaker and current_speaker != "":
        # Case 1: Speaker changed from one speaker to another (including Unknown)
        if active_speaker != "":
            # Normal speaker change - transcribe previous buffer with ASR_ON_TIME removed from end
            if len(current_buffer) > 0:
                samples_to_remove = int((ASR_ON_TIME + ASR_WINDOW / 2) * RATE) # 1.5x to account for potential overlap
                if len(current_buffer) > samples_to_remove:
                    transcribe_buffer = current_buffer[:-samples_to_remove].copy()
                    # Start new buffer with the last ASR_ON_TIME seconds + new audio chunk
                    last_overlap = current_buffer[-samples_to_remove:]
                    current_buffer = np.concatenate([last_overlap, audio_chunk]) if len(audio_chunk) > 0 else last_overlap
                else:
                    transcribe_buffer = current_buffer.copy()
                    current_buffer = audio_chunk.copy() if len(audio_chunk) > 0 else np.array([], dtype=np.int16)
                
                should_transcribe = True
                transcribe_speaker = active_speaker
            else:
                current_buffer = audio_chunk.copy() if len(audio_chunk) > 0 else np.array([], dtype=np.int16)
        else:
            # Case 2: Speaker detected after being "" - transfer accumulated audio to new speaker
            # Keep the existing buffer and add new audio - no transcription yet
            if len(audio_chunk) > 0:
                current_buffer = np.concatenate([current_buffer, audio_chunk])
        
        active_speaker = current_speaker
        buffer_start_time = time.time()
        
    elif active_speaker == "" and current_speaker == "":
        # Case 3: Still in unknown speaker state - keep accumulating
        if len(audio_chunk) > 0:
            current_buffer = np.concatenate([current_buffer, audio_chunk])
        
    elif is_end:
        # Case 4: Speech ended - transcribe current buffer if it has content
        if len(current_buffer) > 0:
            should_transcribe = True
            transcribe_buffer = current_buffer.copy()
            transcribe_speaker = active_speaker
        
        # Clear buffer when speech ends
        current_buffer = np.array([], dtype=np.int16)
        active_speaker = ""
        buffer_start_time = time.time()
        
    else:
        # Case 5: Same speaker, continue accumulating
        if len(audio_chunk) > 0:
            current_buffer = np.concatenate([current_buffer, audio_chunk])
        
        # Check if we've reached maximum length
        buffer_duration = len(current_buffer) / RATE
        if buffer_duration >= TRANSCRIPTION_MAX_LENGTH:
            should_transcribe = True
            transcribe_buffer = current_buffer.copy()
            transcribe_speaker = active_speaker
            # Keep only the last 5 seconds for continuation
            overlap_samples = int(5.0 * RATE)
            current_buffer = current_buffer[-overlap_samples:] if len(current_buffer) > overlap_samples else current_buffer
            buffer_start_time = time.time() - 5.0
    
    # Update state
    new_state = (current_buffer, active_speaker, buffer_start_time)
    
    # Return state and transcription data
    return new_state, (transcribe_buffer, transcribe_speaker, should_transcribe)

# Process transcription requests
def process_transcription(transcription_data):
    """
    Process transcription for accumulated audio buffer.
    Also triggers teaching if audio is long enough.
    """
    audio_buffer, speaker, should_transcribe = transcription_data
    
    if not should_transcribe or len(audio_buffer) == 0:
        return
    
    duration = len(audio_buffer) / RATE
    
    # Convert numpy int16 array directly to the format expected by Whisper server
    # This matches the format used in the working test_whisper.py
    audio_data = audio_buffer.astype(np.float32) / 32768.0
    audio_bytes = audio_data.tobytes()
    
    # WhisperClient expects a list of audio chunks, so we wrap the bytes in a list
    audio_chunks = [audio_bytes]
    
    # Get transcription from Whisper
    transcription = whisper_client.get_transcript(audio_chunks)

    if transcription:    

        # Also trigger teaching if audio is long enough and speaker is known
        if duration >= TEACHING_MIN_LENGTH and speaker not in ("", "Unknown"):
            try:
                diarization_client.teach_speaker(speaker, audio_buffer)
            except Exception as e:
                logger.error(f"Error during teaching for {speaker}: {e}")

        # Write to transcription pipe with separate parameters
        write_to_transcription_pipe(speaker, transcription.strip())

        # Save audio into a file (date_time_speakername.wav)
        timestamp = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
        filename = f"temp/{timestamp}_{speaker}.wav"
        AudioHelper.save_buffer_to_file(audio_buffer, filename, rate=RATE)
        
        # Clean up old audio files, keeping only the most recent 10
        AudioHelper.cleanup_temp_files("temp", max_files=10)
        
        return {"speaker": speaker, "text": transcription, "duration": duration}
    else:
        logger.warning(f"Failed to get transcription for {speaker}")
        return None

def write_to_transcription_pipe(speaker, text):
    """Write transcription messages to a log file with simple name and indented text formatting"""
    try:
        import os
        from datetime import datetime
        
        # Use the global transcription file that was set in main()
        global TRANSCRIPTION_FILE, LAST_TRANSCRIPTION_SPEAKER
        transcription_file = TRANSCRIPTION_FILE or "transcription.log"  # Fallback if not set
        
        # Check if this is the same speaker as last time
        if speaker == LAST_TRANSCRIPTION_SPEAKER:
            # Same speaker - just add indented text on a new line
            formatted_message = f"{text}"
        else:
            # New speaker - add speaker name on its own line, then indented text
            formatted_message = f"\n[{speaker}]:\n{text}"
            LAST_TRANSCRIPTION_SPEAKER = speaker
        
        # Append message to file
        with open(transcription_file, 'a', encoding='utf-8') as f:
            f.write(f"{formatted_message}\n")
            f.flush()
            
    except Exception as e:
        # Don't let transcription file errors crash the main application
        pass

def main():
    """Main entry point for the application."""    

    global TRANSCRIPTION_FILE
    
    # Initialize transcription file with timestamp - back to log format
    TRANSCRIPTION_FILE = datetime.now().strftime("transcription_%d-%m-%y_%H-%M.log")
    logger.info(f"Transcription file: {TRANSCRIPTION_FILE}")

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
    ).map(lambda result: (result[0], result[2])
    )

    # Make ASR sliding window
    asr_np_buffer = chunked_stream.scan(
        accumulate_asr,
        start=([], False),
        returns_state=True,
    )

    # Apply ASR to get speaker scores (asr_np_buffer already contains the is_end flag)
    speaker_scores_stream = asr_np_buffer.map(process_asr)

    # Apply hysteresis to get current speaker
    current_speaker_stream = speaker_scores_stream.scan(
        get_current_speaker,
        start=("", {"Unknown": 0}, False),
        returns_state=True,
    )

    # Combine speaker scores and current speaker for display
    speaker_info_stream = speaker_scores_stream.zip_latest(current_speaker_stream)
    
    # Display speaker information
    speaker_info_stream.sink(display_speaker_info)

    # === TRANSCRIPTION PIPELINE ===
    
    # Combine audio chunks with current speaker and is_end flag for transcription processing
    # We need to ensure the is_end flag from chunked_stream is properly propagated
    transcription_input_stream = chunked_stream.zip_latest(current_speaker_stream).map(
        lambda data: (data[0][0], data[1], data[0][1])  # (audio_chunk, current_speaker, is_end)
    )
    
    # Accumulate audio by speaker and trigger transcriptions (and teaching when appropriate)
    transcription_trigger_stream = transcription_input_stream.scan(
        accumulate_speaker_audio,
        start=(np.array([], dtype=np.int16), "", time.time()),
        returns_state=True,
    )
    
    # Process transcriptions when triggered (includes teaching for long enough audio)
    transcription_trigger_stream.filter(
        lambda transcription_data: transcription_data[2]  # should_transcribe
    ).sink(process_transcription)

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
