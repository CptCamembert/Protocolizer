import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)

class WhisperClient:
	
    def __init__(self, url):
        self.url = url

    def get_transcript(self, audio_buffer):
        # The audio_buffer is already properly formatted as float32 bytes from process_transcription
        # Just join the chunks - no need for additional conversion
        audio_bytes = b''.join(audio_buffer)

        try:
            response = requests.post(
                self.url,
                headers = {'Content-Type': 'application/octet-stream'},
                data = audio_bytes
            )
            
            if response.status_code == 200:
                text = response.json()['text']
                logger.debug(f"text: {text}")
                return text

            else:
                logger.error(f"Error: {response.status_code}, {response.text}")
                return None
            
        except Exception as e:
            logger.error(f"Error making Whisper request: {e}")
            return None