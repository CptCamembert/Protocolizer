import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)

class DiarizationClient:
    
    def __init__(self, url, top_n_speakers=1):
        self.url = url
        self.top_n_speakers = top_n_speakers
        # Extract base URL for teaching endpoint
        self.teach_url = url.replace('/diarize', '/diarize_teach')

    def get_speakers(self, audio_buffer):
        """
        Send audio buffer to diarization server and get speaker identification results.
        
        Args:
            audio_buffer (np.ndarray): Audio data as numpy array
            
        Returns:
            list: List of speaker dictionaries with 'speaker' and 'score' keys,
                  or empty list if no speakers detected or error occurred
        """
        try:
            # Convert audio data to int16 format for server
            audio_data = audio_buffer.astype(np.int16).tobytes()
            
            response = requests.post(
                self.url,
                data=audio_data,
                params={"top_n": self.top_n_speakers},
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if response.status_code == 200:
                response_data = response.json()
                speakers = response_data.get("speakers", [])
                
                if speakers:
                    logger.debug("Top speakers:")
                    for idx, speaker_data in enumerate(speakers, 1):
                        name = speaker_data["speaker"]
                        score = speaker_data["score"]
                        logger.debug(f"{idx}. Speaker: {name}, Score: {score:.2f}")
                else:
                    logger.debug("No speakers detected")
                
                return speakers
            else:
                logger.error(f"Error: Received status code {response.status_code}")
                logger.error(response.text)
                return []
                
        except Exception as e:
            logger.error(f"Error making diarization request: {e}")
            return []

    def teach_speaker(self, name, audio_data):
        """
        Teach the system a new speaker by sending audio data to the diarization teach endpoint.
        
        Args:
            name (str): The name of the speaker to teach.
            audio_data (np.ndarray): Audio data as int16 numpy array.
        
        Returns:
            bool: True if teaching was successful, False otherwise.
        """
        try:
            # Send the audio to the teach endpoint
            response = requests.post(
                self.teach_url,
                data=audio_data.astype(np.int16).tobytes(),
                params={"name": name},
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if response.status_code == 200:
                logger.debug(f"Successfully taught speaker: {name}")
                return True
            else:
                logger.error(f"Failed to teach speaker {name}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error teaching speaker {name}: {e}")
            return False