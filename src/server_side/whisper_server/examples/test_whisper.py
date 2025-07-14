import argparse
import numpy as np
import requests
import wave

URL_WHISPER = "http://192.168.222.2:8001/transcribe"

def get_transcript(audio_data):
	response = requests.post(
		URL_WHISPER,
		headers = {'Content-Type': 'application/octet-stream'},
		data = audio_data
	)

	if response.status_code == 200:
		text = response.json()['text']
		print(f"text: {text}")
		return text

	else:
		print(f"Error: {response.status_code}, {response.text}")
		return None

# ************************************************************************************
def main():
	with wave.open(path_wav, "rb") as file:
		frames = file.readframes(file.getnframes())
		sample_width = file.getsampwidth()
		audio_data = np.frombuffer(frames, dtype = np.int16)
		audio_data = audio_data.astype(np.float32) / 32768.0
		audio_data = audio_data.tobytes()
		get_transcript(audio_data)

# ************************************************************************************
parser = argparse.ArgumentParser(description="Send WAV file to an endpoint.")
parser.add_argument("path_wav", type=str, help="Path to the WAV file")

args = parser.parse_args()
path_wav = args.path_wav

if __name__ == "__main__":
	main()
