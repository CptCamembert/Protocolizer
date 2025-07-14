from flask import Flask, abort, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import whisper

WHISPER_MODEL_NAME = "/opt/models/whisper/base.pt"
WHISPER_MODEL_LANGUAGE = "en"

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model(WHISPER_MODEL_NAME, device = DEVICE)

app = Flask(__name__)
CORS(app)

# ************************************************************************************
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not request.data:
        abort(400, description="No audio data received")

    try:
        audio_data = np.frombuffer(request.data, dtype=np.float32)

        result = model.transcribe(	audio_data,
                                    language = WHISPER_MODEL_LANGUAGE,
                                    fp16 = True)

        return jsonify({'text': result['text']})

    except Exception as e:
        abort(500, description=str(e))

# ************************************************************************************
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000)
