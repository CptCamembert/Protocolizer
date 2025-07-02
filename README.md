# Protocolizer - Real-Time Audio Processing Client

A real-time audio processing client application that captures live audio, performs voice activity detection, identifies speakers using diarization, and provides intelligent transcription with speaker attribution.

## 🎯 Features

### 🎤 **Core Audio Processing**
- Live audio capture from microphone (16kHz, mono)
- Voice Activity Detection (VAD) using Silero VAD
- Real-time audio streaming with intelligent buffering
- Hysteresis-based speech detection to prevent rapid toggling

### 👥 **Speaker Identification**
- Real-time speaker diarization and recognition
- Support for multiple speakers with confidence scoring
- Visual speaker confidence displays with bar charts
- Hysteresis-based speaker tracking (prevents rapid switching)
- Automatic speaker teaching from long audio segments

### 📝 **Intelligent Transcription**
- Speaker-attributed transcription using Whisper
- Automatic transcription triggers on:
  - Speaker changes
  - Speech segment endings  
  - Maximum length thresholds
- Contextual audio accumulation for coherent transcripts
- Audio file saving with speaker attribution

### 🔄 **Processing Pipeline**
```
Audio Input → VAD → Speaker Detection → Audio Accumulation → Transcription → Attributed Results
```

## 🏗️ System Architecture

The protocolizer works as a client that connects to two backend servers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Protocolizer  │    │  Diarization    │    │   Whisper       │
│     (Client)    │◄──►│    Server       │    │   Server        │
│                 │    │                 │    │                 │
│  • Audio Capture│    │  • Speaker ID   │    │  • Transcription│
│  • VAD          │    │  • Embeddings   │    │  • ASR          │
│  • Processing   │    │  • Recognition  │    │  • Text Output  │
│  • Coordination │    │  • Teaching     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Microphone/audio input device
- Access to diarization server (default: http://192.168.222.2:8000)
- Access to whisper server (default: http://192.168.222.2:8001)

### Server Configuration
Update the server URLs in `app/main_copy.py` if your servers are running elsewhere:
```python
# Initialize Diarization client
diarization_client = DiarizationClient("http://YOUR_DIARIZATION_SERVER:8000/diarize", top_n_speakers=-1)

# Initialize Whisper client
whisper_client = WhisperClient("http://YOUR_WHISPER_SERVER:8001/transcribe")
```

### Installation & Running

**Option A: Using Docker (Recommended)**
```bash
./docker/build.sh
./docker/up.sh
```

**Option B: Using Python directly**
```bash
pip install -r requirements.txt
./run.sh
```

### Monitor Real-Time Output

**For Docker:**
```bash
docker logs -f protocolizer
```

**For direct Python:**
The application shows real-time output directly in the terminal.

## 📊 Real-Time Output Examples

### Speaker Identification Display
```
Current Speaker: Ana
Threshold:           | 0.5
→ Ana        :  0.87 |████████████████████████████████████████
  Ben        :  0.23 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  Unknown    : -0.15 |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
--------------------------------------------------
```

### Transcription Output
```
📝 [Ana]: Hello, how are you doing today?
🎓 [Ben]: I'm doing great, thanks for asking! How about you?
📝 [Ana]: I'm wonderful, thank you. Let's discuss the project.
```

**Transcription Icons:**
- 📝 Regular transcription
- 🎓 Transcription with automatic speaker teaching (15+ seconds)
- ❌ Transcription with failed teaching attempt

## ⚙️ Configuration

### Audio Settings
```python
RATE = 16000              # Sample rate (Hz)
CHUNK = 512              # Audio chunk size (samples)
VAD_BEFORE = 0.25        # VAD lookahead (seconds)
VAD_AFTER = 0.25         # VAD lookbehind (seconds)
```

### Speaker Recognition
```python
ASR_THRESHOLD = 0.25     # Minimum confidence for speaker recognition
ASR_ON_TIME = 0.5        # Time to activate speaker (seconds)
ASR_OFF_TIME = 1.0       # Time to deactivate speaker (seconds)
ASR_WINDOW = 1.0         # Speaker analysis window (seconds)
```

### Transcription Settings
```python
TRANSCRIPTION_MAX_LENGTH = 60.0   # Force transcription threshold (seconds)
TEACHING_MIN_LENGTH = 15.0        # Minimum audio for automatic teaching
```

## 🎓 Teaching New Speakers

**Manual Teaching:**
```bash
python app/teach.py "Speaker Name"
```

**Automatic Teaching:**
The system automatically teaches speakers when:
- Audio segment is ≥ 15 seconds
- Speaker is identified (not "Unknown")
- Transcription is successful

## 📁 Project Structure

```
protocolizer/
├── app/
│   ├── main.py           # Basic speaker identification
│   ├── main_copy.py      # Full transcription system (recommended)
│   └── teach.py          # Speaker teaching utility
├── utils/
│   ├── audio_helper.py   # Audio capture and processing
│   ├── diarization_client.py  # Diarization API client
│   ├── vad_helper.py     # Voice activity detection
│   └── whisper_client.py # Transcription API client
├── docker/               # Docker deployment scripts
│   ├── build.sh
│   ├── up.sh
│   └── down.sh
├── temp/                 # Temporary audio files
├── requirements.txt      # Python dependencies
├── run.sh               # Direct Python execution script
├── Dockerfile           # Docker container definition
└── README.md            # This file
```

## 🔧 Dependencies

### System Requirements
- Python 3.10+
- Audio device access (microphone)
- Network access to diarization and whisper servers

### Python Packages
- **pyaudio**: Audio capture
- **numpy**: Audio processing
- **streamz**: Real-time streaming framework
- **requests**: HTTP client for server communication
- **torch**: Deep learning framework
- **silero-vad**: Voice activity detection

## 🐛 Troubleshooting

### No Transcriptions Appearing
1. Check Whisper server: `curl http://YOUR_WHISPER_SERVER:8001/health`
2. Verify audio segments are ≥ 2 seconds
3. Check speaker detection confidence > threshold
4. Monitor logs for "Processing transcription" messages

### Speaker Detection Issues  
1. Check diarization server: `curl http://YOUR_DIARIZATION_SERVER:8000/health`
2. Verify trained speaker embeddings exist on server
3. Ensure clear microphone input
4. Consider lowering ASR_THRESHOLD

### Audio Pipeline Problems
1. Verify microphone permissions and device access
2. Check VAD model initialization 
3. For Docker: ensure `--device /dev/snd` access
4. Restart PulseAudio: `pulseaudio --kill && pulseaudio --start`

## 📈 Performance Characteristics

- **Latency**: < 1 second for speaker detection
- **Transcription Delay**: 2-5 seconds after speech ends
- **CPU Usage**: Moderate (real-time processing)
- **Memory Usage**: ~500MB-1GB (includes VAD model)
- **Network**: Continuous requests to both servers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Backend Services

This client requires two backend services:
- **Diarization Server**: For speaker identification and teaching
- **Whisper Server**: For speech-to-text transcription

Contact the project maintainer for access to compatible server implementations.

---

**Built for real-time meeting protocols, interviews, and multi-speaker conversations.**