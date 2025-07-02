FROM python:3.10-slim

# Install system dependencies for audio processing and compilation
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV DIARIZATION_SERVER_URL=http://host.docker.internal:8000

# Expose any ports if needed (for health checks)
EXPOSE 8001

# Command to run the application
CMD ["python", "app/main.py"]