#!/bin/sh
docker run -it --rm \
	--name whisper \
	--gpus all \
    --device /dev/ \
    -v /dev/:/dev/ \
    -v /tmp/Docker/whisper/tmp:/tmp \
    -v /opt/models/whisper:/opt/models/whisper \
    -v ./whisper/opt/app:/opt/app \
    --workdir /opt/app \
    -p 8001:8001 \
    whisper-server:latest \
    python3 -m flask run --host=0.0.0.0 --port=8001
