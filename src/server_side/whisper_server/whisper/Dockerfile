FROM python:3.10-slim

RUN apt-get update -y && \
	apt-get install -y \
	ffmpeg \
	git \
	&& \
    apt-get autoremove -y -qq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

COPY requirements.txt requirements.txt
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt && \
	rm requirements.txt

RUN pip3 install "git+https://github.com/openai/whisper.git"
