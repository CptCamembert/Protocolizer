# AI-Server Whisper

This is the repo for snippets and tests of the AI Server. You are on the Whisper branch, which is dedicated to [OpenAI Whipser](https://openai.com/index/whisper/).

The branch contains scripts to build and run the docker container as well as examples to use Whisper.

## Downloading
The following steps will get you to the branch:
```shell
cd ~/workspace
git clone https://gitlab.aviate-it.com/wiweb/ai-server.git ai-server_whisper
cd ai-server_whisper
git checkout whisper
```

## docker
This folder contains the scripts to build and run the docker container for Whisper.

### build.sh
This Script builds the docker image for Whisper. It is meant to be started from the ai-server_whisper folder.

> **NOTE:** Unless you do substantial changes on the code or have problems with the docker image, there should be no need to build it on the Server as it should already be there.

> **NOTE:** Even on this powerful server, building the containers can take quite an ammount of time of more than 3 minutes.

```shell
cd ~/workspace/ai-server_whisper
sh ./docker/build.sh
```

### up.sh
This script starts the docker container, which is then serving Whisper on port 8001.

```shell
cd ~/workspace/ai-server_whisper
sh ./docker/up.sh ~
```

> **NOTE:** Make sure to use the ~ at the end to run the script in the background, otherwise you will need to start another terminal to proceed with the example and also the container would stop if you closed the terminal when not using the ~.

### down.sh
In order to stop the container, use the `down.sh` script.

```shell
cd ~/workspace/ai-server_whisper
sh ./docker/down.sh ~
```

## examples
This folder contains an example to use the whisper endpoint with python.

### Creating / courcing the virtual environment
In order to run the example, you need to create a Python Virtual Environment (venv) and install the requirements.

> **NOTE:** You only need to create the virtual environment and install the requirements once. Afterwards you can just source it with the `source ./venv_whisper/bin/activate` command.

```shell
cd ~/workspace/ai-server_whisper/examples
python -m venv venv_whisper
source ./venv_whisper/bin/activate
pip install -r requirements
```

You can now run the whisper example like this:
```shell
python test_whisper.py female.wav
```
This should result in the transcription of the wav file.

To get some insights on the execution time, you can use the keyword 'time' when executing the script.
<img src=resources/time_whisper.png
        alt="Time for execution of the whisper script"
        style="display: block; margin: 0 auto"
/>

> **NOTE:** The wav file needs to be of the following format:
> - channels 1
> - format int16
> - data rate 16k

## Deactivate the virtual environment
In order to deactivate the virtual environment either close the shell window or use `deactivate`.
