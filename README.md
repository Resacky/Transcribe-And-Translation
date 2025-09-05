# Transcribe-And-Translation
This repository will be used as a POC WIP for containerizing and developing local AI models for Transcription and Translation.

# Quick Start:
    Create a virtual enviornment of python 3.13.~ and then run `pip install -r requirements.txt`. For getting openai-whisper set up and configured you need to install `ffmpeg` which is a command-line tool. You can install it based on these commands:
    ```
    # on Ubuntu or Debian
    sudo apt update && sudo apt install ffmpeg

    # on Arch Linux
    sudo pacman -S ffmpeg

    # on MacOS using Homebrew (https://brew.sh/)
    brew install ffmpeg

    # on Windows using Chocolatey (https://chocolatey.org/)
    choco install ffmpeg

    # on Windows using Scoop (https://scoop.sh/)
    scoop install ffmpeg
    ```
    ref: https://github.com/openai/whisper

    Afterwards, you will probably need to set up the env variables for that PATH.