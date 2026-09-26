# Real-Time Urdu Speech-to-Text

Live microphone transcription in Urdu using Faster-Whisper and a local CTranslate2 model. The scripts use CPU inference by default (`int8`).

## Requirements

- Python 3.10 or newer
- A working microphone
- Internet access for installing packages and downloading the model
- About 1 GB of free disk space for the model

## Setup

Run these commands from the project directory.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or continue with the environment's Python directly: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### Ubuntu / Debian / WSL2

Install PortAudio development files before installing PyAudio:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

Install PortAudio, then create the environment and install Python dependencies:

```bash
brew install portaudio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download the model

The scripts expect the model at `models/whisper-large-v3-turbo-urdu-ct2`. If its model files are not present, download them after activating the virtual environment:

```bash
python download_model.py
```

The model is large and is ignored by Git; it only needs to be downloaded once. The transcription scripts load it locally and do not download it at startup.

## Run

Activate `.venv` first, then start either real-time transcription implementation:

```bash
python realtime_urdu_stt.py
```

Or run one of the alternate implementations:

```bash
python claude_code.py
python new.py
python backup.py
```

Speak in Urdu; transcription is printed after a pause. Press `Ctrl+C` to stop. `realtime_urdu_stt.py` uses WebRTC voice activity detection; the other scripts use an audio-level threshold.

## CPU and GPU

CPU (`int8`) is configured by default. In a script, set `DEVICE = "cuda"` and choose a supported `COMPUTE_TYPE` such as `"float16"` to use a compatible NVIDIA GPU. GPU inference also requires a compatible CUDA/cuDNN runtime and CTranslate2 setup.