# Video Transcription App

This application transcribes video files using FFmpeg and OpenAI's Whisper model. The process is split into two steps:
1. Converting videos to audio files
2. Transcribing the audio files to text

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Sufficient disk space for model downloads

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- On macOS: `brew install ffmpeg`
- On Ubuntu/Debian: `sudo apt-get install ffmpeg`
- On Windows: Download from https://ffmpeg.org/download.html

## Usage

1. Place your video files in the `videos` directory

2. Convert videos to audio:
```bash
python convert_to_audio.py
```
This will create WAV files in the `audio` directory.

3. Transcribe the audio files:
```bash
python transcribe_audio.py
```
This will create text files in the `transcripts` directory.

## Features

- Supports multiple video formats (mp4, avi, mov, mkv)
- Uses OpenAI's Whisper model for accurate transcription
- Progress bars for tracking conversion and transcription status
- Automatic audio extraction from video files
- Skips already processed files
- Separate steps for better control and error handling

## Directory Structure

- `videos/`: Place your video files here
- `audio/`: Contains extracted audio files
- `transcripts/`: Contains the final transcriptions 