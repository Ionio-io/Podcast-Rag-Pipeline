# Video Transcription App

This application transcribes video files using FFmpeg and OpenAI's Whisper model. The process is split into two steps:
1. Converting videos to audio files
2. Transcribing the audio files to text with optional speaker diarization

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Sufficient disk space for model downloads
- Hugging Face account with access to Pyannote.audio models

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

4. Set up environment variables:
Create a `.env` file in the project root with:
```env
# Hugging Face token for Pyannote.audio
HF_TOKEN=your_huggingface_token_here

# Optional: Whisper prompt for better transcription
WHISPER_PROMPT="This is a conversation involving Rohan from Ionio. The discussion includes topics about LLMs, Large Language Models, Generative AI, and artificial intelligence. Common terms include: Rohan, Ionio, LLMs, GenAI, artificial intelligence, machine learning, deep learning, transformers, and neural networks."
```

## Usage

### Quick Start (Recommended)
Process a video file directly:
```bash
# Process video with speaker diarization
python main.py /path/to/your/video.mp4

# Process video with simple transcription
python main.py /path/to/your/video.mp4 --simple

# Specify custom output directory
python main.py /path/to/your/video.mp4 --output-dir custom/videos
```

### Manual Processing
If you prefer to process videos manually:

1. Place your video files in the `videos` directory

2. Extract audio from videos:
```bash
python extract_audio.py
```
This will create WAV files in the `audio` directory.

3. Transcribe the audio files:
```bash
# Simple transcription without speaker diarization
python transcribe_simple.py

# Full transcription with speaker diarization
python transcribe.py

# Additional options for transcription with speakers:
python transcribe.py --simple  # Force simple transcription
python transcribe.py --input-dir custom/input --output-dir custom/output  # Custom directories
```

## Features

- Supports multiple video formats (mp4, avi, mov, mkv)
- Uses OpenAI's Whisper model for accurate transcription
- Optional speaker diarization using Pyannote.audio
- Custom prompts for better transcription accuracy
- Progress bars for tracking conversion and transcription status
- Automatic audio extraction from video files
- Skips already processed files
- Separate steps for better control and error handling
- JSON output format for easy parsing
- Configurable input/output directories
- Clean logging without unnecessary warnings
- One-command processing of local video files

## Directory Structure

- `videos/`: Place your video files here
- `audio/`: Contains extracted audio files
- `transcripts/`: Contains the final transcriptions in JSON format
  - `*_simple.json`: Simple transcription without speaker information
  - `*_with_speakers.json`: Transcription with speaker diarization

## Scripts

- `main.py`: Main script for processing video files
- `extract_audio.py`: Extracts audio from video files
- `transcribe_simple.py`: Simple transcription without speaker diarization
- `transcribe.py`: Main transcription script with speaker diarization support

## Output Format

The transcription is saved in JSON format with the following structure:

For simple transcription:
```json
[
  {
    "start": "00:00:00",
    "end": "00:00:05",
    "text": "Transcribed text here"
  }
]
```

For transcription with speakers:
```json
[
  {
    "start": "00:00:00",
    "end": "00:00:05",
    "speaker": "SPEAKER_1",
    "text": "Transcribed text here"
  }
]
``` 