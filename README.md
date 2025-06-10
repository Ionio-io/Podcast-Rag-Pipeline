# TranscribeRohan - Video Transcription & RAG System

A complete video processing pipeline that downloads YouTube videos, transcribes them using OpenAI's Whisper, and provides a RAG (Retrieval-Augmented Generation) interface using OpenAI's Responses API for querying transcripts.

## Features

- 📺 **YouTube Video Download**: Download videos from YouTube channels or individual URLs
- 🎵 **Audio Extraction**: Convert videos to high-quality audio files
- 📝 **AI Transcription**: Transcribe audio using OpenAI's Whisper with speaker diarization
- 🤖 **RAG System**: Query transcripts using OpenAI's Responses API with file search
- 🌐 **Web Interface**: Streamlit-based chat interface for interacting with transcripts
- 📊 **Progress Tracking**: Real-time progress bars for all operations

## Setup

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

4. **Set up environment variables:**
Create a `.env` file in the project root:
```env
# Hugging Face token for speaker diarization
HF_TOKEN=your_huggingface_token_here

# OpenAI API key for RAG system
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom Whisper prompt
WHISPER_PROMPT="This is a conversation involving topics about AI, machine learning, and technology."
```

## Usage

### 1. Download YouTube Videos

```bash
# Download from a channel
python download_youtube.py --channel CHANNEL_ID

# Download single video
python download_youtube.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Download with custom output directory
python download_youtube.py --channel CHANNEL_ID --output-dir my_videos
```

### 2. Transcribe Videos

```bash
# Transcribe all videos in the videos directory
python transcribe.py

# Transcribe specific audio file
python transcribe.py --input path/to/audio.wav

# Simple transcription without speaker diarization
python transcribe.py --simple
```

### 3. Run RAG System

```bash
# Start the Streamlit web interface
streamlit run app.py
```

Then open your browser to the displayed URL (usually http://localhost:8501) to:
- Upload transcript files to OpenAI's file search
- Chat with your transcripts using natural language
- Get source-cited responses from your video content

## Project Structure

```
TranscribeRohan/
├── setup.sh                 # Quick setup script  
├── app.py                   # Streamlit RAG interface
├── rag_system.py            # OpenAI Responses API integration
├── transcribe.py            # Main transcription script
├── extract_audio.py         # Audio extraction utilities
├── download_youtube.py      # YouTube downloader
├── requirements.txt         # All dependencies
├── .env                     # Environment variables (create this)
├── transcripts/             # Generated transcript files (JSON)*
├── audio/                   # Extracted audio files (WAV)*
├── videos/                  # Local video files*
└── downloaded_videos/       # YouTube downloads*
```

*_Directories are created automatically when needed_

## Output Formats

### Transcripts
Saved as JSON files in `transcripts/` directory:

**Simple transcription:**
```json
[
  {
    "start": "00:00:00",
    "end": "00:00:05", 
    "text": "Transcribed text here"
  }
]
```

**With speaker diarization:**
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

## Dependencies

- **Core**: Python 3.8+, FFmpeg
- **AI Models**: OpenAI Whisper, Pyannote.audio for speaker diarization
- **APIs**: OpenAI API for RAG functionality
- **Web**: Streamlit for user interface
- **Media**: yt-dlp for YouTube downloads, ffmpeg-python for audio processing

## Notes

- **Automatic Directory Creation**: All necessary directories (`transcripts/`, `audio/`, `videos/`, `downloaded_videos/`) are created automatically when running scripts or setup
- Large media files (videos/audio) are excluded from git by default
- Transcript JSON files in `transcripts/` are preserved in git
- The RAG system uses OpenAI's latest Responses API with file search capabilities
- Speaker diarization requires a Hugging Face account and token

## Troubleshooting

- **SSL Certificate Issues**: The app handles SSL certificate verification automatically
- **Memory Issues**: For large files, consider processing videos in smaller batches
- **API Rate Limits**: OpenAI API calls are automatically rate-limited
- **File Upload Limits**: OpenAI has file size limits for uploaded documents 