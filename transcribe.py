import os
import whisper
import torch
from pyannote.audio import Pipeline
from tqdm import tqdm
import json
from datetime import timedelta
from dotenv import load_dotenv
import warnings
import logging
from pathlib import Path
import ssl
import urllib.request

# IMPORTANT: Fix SSL certificate issues BEFORE importing anything else
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("SSL certificate verification disabled globally")
except Exception as e:
    print(f"Warning: Could not disable SSL verification: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Default directories
DEFAULT_AUDIO_DIR = "audio"
DEFAULT_TRANSCRIPTS_DIR = "transcripts"

# Get Whisper prompt from environment variable
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "")

def fix_ssl_certificate():
    """Fix SSL certificate issues for macOS"""
    try:
        # Create unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.info("SSL certificate verification disabled for model downloads")
    except Exception as e:
        logger.warning(f"Could not disable SSL verification: {e}")

def load_whisper_model_with_retry(model_name="medium", max_retries=3):
    """
    Load Whisper model with SSL error handling and retries
    
    Args:
        model_name (str): Name of the Whisper model to load
        max_retries (int): Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading Whisper model '{model_name}' (attempt {attempt + 1}/{max_retries})...")
            model = whisper.load_model(model_name)
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in str(e):
                logger.warning(f"SSL certificate error on attempt {attempt + 1}: {e}")
                if attempt == 0:
                    # Try fixing SSL on first failure
                    fix_ssl_certificate()
                    continue
                elif attempt < max_retries - 1:
                    # Try a smaller model if SSL issues persist
                    if model_name == "medium":
                        logger.info("Trying smaller 'small' model due to SSL issues...")
                        model_name = "small"
                        continue
                    elif model_name == "small":
                        logger.info("Trying 'base' model due to SSL issues...")
                        model_name = "base"
                        continue
            else:
                logger.error(f"Error loading model: {e}")
            
            if attempt == max_retries - 1:
                logger.error("Failed to load Whisper model after all retries")
                raise e

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=round(seconds)))

def transcribe_audio(audio_path, with_diarization=True, progress_callback=None, model_size="medium"):
    """
    Transcribe audio file using Whisper and optionally perform speaker diarization
    
    Args:
        audio_path (str): Path to the audio file
        with_diarization (bool): Whether to perform speaker diarization
        progress_callback (callable): Optional callback function to report progress (0-100)
        model_size (str): Size of the Whisper model to use ("base", "small", "medium", "large")
    """
    # Create transcripts directory if it doesn't exist
    os.makedirs("transcripts", exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Load Whisper model
    logger.info(f"Loading Whisper model ({model_size})...")
    model = load_whisper_model_with_retry(model_name=model_size)
    
    # Transcribe audio
    logger.info("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        language="en",
        task="transcribe",
        fp16=torch.cuda.is_available()
    )
    
    if progress_callback:
        progress_callback(50)  # Transcription is 50% complete
    
    # Process transcription results
    segments = []
    for segment in result["segments"]:
        segment_data = {
            "start": format_timestamp(segment["start"]),
            "end": format_timestamp(segment["end"]),
            "text": segment["text"].strip(),
            "speaker": "Unknown"  # Default speaker
        }
        segments.append(segment_data)
    
    if with_diarization:
        # Perform speaker diarization
        logger.info("Performing speaker diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        
        # Get speaker diarization results
        diarization = pipeline(audio_path)
        
        if progress_callback:
            progress_callback(75)  # Diarization is 75% complete
        
        # Match speakers to segments
        for segment in segments:
            start_time = float(segment["start"].split(":")[-1])
            end_time = float(segment["end"].split(":")[-1])
            
            # Find the most common speaker in this time range
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= end_time and turn.end >= start_time:
                    speakers.append(speaker)
            
            if speakers:
                # Use the most common speaker
                segment["speaker"] = max(set(speakers), key=speakers.count)
        
        if progress_callback:
            progress_callback(90)  # Speaker matching is 90% complete
    
    # Save results
    output_file = os.path.join("transcripts", f"{base_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    if progress_callback:
        progress_callback(100)  # Process is complete
    
    logger.info(f"Transcription saved to {output_file}")
    return output_file

def process_audio_folder(input_dir=DEFAULT_AUDIO_DIR, output_dir=DEFAULT_TRANSCRIPTS_DIR, with_diarization=True):
    """
    Process all audio files in the input directory
    
    Args:
        input_dir (str): Directory containing audio files
        output_dir (str): Directory to save transcripts
        with_diarization (bool): Whether to perform speaker diarization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    
    if not audio_files:
        logger.info(f"No audio files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing files"):
        audio_path = os.path.join(input_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        output_suffix = "with_speakers" if with_diarization else "simple"
        output_file = os.path.join(output_dir, f"{base_name}_{output_suffix}.json")
        
        # Skip if transcript already exists
        if os.path.exists(output_file):
            logger.info(f"\nSkipping {audio_file} - transcript already exists")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {audio_file}")
        logger.info(f"{'='*50}")
        transcribe_audio(audio_path, with_diarization)
        logger.info(f"{'='*50}\n")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio files with optional speaker diarization")
    parser.add_argument("--input-dir", default=DEFAULT_AUDIO_DIR,
                      help=f"Input directory containing audio files (default: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_TRANSCRIPTS_DIR,
                      help=f"Output directory for transcripts (default: {DEFAULT_TRANSCRIPTS_DIR})")
    parser.add_argument("--simple", action="store_true",
                      help="Perform simple transcription without speaker diarization")
    parser.add_argument("--file", type=str,
                      help="Transcribe a single audio file instead of processing a directory")
    
    args = parser.parse_args()
    
    # Handle single file transcription
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"Error: File {args.file} does not exist")
            sys.exit(1)
        
        logger.info(f"Transcribing single file: {args.file}")
        transcribe_audio(args.file, not args.simple)
        sys.exit(0)
    
    # Handle directory processing
    if not os.path.exists(args.input_dir):
        logger.error(f"Error: Directory {args.input_dir} does not exist")
        logger.error(f"Please ensure the directory exists and contains audio files")
        sys.exit(1)
    
    process_audio_folder(args.input_dir, args.output_dir, not args.simple) 