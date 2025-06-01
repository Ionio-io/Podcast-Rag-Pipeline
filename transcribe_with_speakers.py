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

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=round(seconds)))

def transcribe_audio(audio_path, output_dir=DEFAULT_TRANSCRIPTS_DIR, with_diarization=True):
    """
    Transcribe audio with optional speaker diarization
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save transcripts
        with_diarization (bool): Whether to perform speaker diarization
    
    Returns:
        str: Path to the output transcript file
    """
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    logger.info(f"Processing {base_name}...")
    
    # Transcribe with Whisper
    logger.info("Transcribing audio...")
    if WHISPER_PROMPT:
        logger.info(f"Using custom prompt: {WHISPER_PROMPT}")
        result = whisper_model.transcribe(audio_path, initial_prompt=WHISPER_PROMPT)
    else:
        logger.info("No custom prompt provided, using default transcription")
        result = whisper_model.transcribe(audio_path)
    
    if not with_diarization:
        # Save simple transcript without speaker information
        final_transcript = [
            {
                "start": format_timestamp(segment["start"]),
                "end": format_timestamp(segment["end"]),
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]
    else:
        # Perform speaker diarization
        logger.info("Performing speaker diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        diarization = pipeline(audio_path)
        
        # Convert diarization to a more usable format
        speaker_segments = [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        
        # Merge transcription with speaker information
        logger.info("Merging transcription with speaker information...")
        final_transcript = []
        
        for segment in result["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find overlapping speaker segments
            overlapping_speakers = [
                s for s in speaker_segments
                if (s["start"] <= segment_end and s["end"] >= segment_start)
            ]
            
            # If multiple speakers overlap, use the one with the most overlap
            if overlapping_speakers:
                best_speaker = max(
                    overlapping_speakers,
                    key=lambda s: min(s["end"], segment_end) - max(s["start"], segment_start)
                )
                speaker = best_speaker["speaker"]
            else:
                speaker = "UNKNOWN"
            
            final_transcript.append({
                "start": format_timestamp(segment_start),
                "end": format_timestamp(segment_end),
                "speaker": speaker,
                "text": segment["text"].strip()
            })
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the transcript
    output_suffix = "with_speakers" if with_diarization else "simple"
    output_file = os.path.join(output_dir, f"{base_name}_{output_suffix}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_transcript, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transcript saved to {output_file}")
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
        transcribe_audio(audio_path, output_dir, with_diarization)
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Error: Directory {args.input_dir} does not exist")
        logger.error(f"Please ensure the directory exists and contains audio files")
        sys.exit(1)
    
    process_audio_folder(args.input_dir, args.output_dir, not args.simple) 