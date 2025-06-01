import os
import whisper
from pyannote.audio import Pipeline
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add a small epsilon to prevent division by zero
EPSILON = 1e-8

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def transcribe_with_speakers(audio_path, output_path):
    """Transcribe audio with speaker diarization."""
    try:
        # Get Hugging Face token from environment
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
        
        # Load Whisper model
        print("\n[1/4] Loading Whisper model...")
        start_time = time.time()
        whisper_model = whisper.load_model("base")
        print(f"✓ Whisper model loaded in {time.time() - start_time:.1f} seconds")
        
        # Load Pyannote pipeline
        print("\n[2/4] Loading Pyannote pipeline...")
        start_time = time.time()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print(f"✓ Pyannote pipeline loaded in {time.time() - start_time:.1f} seconds")
        
        # Perform diarization
        print("\n[3/4] Performing speaker diarization...")
        start_time = time.time()
        diarization = pipeline(audio_path)
        print(f"✓ Speaker diarization completed in {time.time() - start_time:.1f} seconds")
        
        # Transcribe the entire audio
        print("\n[4/4] Transcribing audio...")
        start_time = time.time()
        result = whisper_model.transcribe(audio_path, word_timestamps=True)
        print(f"✓ Audio transcription completed in {time.time() - start_time:.1f} seconds")
        
        # Process and combine diarization with transcription
        print("\nProcessing and combining results...")
        start_time = time.time()
        
        # Create a progress bar for processing segments
        total_segments = len(list(diarization.itertracks(yield_label=True)))
        with tqdm(total=total_segments, desc="Processing segments", unit="segment") as pbar:
            with open(output_path, "w", encoding="utf-8") as f:
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    start = segment.start
                    end = segment.end
                    speaker_text = []
                    
                    # Match Whisper words to speaker segments
                    for word in result["segments"]:
                        word_start = word["start"]
                        word_end = word["end"]
                        if start <= word_start <= end:
                            speaker_text.append(word["text"])
                    
                    # Write to file
                    timestamp = f"[{format_timestamp(start)} - {format_timestamp(end)}]"
                    speaker_label = f"Speaker {speaker}"
                    text = " ".join(speaker_text)
                    f.write(f"{timestamp} {speaker_label}: {text}\n")
                    pbar.update(1)
        
        print(f"✓ Results combined and saved in {time.time() - start_time:.1f} seconds")
        print(f"\n✓ Transcription with speaker diarization saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {str(e)}")
        return False

def main():
    # Create necessary directories if they don't exist
    Path("audio").mkdir(exist_ok=True)
    Path("transcripts").mkdir(exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir("audio") if f.endswith('.wav')]
    
    if not audio_files:
        print("No audio files found in the 'audio' directory.")
        return
    
    print(f"Found {len(audio_files)} audio files to process.")
    
    # Process each audio file with a progress bar
    for audio_file in tqdm(audio_files, desc="Processing files", unit="file"):
        audio_path = os.path.join("audio", audio_file)
        transcript_path = os.path.join("transcripts", f"{os.path.splitext(audio_file)[0]}_with_speakers.txt")
        
        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            print(f"\nSkipping {audio_file} - transcript already exists")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {audio_file}")
        print(f"{'='*50}")
        transcribe_with_speakers(audio_path, transcript_path)
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main() 