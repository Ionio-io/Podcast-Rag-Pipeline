import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
from transcribe import transcribe_audio
from extract_audio import extract_audio
import warnings
from tqdm import tqdm
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def prepare_video(video_path, output_dir="videos"):
    """
    Prepare video file for processing
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the video
    
    Returns:
        str: Path to the video file in the output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename from path
    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, filename)
    
    # If the file is not already in the output directory, copy it
    if os.path.abspath(video_path) != os.path.abspath(output_path):
        logger.info(f"Copying video to {output_dir}...")
        try:
            shutil.copy2(video_path, output_path)
            logger.info(f"Video copied to {output_path}")
        except Exception as e:
            logger.error(f"Error copying video: {str(e)}")
            sys.exit(1)
    
    return output_path

def process_video(video_path, with_diarization=True):
    """
    Process a video file through the entire pipeline
    
    Args:
        video_path (str): Path to the video file
        with_diarization (bool): Whether to perform speaker diarization
    """
    # Step 1: Extract audio
    logger.info("Step 1: Extracting audio from video...")
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
    audio_path = os.path.join(audio_dir, audio_filename)
    
    extract_audio(video_path, audio_path)
    
    # Step 2: Transcribe audio
    logger.info("Step 2: Transcribing audio...")
    
    # Create a progress bar for the entire process
    with tqdm(total=100, desc="Overall Progress", position=0) as pbar:
        # Transcription progress (100% for simple, 40% for with diarization)
        transcription_weight = 100 if not with_diarization else 40
        pbar.set_description("Transcribing")
        transcribe_audio(
            audio_path, 
            with_diarization=with_diarization, 
            progress_callback=lambda x: pbar.update(transcription_weight * x / 100)
        )
        
        # Diarization progress (60% of total)
        if with_diarization:
            pbar.set_description("Diarizing")
            # Update to 100% when diarization is complete
            pbar.update(60)
        
        pbar.set_description("Complete")

def main():
    parser = argparse.ArgumentParser(description="Process video file to transcription")
    parser.add_argument("video_path", help="Path to the video file to process")
    parser.add_argument("--simple", action="store_true", help="Perform simple transcription without speaker diarization")
    parser.add_argument("--output-dir", default="videos", help="Directory to save processed videos")
    
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        if not os.path.exists(args.video_path):
            logger.error(f"Error: File {args.video_path} does not exist")
            sys.exit(1)
        
        # Prepare video
        video_path = prepare_video(args.video_path, args.output_dir)
        
        # Process video through pipeline
        process_video(video_path, not args.simple)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 