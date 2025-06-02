import os
import ffmpeg
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_audio(video_path, audio_path):
    """Extract audio from video file using ffmpeg."""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path)
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        return False

def extract_audio_from_folder(input_folder, output_folder="audio"):
    """
    Extract audio from all video files in a specified folder.
    
    Args:
        input_folder (str): Path to the folder containing video files
        output_folder (str): Path to the folder where audio files will be saved (default: "audio")
    """
    # Create necessary directories if they don't exist
    Path(input_folder).mkdir(exist_ok=True)
    Path(output_folder).mkdir(exist_ok=True)
    
    # Get list of video files
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in the '{input_folder}' directory.")
        return
    
    print(f"Found {len(video_files)} video files to process in '{input_folder}'.")
    
    # Process each video file
    for video_file in tqdm(video_files, desc="Converting videos to audio"):
        video_path = os.path.join(input_folder, video_file)
        audio_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.wav")
        
        # Skip if audio already exists
        if os.path.exists(audio_path):
            print(f"Audio already exists for {video_file}, skipping...")
            continue
        
        print(f"\nProcessing {video_file}...")
        
        # Extract audio
        print("Extracting audio...")
        if extract_audio(video_path, audio_path):
            print(f"Audio saved to {audio_path}")
        else:
            print(f"Failed to extract audio from {video_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract audio from video files in a specified folder")
    parser.add_argument("--input-folder", "-i", type=str, default="downloaded_videos", 
                       help="Path to the folder containing video files (default: downloaded_videos)")
    parser.add_argument("--output-folder", "-o", type=str, default="audio",
                       help="Path to the folder where audio files will be saved (default: audio)")
    
    args = parser.parse_args()
    
    extract_audio_from_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main() 